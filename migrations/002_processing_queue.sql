-- Processing Queue for Pull-Based Architecture
-- Run this migration in your Supabase SQL editor after 001_ai_tables.sql

-- ============================================
-- Processing Queue
-- Jobs that local AI workers pull from
-- ============================================
CREATE TABLE IF NOT EXISTS processing_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, processing, completed, failed
    operations TEXT[] NOT NULL DEFAULT ARRAY['embedding', 'objects', 'faces'],
    priority INTEGER NOT NULL DEFAULT 5,  -- 1=highest, 10=lowest
    worker_id TEXT,  -- Claims ownership when processing
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    result JSONB,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    -- Prevent duplicate pending jobs for same asset
    CONSTRAINT unique_pending_asset UNIQUE (asset_id) DEFERRABLE INITIALLY DEFERRED
);

-- Indexes for efficient queue operations
CREATE INDEX IF NOT EXISTS idx_processing_queue_status
ON processing_queue(status) WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS idx_processing_queue_priority
ON processing_queue(priority, created_at) WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS idx_processing_queue_user
ON processing_queue(user_id);

CREATE INDEX IF NOT EXISTS idx_processing_queue_asset
ON processing_queue(asset_id);

CREATE INDEX IF NOT EXISTS idx_processing_queue_worker
ON processing_queue(worker_id) WHERE status = 'processing';

-- Enable RLS
ALTER TABLE processing_queue ENABLE ROW LEVEL SECURITY;

-- Service role can do everything (for local worker)
CREATE POLICY "Service role full access to processing_queue"
ON processing_queue FOR ALL
USING (true)
WITH CHECK (true);

-- Users can view their own jobs
CREATE POLICY "Users can view own processing_queue"
ON processing_queue FOR SELECT
USING (auth.uid() = user_id);

-- ============================================
-- Function to claim a job atomically
-- Returns the claimed job or NULL if none available
-- ============================================
CREATE OR REPLACE FUNCTION claim_processing_job(
    p_worker_id TEXT,
    p_max_attempts INTEGER DEFAULT 3
)
RETURNS TABLE (
    id UUID,
    asset_id UUID,
    user_id UUID,
    operations TEXT[],
    attempts INTEGER
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_job_id UUID;
BEGIN
    -- Atomically claim the highest priority pending job
    UPDATE processing_queue pq
    SET
        status = 'processing',
        worker_id = p_worker_id,
        started_at = NOW(),
        attempts = pq.attempts + 1
    WHERE pq.id = (
        SELECT pq2.id
        FROM processing_queue pq2
        WHERE pq2.status = 'pending'
          AND pq2.attempts < p_max_attempts
        ORDER BY pq2.priority ASC, pq2.created_at ASC
        LIMIT 1
        FOR UPDATE SKIP LOCKED
    )
    RETURNING pq.id INTO v_job_id;

    -- Return the claimed job details
    RETURN QUERY
    SELECT
        pq.id,
        pq.asset_id,
        pq.user_id,
        pq.operations,
        pq.attempts
    FROM processing_queue pq
    WHERE pq.id = v_job_id;
END;
$$;

-- ============================================
-- Function to complete a job
-- ============================================
CREATE OR REPLACE FUNCTION complete_processing_job(
    p_job_id UUID,
    p_result JSONB DEFAULT NULL
)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE processing_queue
    SET
        status = 'completed',
        result = p_result,
        completed_at = NOW()
    WHERE id = p_job_id AND status = 'processing';

    RETURN FOUND;
END;
$$;

-- ============================================
-- Function to fail a job
-- ============================================
CREATE OR REPLACE FUNCTION fail_processing_job(
    p_job_id UUID,
    p_error TEXT,
    p_max_attempts INTEGER DEFAULT 3
)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    v_attempts INTEGER;
BEGIN
    -- Get current attempts
    SELECT attempts INTO v_attempts
    FROM processing_queue
    WHERE id = p_job_id;

    IF v_attempts >= p_max_attempts THEN
        -- Max attempts reached, mark as failed permanently
        UPDATE processing_queue
        SET
            status = 'failed',
            error = p_error,
            completed_at = NOW()
        WHERE id = p_job_id;
    ELSE
        -- Reset to pending for retry
        UPDATE processing_queue
        SET
            status = 'pending',
            worker_id = NULL,
            started_at = NULL,
            error = p_error
        WHERE id = p_job_id;
    END IF;

    RETURN FOUND;
END;
$$;

-- ============================================
-- Trigger: Auto-create queue job on asset insert
-- ============================================
CREATE OR REPLACE FUNCTION create_processing_job_on_asset_insert()
RETURNS TRIGGER AS $$
BEGIN
    -- Only create job for image/video assets
    IF NEW.type IN ('image', 'photo', 'video') THEN
        INSERT INTO processing_queue (asset_id, user_id, operations, priority)
        VALUES (
            NEW.id,
            NEW.user_id,
            ARRAY['embedding', 'objects', 'faces'],
            5  -- Default priority
        )
        ON CONFLICT (asset_id) DO NOTHING;  -- Ignore if job already exists
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger on assets table
DROP TRIGGER IF EXISTS trigger_create_processing_job ON assets;
CREATE TRIGGER trigger_create_processing_job
AFTER INSERT ON assets
FOR EACH ROW
EXECUTE FUNCTION create_processing_job_on_asset_insert();

-- ============================================
-- Enable Realtime for processing_queue
-- Workers subscribe to INSERT events
-- ============================================
ALTER PUBLICATION supabase_realtime ADD TABLE processing_queue;

-- ============================================
-- Cleanup function: Reset stale processing jobs
-- Call periodically to handle crashed workers
-- ============================================
CREATE OR REPLACE FUNCTION reset_stale_processing_jobs(
    p_stale_threshold INTERVAL DEFAULT '10 minutes'
)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_count INTEGER;
BEGIN
    UPDATE processing_queue
    SET
        status = 'pending',
        worker_id = NULL,
        started_at = NULL
    WHERE status = 'processing'
      AND started_at < NOW() - p_stale_threshold;

    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END;
$$;
