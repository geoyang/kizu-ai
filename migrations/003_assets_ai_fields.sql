-- Add AI processing tracking fields to assets table
-- Run this migration in your Supabase SQL editor after 002_processing_queue.sql

-- ============================================
-- AI Processing Fields on Assets
-- Tracks when/how each asset was AI processed
-- ============================================

-- Add AI processing tracking columns
ALTER TABLE assets
ADD COLUMN IF NOT EXISTS ai_processed_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS ai_processed_date DATE,
ADD COLUMN IF NOT EXISTS ai_processed_by TEXT,
ADD COLUMN IF NOT EXISTS ai_processed_operations TEXT[],
ADD COLUMN IF NOT EXISTS ai_model_versions JSONB;

-- Comments for documentation
COMMENT ON COLUMN assets.ai_processed_at IS 'Timestamp when AI processing completed';
COMMENT ON COLUMN assets.ai_processed_date IS 'Date when AI processing completed (for easier date-based queries)';
COMMENT ON COLUMN assets.ai_processed_by IS 'Worker ID that processed this asset';
COMMENT ON COLUMN assets.ai_processed_operations IS 'Array of operations performed: embedding, objects, faces, ocr, describe';
COMMENT ON COLUMN assets.ai_model_versions IS 'JSON object with model versions used: {"clip": "ViT-B-32", "yolo": "v8m", "insightface": "buffalo-l"}';

-- Index for finding unprocessed assets
CREATE INDEX IF NOT EXISTS idx_assets_ai_processed
ON assets(ai_processed_at) WHERE ai_processed_at IS NULL;

-- Index for date-based queries
CREATE INDEX IF NOT EXISTS idx_assets_ai_processed_date
ON assets(ai_processed_date);

-- Index for finding assets processed by specific worker
CREATE INDEX IF NOT EXISTS idx_assets_ai_processed_by
ON assets(ai_processed_by) WHERE ai_processed_by IS NOT NULL;
