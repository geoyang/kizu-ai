-- Kizu AI Database Schema
-- Run this migration in your Supabase SQL editor

-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================
-- Image Embeddings (CLIP)
-- Stores semantic embeddings for image search
-- ============================================
CREATE TABLE IF NOT EXISTS image_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    embedding vector(512),  -- CLIP ViT-B/32 dimension
    model_version TEXT NOT NULL DEFAULT 'openclip-vit-b-32',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(asset_id, model_version)
);

-- Index for fast similarity search
CREATE INDEX IF NOT EXISTS idx_image_embeddings_vector
ON image_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_image_embeddings_user
ON image_embeddings(user_id);

CREATE INDEX IF NOT EXISTS idx_image_embeddings_asset
ON image_embeddings(asset_id);

-- ============================================
-- Face Clusters
-- Groups similar faces for easier labeling
-- ============================================
CREATE TABLE IF NOT EXISTS face_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT,  -- User-assigned name (nullable until labeled)
    knox_contact_id UUID,  -- Link to Knox contact (if assigned)
    representative_face_id UUID,  -- Best face image for this cluster
    face_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_face_clusters_user
ON face_clusters(user_id);

CREATE INDEX IF NOT EXISTS idx_face_clusters_contact
ON face_clusters(knox_contact_id);

-- ============================================
-- Face Embeddings
-- Stores face detection and recognition data
-- ============================================
CREATE TABLE IF NOT EXISTS face_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    face_index INTEGER NOT NULL DEFAULT 0,  -- Which face in the image
    embedding vector(512),  -- InsightFace dimension
    bounding_box JSONB,  -- {x, y, width, height}
    landmarks JSONB,  -- Facial landmarks
    cluster_id UUID REFERENCES face_clusters(id) ON DELETE SET NULL,
    knox_contact_id UUID,  -- Direct assignment to contact
    confidence FLOAT,  -- Match confidence (1.0 for manual assignment)
    model_version TEXT NOT NULL DEFAULT 'insightface-buffalo-l',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_face_embeddings_vector
ON face_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_face_embeddings_user
ON face_embeddings(user_id);

CREATE INDEX IF NOT EXISTS idx_face_embeddings_asset
ON face_embeddings(asset_id);

CREATE INDEX IF NOT EXISTS idx_face_embeddings_cluster
ON face_embeddings(cluster_id);

CREATE INDEX IF NOT EXISTS idx_face_embeddings_contact
ON face_embeddings(knox_contact_id);

-- ============================================
-- Detected Objects
-- Stores YOLO object detection results
-- ============================================
CREATE TABLE IF NOT EXISTS detected_objects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    object_class TEXT NOT NULL,  -- 'person', 'dog', 'car', etc.
    confidence FLOAT NOT NULL,
    bounding_box JSONB,  -- {x, y, width, height}
    model_version TEXT NOT NULL DEFAULT 'yolov8-m',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_detected_objects_asset
ON detected_objects(asset_id);

CREATE INDEX IF NOT EXISTS idx_detected_objects_class
ON detected_objects(object_class);

CREATE INDEX IF NOT EXISTS idx_detected_objects_user
ON detected_objects(user_id);

-- ============================================
-- Image Descriptions
-- Stores AI-generated natural language descriptions
-- ============================================
CREATE TABLE IF NOT EXISTS image_descriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    description TEXT NOT NULL,
    detail_level TEXT DEFAULT 'medium',  -- 'brief', 'medium', 'detailed'
    model_version TEXT NOT NULL DEFAULT 'llava-1.5-7b',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(asset_id, model_version, detail_level)
);

CREATE INDEX IF NOT EXISTS idx_image_descriptions_asset
ON image_descriptions(asset_id);

-- ============================================
-- Image Text (OCR)
-- Stores extracted text from images
-- ============================================
CREATE TABLE IF NOT EXISTS image_text (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    extracted_text TEXT NOT NULL,
    text_regions JSONB,  -- Array of {text, bbox, confidence}
    languages JSONB,  -- Detected languages
    model_version TEXT NOT NULL DEFAULT 'easyocr',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(asset_id, model_version)
);

CREATE INDEX IF NOT EXISTS idx_image_text_asset
ON image_text(asset_id);

-- Full-text search on extracted text
CREATE INDEX IF NOT EXISTS idx_image_text_fts
ON image_text USING gin(to_tsvector('english', extracted_text));

-- ============================================
-- Processing Jobs
-- Tracks async processing jobs
-- ============================================
CREATE TABLE IF NOT EXISTS ai_processing_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    job_type TEXT NOT NULL,  -- 'process_image', 'batch_process', 'cluster_faces'
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, processing, completed, failed
    input_params JSONB,
    result JSONB,
    progress INTEGER DEFAULT 0,  -- 0-100
    processed INTEGER DEFAULT 0,
    total INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_ai_jobs_user
ON ai_processing_jobs(user_id);

CREATE INDEX IF NOT EXISTS idx_ai_jobs_status
ON ai_processing_jobs(status);

-- ============================================
-- Row Level Security (RLS) Policies
-- ============================================

-- Enable RLS on all tables
ALTER TABLE image_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE face_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE face_clusters ENABLE ROW LEVEL SECURITY;
ALTER TABLE detected_objects ENABLE ROW LEVEL SECURITY;
ALTER TABLE image_descriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE image_text ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_processing_jobs ENABLE ROW LEVEL SECURITY;

-- Policies: Users can only access their own data
CREATE POLICY "Users can access own image_embeddings"
ON image_embeddings FOR ALL
USING (auth.uid() = user_id);

CREATE POLICY "Users can access own face_embeddings"
ON face_embeddings FOR ALL
USING (auth.uid() = user_id);

CREATE POLICY "Users can access own face_clusters"
ON face_clusters FOR ALL
USING (auth.uid() = user_id);

CREATE POLICY "Users can access own detected_objects"
ON detected_objects FOR ALL
USING (auth.uid() = user_id);

CREATE POLICY "Users can access own image_descriptions"
ON image_descriptions FOR ALL
USING (auth.uid() = user_id);

CREATE POLICY "Users can access own image_text"
ON image_text FOR ALL
USING (auth.uid() = user_id);

CREATE POLICY "Users can access own ai_processing_jobs"
ON ai_processing_jobs FOR ALL
USING (auth.uid() = user_id);

-- ============================================
-- Vector Search Functions (RPC)
-- ============================================

-- Search image embeddings by similarity
CREATE OR REPLACE FUNCTION search_image_embeddings(
    query_embedding vector(512),
    match_count INT DEFAULT 10,
    match_threshold FLOAT DEFAULT 0.5,
    filter_user_id UUID DEFAULT NULL
)
RETURNS TABLE (
    asset_id UUID,
    similarity FLOAT,
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        ie.asset_id,
        1 - (ie.embedding <=> query_embedding) AS similarity,
        jsonb_build_object(
            'model_version', ie.model_version,
            'created_at', ie.created_at
        ) AS metadata
    FROM image_embeddings ie
    WHERE (filter_user_id IS NULL OR ie.user_id = filter_user_id)
      AND 1 - (ie.embedding <=> query_embedding) >= match_threshold
    ORDER BY ie.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Search face embeddings by similarity
CREATE OR REPLACE FUNCTION search_face_embeddings(
    query_embedding vector(512),
    match_count INT DEFAULT 10,
    match_threshold FLOAT DEFAULT 0.6,
    filter_user_id UUID DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    asset_id UUID,
    face_index INT,
    similarity FLOAT,
    cluster_id UUID,
    knox_contact_id UUID,
    bounding_box JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        fe.id,
        fe.asset_id,
        fe.face_index,
        1 - (fe.embedding <=> query_embedding) AS similarity,
        fe.cluster_id,
        fe.knox_contact_id,
        fe.bounding_box
    FROM face_embeddings fe
    WHERE (filter_user_id IS NULL OR fe.user_id = filter_user_id)
      AND 1 - (fe.embedding <=> query_embedding) >= match_threshold
    ORDER BY fe.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Update face cluster count trigger
CREATE OR REPLACE FUNCTION update_face_cluster_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN
        IF NEW.cluster_id IS NOT NULL THEN
            UPDATE face_clusters
            SET face_count = (
                SELECT COUNT(*) FROM face_embeddings WHERE cluster_id = NEW.cluster_id
            ),
            updated_at = NOW()
            WHERE id = NEW.cluster_id;
        END IF;
    END IF;

    IF TG_OP = 'DELETE' OR TG_OP = 'UPDATE' THEN
        IF OLD.cluster_id IS NOT NULL THEN
            UPDATE face_clusters
            SET face_count = (
                SELECT COUNT(*) FROM face_embeddings WHERE cluster_id = OLD.cluster_id
            ),
            updated_at = NOW()
            WHERE id = OLD.cluster_id;
        END IF;
    END IF;

    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_face_cluster_count
AFTER INSERT OR UPDATE OR DELETE ON face_embeddings
FOR EACH ROW
EXECUTE FUNCTION update_face_cluster_count();
