# Kizu AI Processing Documentation

This document describes the AI processing capabilities available in Kizu for image analysis, face recognition, and semantic search.

---

## Table of Contents

1. [Overview](#overview)
2. [Processing Operations](#processing-operations)
3. [Single Image Processing](#single-image-processing)
4. [Batch Processing](#batch-processing)
5. [Face Clustering](#face-clustering)
6. [Tag Sync Preview](#tag-sync-preview)
7. [Search Capabilities](#search-capabilities)
8. [Reindexing](#reindexing)
9. [Database Schema](#database-schema)

---

## Overview

Kizu AI provides automated image analysis using:
- **CLIP** - Semantic image embeddings for natural language search
- **InsightFace** - Face detection and 512-dimensional face embeddings
- **YOLO** - Object detection (80+ object classes)
- **OCR** - Text extraction from images
- **VLM** - Vision-language model for image descriptions

All processing is user-scoped - each user's data is isolated and only accessible to them.

---

## Processing Operations

| Operation | Description | Output Table |
|-----------|-------------|--------------|
| `embedding` | CLIP semantic vectors (512-dim) | `image_embeddings` |
| `faces` | Face detection + embeddings | `face_embeddings` |
| `objects` | YOLO object detection | `detected_objects` |
| `ocr` | Text extraction | `extracted_text` |
| `describe` | AI-generated captions | `image_descriptions` |
| `all` | Run all operations | All tables |

---

## Single Image Processing

**Endpoint:** `POST /api/v1/process/image`

Processes a single image synchronously and returns results immediately.

**Parameters:**
- `image_url` or `image_base64` - The image to process (required)
- `asset_id` - Identifier for the asset (default: "test")
- `operations` - List of operations to run (default: ["all"])
- `store_results` - Save results to database (default: false)
- `force_reprocess` - Clear existing AI data before processing (default: false)

**Use Cases:**
- Testing AI models on specific images
- Debugging processing issues
- Quick analysis without database storage

---

## Batch Processing

**Endpoint:** `POST /api/v1/process/batch`

Queues multiple images for asynchronous background processing.

**Parameters:**
- `asset_ids` - List of asset IDs to process (optional, processes unprocessed if omitted)
- `operations` - List of operations to run
- `limit` - Maximum assets to process (1-1000)
- `skip_processed` - Skip assets that already have AI data
- `force_reprocess` - Clear existing AI data before processing

**Returns:** Job ID for polling progress via `GET /api/v1/jobs/{job_id}`

---

## Face Clustering

Groups similar faces together using HDBSCAN clustering algorithm on face embeddings.

### Run Clustering

**Endpoint:** `POST /api/v1/faces/cluster`

**Parameters:**
- `threshold` - Distance threshold for grouping (0.3-0.9, default: 0.6)
- `min_cluster_size` - Minimum faces per cluster (default: 2)

**How it works:**
1. Fetches all face embeddings for the user
2. Runs HDBSCAN clustering on 512-dim embeddings
3. Creates `face_clusters` records for each group
4. Updates `face_embeddings.cluster_id` for each face

### List Clusters

**Endpoint:** `GET /api/v1/faces/clusters`

Returns all face clusters with sample thumbnails.

**Query Parameters:**
- `include_labeled` - Include clusters assigned to contacts (default: true)
- `include_unlabeled` - Include unassigned clusters (default: true)

### Assign Cluster to Contact

**Endpoint:** `POST /api/v1/faces/clusters/{cluster_id}/assign`

Links a face cluster to a Knox contact for future recognition.

**Body:**
- `knox_contact_id` - The contact ID to assign
- `name` - Optional display name
- `exclude_face_ids` - Face IDs to exclude from assignment

### Merge Clusters

**Endpoint:** `POST /api/v1/faces/clusters/merge`

Combines multiple clusters into one (for when the same person was split into multiple clusters).

**Body:**
- `cluster_ids` - Array of cluster IDs to merge (minimum 2)

### Clear Clustering

**Endpoint:** `POST /api/v1/faces/clear-clustering`

Resets all clustering data to start fresh.

**Query Parameters:**
- `clear_thumbnails` - Also delete face thumbnail images (default: false)

---

## Tag Sync Preview

Compares manual face tags from the mobile app with AI-detected faces to train and improve clustering.

**Architecture:** The AI service calls the `tag-sync-api` edge function, which handles database access. This follows the principle that all database access goes through edge functions.

### Preview Matches

**Endpoint:** `POST /api/v1/faces/tag-sync/preview`
**Edge Function:** `tag-sync-api/preview`

**Parameters:**
- `asset_ids` - Specific assets to check (optional, checks all if omitted)
- `iou_threshold` - Minimum bounding box overlap for matching (0.1-0.9, default: 0.3)
- `limit` - Maximum assets to return (1-200, default: 50)

**Data Sources:**
- Manual tags: `asset_tags` table where `bounding_box` is not null
- AI detections: `face_embeddings` table where `bounding_box` is not null

**Matching Algorithm:**
1. For each manual tag, finds the AI detection with highest IoU (Intersection over Union) overlap
2. IoU >= threshold: marked as "matched"
3. IoU between 0.1 and threshold: marked as "low_confidence"
4. IoU < 0.1: no match recorded
5. Each AI detection can only match one manual tag (greedy matching)

**Response:**
```json
{
  "assets": [
    {
      "asset_id": "abc123",
      "thumbnail_url": "https://...",
      "manual_tags": [
        {
          "tag_id": "tag1",
          "contact_id": "contact1",
          "contact_name": "John Doe",
          "bounding_box": {"x": 100, "y": 50, "width": 200, "height": 250}
        }
      ],
      "ai_detections": [
        {
          "face_id": "face1",
          "bounding_box": {"x": 105, "y": 55, "width": 195, "height": 245},
          "cluster_id": "cluster1",
          "thumbnail_url": "https://..."
        }
      ],
      "matches": [
        {
          "manual_tag_id": "tag1",
          "ai_face_id": "face1",
          "iou_score": 0.87,
          "status": "matched"
        }
      ]
    }
  ],
  "summary": {
    "total_assets": 1,
    "total_manual_tags": 1,
    "total_ai_faces": 1,
    "matched": 1,
    "unmatched_manual": 0,
    "unmatched_ai": 0
  }
}
```

### Apply Matches

**Endpoint:** `POST /api/v1/faces/tag-sync/apply`
**Edge Function:** `tag-sync-api/apply`

Applies confirmed matches to link AI faces with Knox contacts.

**Body:**
```json
{
  "matches": [
    {"manual_tag_id": "tag1", "ai_face_id": "face1"},
    {"manual_tag_id": "tag2", "ai_face_id": "face2"}
  ]
}
```

**What this does:**
1. Looks up the `contact_id` from each manual tag
2. Updates the corresponding `face_embeddings.knox_contact_id`
3. This links the AI-detected face to a known contact

**How it helps clustering:**
- Face embeddings with the same `knox_contact_id` are known to be the same person
- Future clustering can use this as ground truth
- Search by face becomes more accurate

---

## Search Capabilities

### Semantic Search

**Endpoint:** `POST /api/v1/search`

Natural language search using CLIP embeddings.

**Examples:**
- "photos at the beach"
- "sunset photos from 2023"
- "images with dogs"

**Parameters:**
- `query` - Natural language search query
- `limit` - Max results (1-200, default: 50)
- `threshold` - Minimum similarity (0.0-1.0, default: 0.2)

### Search by Face

**Endpoint:** `POST /api/v1/search/by-face`

Find all images containing a specific person.

**Parameters (one required):**
- `contact_id` - Knox contact ID
- `cluster_id` - Face cluster ID
- `face_embedding` - Direct 512-dim embedding

### Search by Object

**Endpoint:** `POST /api/v1/search/by-object`

Find images containing specific detected objects.

**Parameters:**
- `object_class` - Object type (e.g., "dog", "car", "wine glass")
- `min_confidence` - Minimum detection confidence (default: 0.5)

**Get Available Objects:** `GET /api/v1/search/objects`

### Search by Description

**Endpoint:** `POST /api/v1/search/by-description`

Search AI-generated image descriptions using fuzzy text matching.

### Search by OCR Text

**Endpoint:** `POST /api/v1/search/by-text`

Find images containing specific text (signs, documents, etc.).

### Combined Search

**Endpoint:** `POST /api/v1/search/combined`

Multi-filter search combining semantic query with filters.

**Filters:**
- `date_start`, `date_end` - Date range
- `people` - Contact or cluster IDs
- `object_classes` - Detected objects
- `description_query` - Description text
- `text_query` - OCR text
- `has_faces` - Images with detected faces
- `has_text` - Images with extracted text

All filters use AND logic.

---

## Reindexing

**Endpoint:** `POST /api/v1/process/reindex`

Process assets that don't have embeddings yet.

**Parameters:**
- `limit` - Batch size (default: 100)
- `offset` - Starting offset (default: 0)

**Usage:** Call repeatedly with increasing offset to process all assets:
```
POST /api/v1/process/reindex?limit=100&offset=0
POST /api/v1/process/reindex?limit=100&offset=100
POST /api/v1/process/reindex?limit=100&offset=200
...
```

---

## Database Schema

### Core Tables

| Table | Description |
|-------|-------------|
| `image_embeddings` | CLIP vectors for semantic search |
| `face_embeddings` | Face detection + embeddings |
| `face_clusters` | Grouped faces |
| `detected_objects` | YOLO detections |
| `extracted_text` | OCR results |
| `image_descriptions` | AI captions |

### Key Fields

**face_embeddings:**
- `id` - Primary key
- `asset_id` - Source image
- `user_id` - Owner
- `embedding` - 512-dim vector
- `bounding_box` - Face location {x, y, width, height}
- `cluster_id` - Assigned cluster (nullable)
- `knox_contact_id` - Linked contact (nullable)
- `face_thumbnail_url` - Cropped face image

**asset_tags (mobile app):**
- `id` - Primary key
- `asset_id` - Tagged image
- `tag_type` - 'person' or 'object'
- `tag_value` - Display name/label
- `knox_contact_id` - Knox contact UUID (nullable)
- `bounding_box` - Tag location {x, y, width, height}
- `created_by` - User who created the tag

---

## Web Admin UI

Access AI processing features at `/admin/ai-processing`:

| Tab | Description |
|-----|-------------|
| Single Test | Process individual images for testing |
| Album Processing | Batch process albums |
| Face Clusters | View and manage face groups |
| Tag Sync | Compare manual tags with AI detections |
| Search Test | Test search capabilities |
| Activity | Monitor processing activity |
| Upload Queue | View queued processing jobs |
| Reindex | Process unindexed assets |
