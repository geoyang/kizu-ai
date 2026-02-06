# Kizu AI Processing Documentation

This document describes the AI processing capabilities available in Kizu for image analysis, face recognition, semantic search, photo moments, and image optimization.

---

## Table of Contents

1. [Overview](#overview)
2. [Processing Operations](#processing-operations)
3. [Single Image Processing](#single-image-processing)
4. [Batch Processing](#batch-processing)
5. [Photo Moments](#photo-moments)
6. [Image Optimization](#image-optimization)
7. [Face Clustering](#face-clustering)
8. [Tag Sync Preview](#tag-sync-preview)
9. [Search Capabilities](#search-capabilities)
10. [Reindexing](#reindexing)
11. [Database Schema](#database-schema)
12. [Web Admin UI](#web-admin-ui)

---

## Overview

Kizu AI provides automated image analysis using:
- **CLIP** — Semantic image embeddings for natural language search
- **InsightFace** — Face detection and 512-dimensional face embeddings
- **YOLO** — Object detection (80+ object classes)
- **OCR** — Text extraction from images
- **Florence-2** — Vision-language model for image descriptions

All processing is user-scoped — each user's data is isolated and only accessible to them.

### Worker Architecture

A single **Unified Worker** process handles 3 job types from 2 Supabase tables:

| Job Type | Table | What It Does |
|----------|-------|-------------|
| Image AI (`image_ai`) | `ai_processing_jobs` | Run AI models (embedding, faces, objects, OCR, describe) |
| Moments (`generate_moments`) | `ai_processing_jobs` | Generate photo collections via clustering |
| Image Optimization | `image_processing_jobs` | Generate thumbnails and web versions |

The worker uses Supabase Realtime for instant notifications with polling as a fallback. Multiple workers can run simultaneously — atomic job claiming prevents duplicate processing.

See [ARCHITECTURE.md](ARCHITECTURE.md) for full worker architecture details.

---

## Processing Operations

| Operation | Model | Description | Output Table |
|-----------|-------|-------------|--------------|
| `embedding` | CLIP ViT-B/32 | Semantic vectors (512-dim) | `image_embeddings` |
| `faces` | InsightFace | Face detection + embeddings | `face_embeddings` |
| `objects` | YOLOv8-m | Object detection | `detected_objects` |
| `ocr` | EasyOCR | Text extraction | `extracted_text` |
| `describe` | Florence-2 | AI-generated captions | `image_descriptions` |
| `all` | All models | Run all operations | All tables |

---

## Single Image Processing

**Endpoint:** `POST /api/v1/process/image`

Processes a single image synchronously and returns results immediately.

**Parameters:**
- `image_url` or `image_base64` — The image to process (required)
- `asset_id` — Identifier for the asset (default: "test")
- `operations` — List of operations to run (default: ["all"])
- `store_results` — Save results to database (default: false)
- `force_reprocess` — Clear existing AI data before processing (default: false)

**Use Cases:**
- Testing AI models on specific images
- Debugging processing issues
- Quick analysis without database storage

---

## Batch Processing

**Endpoint:** `POST /api/v1/process/batch`

Queues multiple images for asynchronous background processing.

**Parameters:**
- `asset_ids` — List of asset IDs to process (optional, processes unprocessed if omitted)
- `operations` — List of operations to run
- `limit` — Maximum assets to process (1-1000)
- `skip_processed` — Skip assets that already have AI data
- `force_reprocess` — Clear existing AI data before processing

**Returns:** Job ID for polling progress via `GET /api/v1/jobs/{job_id}`

---

## Photo Moments

Photo moments are auto-generated collections that surface meaningful groups of photos from a user's library.

### How Moments Are Generated

When a `generate_moments` job is created in `ai_processing_jobs`, the worker runs `MomentsService.generate_moments_for_user()` which applies 4 clustering methods (+ 2 fallbacks):

#### 1. Location-Based Clustering
- **Algorithm:** DBSCAN on GPS coordinates
- **Parameters:** ~5km radius, minimum 4 photos per cluster
- **Example output:** "Adventures in San Francisco" — 24 photos

#### 2. People-Based Clustering
- **Source:** `face_clusters` table (from face recognition)
- **Requirements:** 4+ photos of a person with a linked contact name
- **Example output:** "Moments with John" — 15 photos

#### 3. On This Day
- **Algorithm:** Find photos from same week (+/- 3 days) in previous years
- **Requirements:** 3+ photos from the same year
- **Example output:** "This week, 2 years ago" — 8 photos from Portland

#### 4. Event-Based Clustering
- **Algorithm:** Temporal clustering with 4-hour gap threshold
- **Constraints:** Max 3-day event duration, last 90 days only, max 5 events
- **Example output:** "Saturday afternoon at Central Park" — 12 photos

#### Fallback 1: Most Common Location
Triggers when the 4 primary methods produce no results. Creates a moment from the user's most photographed location.

#### Fallback 2: Recent Photos
Last resort — collects the 20 most recent photos.

### Cover Photo Selection

Each moment gets 6 cover photos selected for visual diversity using furthest-first traversal on CLIP embeddings. This ensures the covers represent a variety of scenes rather than similar shots.

### User Settings

Users can configure moment preferences in their profile settings (`notification_settings` JSONB column):

| Setting | Default | Description |
|---------|---------|-------------|
| `moments` | `true` | Enable/disable moments |
| `moments_time` | `"09:00"` | Preferred generation time |
| `moments_auto_delete_days` | `7` | Days before unsaved moments expire |
| `moments_location` | `true` | Enable location clustering |
| `moments_people` | `true` | Enable people clustering |
| `moments_events` | `true` | Enable event clustering |
| `moments_on_this_day` | `true` | Enable historical clustering |

### Moments API

The `moments-api` edge function provides the user-facing API:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/feed` | List moments (saved + not expired) |
| GET | `/:id` | Get single moment with all assets |
| POST | `/:id/save` | Save moment (prevents expiration) |
| POST | `/:id/dismiss` | Hide moment from feed |
| POST | `/:id/view` | Mark moment as viewed |
| GET | `/settings` | Get user moment preferences |
| POST | `/settings` | Update moment preferences |

### Moment Lifecycle

1. **Generated** — Worker creates moment with `expires_at` set (default: 7 days)
2. **Viewed** — User opens moment, `viewed_at` is set
3. **Saved** — User saves moment, `expires_at` cleared (permanent)
4. **Dismissed** — User hides moment, `dismissed_at` set (filtered from feed)
5. **Expired** — Unsaved moments past `expires_at` are filtered from feed

---

## Image Optimization

The worker generates optimized image versions for fast display.

### What It Does

| Output | Size | Quality | Purpose |
|--------|------|---------|---------|
| Thumbnail | 300x300px, center-cropped | JPEG 80 | Grid views, lists |
| Web version | Max 4096px longest side | JPEG 92 | Full-screen display |

### Processing Steps

1. Download source image from Supabase Storage
2. Apply EXIF orientation correction (all 8 orientations supported)
3. Convert HEIC/HEIF to JPEG (via `pillow-heif`)
4. Generate thumbnail (center-crop to square)
5. Generate web version (proportional resize)
6. Upload to Supabase Storage: `{user_id}/processed/{asset_id}/thumbnail.jpg` and `web.jpg`
7. Update `assets` table with `thumbnail` and `web_uri` URLs

---

## Face Clustering

Groups similar faces together using HDBSCAN clustering algorithm on face embeddings.

### Run Clustering

**Endpoint:** `POST /api/v1/faces/cluster`

**Parameters:**
- `threshold` — Distance threshold for grouping (0.3-0.9, default: 0.6)
- `min_cluster_size` — Minimum faces per cluster (default: 2)

**How it works:**
1. Fetches all face embeddings for the user
2. Runs HDBSCAN clustering on 512-dim embeddings
3. Creates `face_clusters` records for each group
4. Updates `face_embeddings.cluster_id` for each face

### List Clusters

**Endpoint:** `GET /api/v1/faces/clusters`

Returns all face clusters with sample thumbnails.

**Query Parameters:**
- `include_labeled` — Include clusters assigned to contacts (default: true)
- `include_unlabeled` — Include unassigned clusters (default: true)

### Assign Cluster to Contact

**Endpoint:** `POST /api/v1/faces/clusters/{cluster_id}/assign`

Links a face cluster to a contact for future recognition.

**Body:**
- `knox_contact_id` — The contact ID to assign
- `name` — Optional display name
- `exclude_face_ids` — Face IDs to exclude from assignment

### Merge Clusters

**Endpoint:** `POST /api/v1/faces/clusters/merge`

Combines multiple clusters into one (for when the same person was split into multiple clusters).

**Body:**
- `cluster_ids` — Array of cluster IDs to merge (minimum 2)

### Clear Clustering

**Endpoint:** `POST /api/v1/faces/clear-clustering`

Resets all clustering data to start fresh.

**Query Parameters:**
- `clear_thumbnails` — Also delete face thumbnail images (default: false)

---

## Tag Sync Preview

Compares manual face tags from the mobile app with AI-detected faces to train and improve clustering.

### Preview Matches

**Endpoint:** `POST /api/v1/faces/tag-sync/preview`

**Parameters:**
- `asset_ids` — Specific assets to check (optional, checks all if omitted)
- `iou_threshold` — Minimum bounding box overlap for matching (0.1-0.9, default: 0.3)
- `limit` — Maximum assets to return (1-200, default: 50)

**Matching Algorithm:**
1. For each manual tag, finds the AI detection with highest IoU overlap
2. IoU >= threshold: marked as "matched"
3. IoU between 0.1 and threshold: marked as "low_confidence"
4. IoU < 0.1: no match recorded
5. Each AI detection can only match one manual tag (greedy matching)

### Apply Matches

**Endpoint:** `POST /api/v1/faces/tag-sync/apply`

Applies confirmed matches to link AI faces with contacts.

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
3. This links the AI-detected face to a known contact, improving future clustering and search

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
- `query` — Natural language search query
- `limit` — Max results (1-200, default: 50)
- `threshold` — Minimum similarity (0.0-1.0, default: 0.2)

### Search by Face

**Endpoint:** `POST /api/v1/search/by-face`

Find all images containing a specific person.

**Parameters (one required):**
- `contact_id` — Contact ID
- `cluster_id` — Face cluster ID
- `face_embedding` — Direct 512-dim embedding

### Search by Object

**Endpoint:** `POST /api/v1/search/by-object`

Find images containing specific detected objects.

**Parameters:**
- `object_class` — Object type (e.g., "dog", "car", "wine glass")
- `min_confidence` — Minimum detection confidence (default: 0.5)

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
- `date_start`, `date_end` — Date range
- `people` — Contact or cluster IDs
- `object_classes` — Detected objects
- `description_query` — Description text
- `text_query` — OCR text
- `has_faces` — Images with detected faces
- `has_text` — Images with extracted text

All filters use AND logic.

---

## Reindexing

**Endpoint:** `POST /api/v1/process/reindex`

Process assets that don't have embeddings yet.

**Parameters:**
- `limit` — Batch size (default: 100)
- `offset` — Starting offset (default: 0)

**Usage:** Call repeatedly with increasing offset to process all assets:
```
POST /api/v1/process/reindex?limit=100&offset=0
POST /api/v1/process/reindex?limit=100&offset=100
POST /api/v1/process/reindex?limit=100&offset=200
```

---

## Database Schema

### Job Tables

| Table | Purpose |
|-------|---------|
| `ai_processing_jobs` | Image AI + moments generation jobs |
| `image_processing_jobs` | Thumbnail and web version generation |

### AI Result Tables

| Table | Description |
|-------|-------------|
| `image_embeddings` | CLIP vectors for semantic search (pgvector) |
| `face_embeddings` | Face detection + embeddings + bounding boxes |
| `face_clusters` | Grouped face identities |
| `detected_objects` | YOLO detections |
| `extracted_text` | OCR results |
| `image_descriptions` | Florence-2 captions |

### Moments Tables

| Table | Description |
|-------|-------------|
| `moments` | Moment metadata (title, covers, expiry, type) |
| `moment_assets` | Join table linking moments to photos |

### Key Fields

**face_embeddings:**
- `id` — Primary key
- `asset_id` — Source image
- `user_id` — Owner
- `embedding` — 512-dim vector
- `bounding_box` — Face location {x, y, width, height}
- `cluster_id` — Assigned cluster (nullable)
- `knox_contact_id` — Linked contact (nullable)
- `face_thumbnail_url` — Cropped face image

**moments:**
- `grouping_type` — 'location', 'people', 'on_this_day', 'event', or 'recent'
- `grouping_criteria` — JSONB with clustering parameters
- `cover_asset_ids` — Array of 6 diverse cover photo UUIDs
- `is_saved` — Whether user manually saved the moment
- `expires_at` — Auto-delete timestamp (null if saved)

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

Moments are managed via a separate **Moments** tab in the admin dashboard, which allows triggering moment generation for specific users and viewing generated moments.
