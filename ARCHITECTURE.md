# Kizu-AI Architecture

A privacy-first, locally-hosted AI engine for image intelligence. All processing runs on-device with open source models only. No external API calls. No cloud AI providers.

## System Overview

The system uses a **pull-based architecture** where local AI workers pull jobs from Supabase. This allows the AI engine to run anywhere (home NAS, laptop, cloud) without requiring a public endpoint.

```
+-------------------+          +----------------------------------------+
|   Kizu Mobile /   |          |              SUPABASE (Cloud)          |
|   Web App         |          |                                        |
+--------+----------+          |  +------------+  +-------------------+ |
         |                     |  |  Storage   |  | ai_processing_    | |
         | Upload              |  |  (images)  |  | jobs              | |
         v                     |  +-----+------+  +--------+----------+ |
+-------------------+          |        |  trigger         |            |
|  Supabase Storage |--------->|        v                  | Realtime   |
+-------------------+          |  +------------+  +--------+----------+ |
                               |  | Insert job |  | image_processing_ | |
                               |  +------------+  | jobs              | |
                               |                   +--------+----------+ |
                               +----------------------------------------+
                                                           |
                             ============ INTERNET =========
                                                           |
                              +----------------------------v-----------+
                              |          LOCAL AI (Your NAS/PC)        |
                              |                                        |
                              |  +----------------------------------+  |
                              |  |       Unified AI Worker          |  |
                              |  |                                  |  |
                              |  |  Polls 2 tables, 3 job types:   |  |
                              |  |  1. Image AI (embed/face/obj)   |  |
                              |  |  2. Moments generation          |  |
                              |  |  3. Image optimization          |  |
                              |  +----------------------------------+  |
                              |                                        |
                              |  +----------------------------------+  |
                              |  |       Video Worker               |  |
                              |  |  FFmpeg transcoding              |  |
                              |  +----------------------------------+  |
                              |                                        |
                              |  Models: CLIP | YOLO | InsightFace |   |
                              |          Florence-2 | EasyOCR         |
                              +----------------------------------------+
```

## The Unified Worker

The core of Kizu-AI is the **Unified Worker** (`api/workers/local_worker.py`) — a single process that manages **3 distinct job types** from **2 Supabase tables**. This design keeps deployment simple (one process to run) while handling all AI and image processing needs.

### Job Types at a Glance

| # | Job Type | Table | Handler | Purpose |
|---|----------|-------|---------|---------|
| 1 | Image AI | `ai_processing_jobs` | `_process_image_ai_job()` | CLIP embeddings, face detection, object detection, OCR, descriptions |
| 2 | Moments Generation | `ai_processing_jobs` | `_process_moments_job()` | Auto-generate photo collections using 4 clustering methods |
| 3 | Image Optimization | `image_processing_jobs` | `_process_image_job()` | Generate thumbnails and web-optimized versions |

### How It Works

```
UnifiedWorker.start()
    │
    ├── Initialize Supabase client (service role)
    ├── Initialize ProcessService (AI models)
    ├── Subscribe to Realtime (instant notifications)
    └── Enter main loop
            │
            ├── _poll_ai_jobs()        ← ai_processing_jobs table
            │     │
            │     ├── Fetch first pending job
            │     ├── Claim atomically (eq status='pending')
            │     └── Route by job_type:
            │           ├── 'image_ai'        → _process_image_ai_job()
            │           └── 'generate_moments' → _process_moments_job()
            │
            ├── _poll_image_jobs()     ← image_processing_jobs table
            │     │
            │     ├── Fetch first pending job
            │     ├── Claim atomically
            │     └── _process_image_job()
            │
            └── sleep(poll_interval)   ← default 5 seconds
```

### Atomic Job Claiming

Multiple workers can run simultaneously without conflicts. Each worker claims jobs atomically:

```python
# Only succeeds if job is still 'pending' — prevents race conditions
update_result = client.table('ai_processing_jobs').update({
    'status': 'processing',
    'worker_id': self._worker_id,
    'picked_up_at': datetime.utcnow().isoformat(),
}).eq('id', job['id']).eq('status', 'pending').execute()

# If another worker claimed it first, update_result.data is empty
if not update_result.data:
    return  # Skip — already claimed
```

### Realtime + Polling Hybrid

The worker uses two mechanisms for responsiveness:

1. **Supabase Realtime**: Subscribes to INSERT events on both job tables for instant notifications
2. **Polling fallback**: Polls every N seconds (default: 5) in case Realtime misses events

---

## Job Type 1: Image AI Processing

**Table:** `ai_processing_jobs` with `job_type = 'image_ai'` (default)

Runs AI models on a single image and stores results for search.

### Operations

| Operation | Model | Output Table | Description |
|-----------|-------|-------------|-------------|
| `embedding` | OpenCLIP ViT-B/32 | `image_embeddings` | 512-dim semantic vector for search |
| `faces` | InsightFace buffalo_l | `face_embeddings` | Face detection + 512-dim recognition vectors |
| `objects` | YOLOv8-m | `detected_objects` | Object detection (80 COCO classes) |
| `ocr` | EasyOCR | `extracted_text` | Text extraction from images |
| `describe` | Florence-2 | `image_descriptions` | AI-generated image captions |
| `all` | All models | All tables | Run every operation |

### Processing Flow

1. Worker downloads image from Supabase Storage (or URL)
2. `ProcessService.process_image()` runs requested operations
3. Results stored to respective tables with `user_id` for isolation
4. Face embeddings matched against existing clusters for recognition
5. Job marked completed with result metadata

### Input Parameters

```json
{
  "asset_id": "uuid",
  "image_url": "https://...",
  "operations": ["embedding", "faces", "objects"]
}
```

---

## Job Type 2: Moments Generation

**Table:** `ai_processing_jobs` with `job_type = 'generate_moments'`

Automatically generates curated photo collections ("moments") for a user by analyzing their photo library. Uses 4 clustering methods with 2 fallbacks.

### Clustering Methods

#### 1. Location-Based (DBSCAN)
- Groups photos by GPS coordinates using DBSCAN clustering
- Parameters: ~5km radius (`eps_degrees = 0.045`), minimum 4 photos
- Produces moments like "Adventures in San Francisco"

#### 2. People-Based (Face Clusters)
- Groups photos by recognized people using face clusters
- Requires 4+ photos of the same person with a linked contact name
- Produces moments like "Moments with John"

#### 3. On This Day (Historical)
- Finds photos from the same week in previous years
- Uses a +/- 3 day window around today
- Requires 3+ photos from the same year
- Produces moments like "This week, 2 years ago"

#### 4. Event-Based (Temporal)
- Groups photos taken within 4-hour gaps into events
- Maximum event duration: 3 days
- Only considers the last 90 days, limits to 5 events
- Produces moments like "Saturday afternoon at Central Park"

#### Fallback 1: Most Common Location
- If no moments from the 4 primary methods, finds the location with most photos

#### Fallback 2: Recent Photos
- Last resort — takes the 20 most recent photos

### Cover Photo Selection

Uses furthest-first traversal on CLIP embeddings to select 6 visually diverse cover photos.

### User Settings

Stored in `profiles.notification_settings`:

```json
{
  "moments": true,
  "moments_time": "09:00",
  "moments_auto_delete_days": 7,
  "moments_location": true,
  "moments_people": true,
  "moments_events": true,
  "moments_on_this_day": true
}
```

### Output

Moments are stored in two tables:
- `moments` — the moment metadata (title, subtitle, cover photos, expiry)
- `moment_assets` — join table linking moments to photos (with display order)

Unsaved moments auto-expire after the configured number of days (default: 7).

---

## Job Type 3: Image Optimization

**Table:** `image_processing_jobs`

Generates optimized image versions for fast display in the mobile and web apps.

### Processing Steps

1. Download source image from URL
2. Apply EXIF orientation correction (supports all 8 orientations)
3. Generate **thumbnail**: 300x300px, center-cropped, JPEG quality 80
4. Generate **web version**: max 4096px longest side, JPEG quality 92
5. Upload both to Supabase Storage: `{user_id}/processed/{asset_id}/`
6. Update the `assets` table with `thumbnail` and `web_uri` URLs

### HEIC Support

Uses `pillow-heif` to handle Apple HEIC/HEIF images when available.

---

## Video Worker

A separate worker (`api/workers/video_worker.py`) handles video transcoding using FFmpeg. It polls the same `image_processing_jobs` table for video jobs.

**Entry point:** `python run_video_worker.py`

---

## Core Components

| Component | File | Purpose |
|-----------|------|---------|
| Unified Worker | `api/workers/local_worker.py` | Polls and processes all 3 job types |
| Video Worker | `api/workers/video_worker.py` | FFmpeg video transcoding |
| Process Service | `api/services/process_service.py` | Orchestrates AI model operations |
| Moments Service | `api/services/moments_service.py` | Photo moments clustering and generation |
| Search Service | `api/services/search_service.py` | Semantic search with NLP date parsing |
| Clustering Service | `api/services/clustering_service.py` | Face clustering (HDBSCAN) |
| Face Service | `api/services/face_service.py` | Face operations |
| Model Registry | `api/core/registry.py` | Lazy-loaded model management (singleton) |
| Vector Store | `api/stores/supabase_store.py` | pgvector operations for embeddings |

## AI Models

All models are fully open source with no external API dependencies.

| Model | Implementation | Dimension | Use Case |
|-------|---------------|-----------|----------|
| **OpenCLIP ViT-B/32** | `CLIPEmbedder` | 512 | Semantic image search, text-to-image matching |
| **YOLOv8-m** | `YOLODetector` | N/A | Object detection (80 COCO classes) |
| **InsightFace buffalo_l** | `InsightFaceModel` | 512 | Face detection, recognition, clustering |
| **EasyOCR** | `EasyOCRModel` | N/A | Text extraction (15+ languages) |
| **Florence-2** | `Florence2Model` | N/A | Image descriptions & captions |

### Model Registry Pattern

Models are registered with lazy-loading factories and loaded on first use:

```python
from api.core import ModelRegistry
from api.core.registry import ModelType

# Get default model (lazy loaded on first call)
embedder = ModelRegistry.get(ModelType.EMBEDDER)

# Get specific variant
detector = ModelRegistry.get(ModelType.DETECTOR, "yolo")

# Unload to free memory
ModelRegistry.unload(ModelType.EMBEDDER, "clip")
ModelRegistry.unload_all()
```

## Database Schema

### Job Tables

```
ai_processing_jobs
├── id              UUID PRIMARY KEY
├── user_id         UUID
├── asset_id        UUID
├── job_type        TEXT ('image_api' | 'generate_moments')
├── status          TEXT ('pending' | 'processing' | 'completed' | 'failed')
├── input_params    JSONB {asset_id, image_url, operations[]}
├── worker_id       TEXT (which worker claimed the job)
├── progress        INT (0-100)
├── result          JSONB
├── error_message   TEXT
├── created_at      TIMESTAMPTZ
├── picked_up_at    TIMESTAMPTZ
└── completed_at    TIMESTAMPTZ

image_processing_jobs
├── id              UUID PRIMARY KEY
├── asset_id        UUID
├── user_id         UUID
├── status          TEXT
├── input_url       TEXT (source image URL)
├── input_orientation INT (EXIF 1-8)
├── needs_web_version BOOLEAN
├── metadata        JSONB
├── progress        INT (0-100)
├── current_step    TEXT
├── thumbnail_url   TEXT (output)
├── web_url         TEXT (output)
├── error_message   TEXT
├── created_at      TIMESTAMPTZ
├── started_at      TIMESTAMPTZ
└── completed_at    TIMESTAMPTZ
```

### AI Result Tables

```
image_embeddings      ← CLIP 512-dim vectors (pgvector)
face_embeddings       ← InsightFace 512-dim vectors + bounding boxes
face_clusters         ← Grouped face identities
detected_objects      ← YOLO detection results
extracted_text        ← OCR text content
image_descriptions    ← Florence-2 captions
```

### Moments Tables

```
moments
├── id                UUID PRIMARY KEY
├── user_id           UUID
├── grouping_type     TEXT ('location' | 'people' | 'on_this_day' | 'event' | 'recent')
├── grouping_criteria JSONB (clustering parameters)
├── title             TEXT
├── subtitle          TEXT
├── cover_asset_ids   UUID[] (6 diverse cover photos)
├── date_range_start  TIMESTAMPTZ
├── date_range_end    TIMESTAMPTZ
├── is_saved          BOOLEAN (user manually saved)
├── expires_at        TIMESTAMPTZ (auto-delete)
└── created_at        TIMESTAMPTZ

moment_assets
├── id              UUID PRIMARY KEY
├── moment_id       UUID (references moments)
├── asset_id        UUID (references assets)
└── display_order   INT
```

## Processing Flows

### Image Upload → AI Results

1. User uploads image to Kizu Mobile/Web
2. Image stored in Supabase Storage
3. Edge function inserts job into `ai_processing_jobs` with `job_type='image_ai'`
4. Worker receives Realtime notification (or polls)
5. Worker claims job atomically, downloads image
6. AI processes image (embedding, faces, objects, OCR, description)
7. Results stored to pgvector tables
8. Job marked completed — image is now searchable

### Moments Generation

1. Admin or scheduled trigger inserts job into `ai_processing_jobs` with `job_type='generate_moments'`
2. Worker picks up job, loads user's moment settings
3. `MomentsService.generate_moments_for_user()` runs 4 clustering methods
4. Candidates scored and deduplicated
5. Cover photos selected using CLIP embedding diversity
6. Moments saved to `moments` + `moment_assets` tables with auto-expiry
7. User sees moments in their feed

### Image Optimization

1. Asset created in Supabase, job inserted into `image_processing_jobs`
2. Worker downloads source image
3. EXIF orientation applied, thumbnail and web version generated
4. Optimized images uploaded to Supabase Storage
5. Asset record updated with `thumbnail` and `web_uri` URLs

### Search Query

1. User enters natural language query (e.g., "sunset photos from Hawaii")
2. NLP parser extracts dates, locations, semantic terms
3. CLIP embeds the semantic query into 512-dim vector
4. pgvector cosine similarity search on `image_embeddings`
5. Post-filter by date/location/object/face constraints
6. Return ranked results with thumbnails

## Deployment

### Docker Compose Services

```yaml
services:
  api:           # FastAPI server (port 8000)
  worker:        # Unified AI worker (3 job types)
  video-worker:  # FFmpeg video transcoding
  worker-dev:    # Pre-configured for dev Supabase
  video-worker-dev:  # Pre-configured for dev Supabase
```

### Running Locally

```bash
# All services
docker-compose up -d

# Or individually
uvicorn api.main:app --reload          # API server
python run_worker.py --debug           # Unified worker
python run_video_worker.py --debug     # Video worker
```

### Multiple Workers

Scale horizontally by running multiple worker instances. Atomic job claiming prevents duplicate processing:

```bash
python run_worker.py --worker-id nas-1 &
python run_worker.py --worker-id nas-2 &
```

## Configuration

Environment variables (`.env`):

```bash
# Required
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# Models
MODEL_CACHE_DIR=./model_cache
CLIP_MODEL=ViT-B-32
CLIP_PRETRAINED=openai
YOLO_SIZE=m                    # n, s, m, l, x
FACE_MODEL_NAME=buffalo_l
FACE_THRESHOLD=0.6
DEFAULT_VLM=florence2
FLORENCE_MODEL=microsoft/Florence-2-base
VLM_QUANTIZATION=int8

# Processing
MAX_IMAGE_SIZE=2048
BATCH_SIZE=8

# API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Privacy (always enforced)
ALLOW_EXTERNAL_APIS=false
```

## Why Pull-Based?

| Push (Webhook) | Pull (Realtime + Polling) |
|----------------|--------------------------|
| Requires public endpoint | Works behind NAT/firewall |
| Single point of failure | Multiple workers anywhere |
| Complex networking | Simple: just outbound HTTPS |
| Hard to scale | Add more workers instantly |

## Privacy Philosophy

1. **All AI runs locally** — No data leaves the device
2. **No cloud AI providers** — No OpenAI, Google, AWS AI services
3. **Open source models only** — Fully auditable, no proprietary weights
4. **No external API calls** — `ALLOW_EXTERNAL_APIS=false` enforced
5. **User data isolation** — All queries filtered by `user_id`

## Adding New Models

See the [Model Abstraction](#model-registry-pattern) section. The registry pattern enables swapping models without changing worker or service code:

1. Create a class extending the appropriate base (`BaseEmbedder`, `BaseDetector`, etc.)
2. Register it in the model registry
3. Set it as default or load it on demand via the API
