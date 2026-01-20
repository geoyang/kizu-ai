# Kizu-AI Architecture

A privacy-first, locally-hosted AI engine for image intelligence. All processing runs on-device with open source models only. No external API calls. No cloud AI providers.

## System Overview

The system uses a **pull-based architecture** where local AI workers pull jobs from Supabase. This allows the AI engine to run anywhere (home NAS, laptop, cloud) without requiring a public endpoint.

```
+-------------------+          +----------------------------------------+
|   Kizu Mobile /   |          |              SUPABASE (Cloud)          |
|   Web App         |          |                                        |
+--------+----------+          |  +------------+    +-----------------+ |
         |                     |  |  Storage   |    | processing_queue| |
         | Upload              |  |  (images)  |    |   (job queue)   | |
         v                     |  +-----+------+    +--------+--------+ |
+-------------------+          |        |  trigger          |           |
|  Supabase Storage |--------->|        v                   | Realtime  |
+-------------------+          |  +------------+            | WebSocket |
                               |  | Insert job |            |           |
                               |  +------------+            |           |
                               +----------------------------------------+
                                                            |
                              ============ INTERNET =========
                                                            |
                               +----------------------------v-----------+
                               |          LOCAL AI (Your NAS/PC)        |
                               |                                        |
                               |  +----------------------------------+  |
                               |  |         Local Worker             |  |
                               |  |                                  |  |
                               |  |  1. Subscribe to Realtime        |  |
                               |  |  2. Claim job atomically         |  |
                               |  |  3. Download & process image     |  |
                               |  |  4. Store results to Supabase    |  |
                               |  +----------------------------------+  |
                               |                                        |
                               |  Models: CLIP | YOLO | InsightFace |   |
                               |          Florence-2 | EasyOCR         |
                               +----------------------------------------+
```

## Core Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| API Gateway | REST endpoints for testing | FastAPI (uvicorn) |
| Local Worker | Pull-based job processing | Supabase Realtime + async Python |
| Processing Queue | Job queue with atomic claims | Supabase `processing_queue` table |
| Vector Storage | Embedding similarity search | Supabase pgvector |
| Model Registry | Dynamic model loading/swapping | Custom registry pattern |

## Directory Structure

```
Kizu-AI/
├── api/
│   ├── core/
│   │   ├── abstractions/          # Base classes for all model types
│   │   │   ├── base_embedder.py   # Image/text embedding interface
│   │   │   ├── base_detector.py   # Object detection interface
│   │   │   ├── base_face_model.py # Face detection/recognition interface
│   │   │   ├── base_ocr.py        # Text extraction interface
│   │   │   └── base_vlm.py        # Vision-language model interface
│   │   └── registry.py            # Model registry (lazy loading, singleton)
│   ├── models/                    # Concrete model implementations
│   │   ├── embedders/
│   │   │   └── clip_embedder.py   # OpenCLIP ViT-B/32
│   │   ├── detectors/
│   │   │   └── yolo_detector.py   # YOLOv8 (COCO 80 classes)
│   │   ├── faces/
│   │   │   └── insightface_model.py # InsightFace buffalo_l
│   │   ├── ocr/
│   │   │   └── easyocr_model.py   # EasyOCR (15+ languages)
│   │   └── vlm/
│   │       └── llava_model.py     # LLaVA 1.5 7B (optional)
│   ├── services/                  # Business logic layer
│   │   ├── process_service.py     # Image processing orchestration
│   │   ├── search_service.py      # Semantic search with NLP parsing
│   │   ├── clustering_service.py  # Face clustering (DBSCAN)
│   │   └── face_service.py        # Face operations
│   ├── routers/                   # API endpoint definitions
│   │   ├── process.py             # /process/* endpoints
│   │   ├── search.py              # /search/* endpoints
│   │   ├── faces.py               # /faces/* endpoints
│   │   ├── jobs.py                # /jobs/* endpoints
│   │   └── health.py              # /health, /models endpoints
│   ├── stores/                    # Data persistence
│   │   └── supabase_store.py      # pgvector operations
│   ├── workers/                   # Background processing
│   │   ├── local_worker.py        # Pull-based Supabase Realtime worker
│   │   ├── celery_app.py          # Legacy Celery config (optional)
│   │   └── tasks.py               # Legacy Celery tasks (optional)
│   └── utils/                     # Shared utilities
│       ├── device_utils.py        # CPU/GPU detection
│       ├── image_utils.py         # Image loading/processing
│       └── date_parser.py         # NLP date extraction
├── migrations/                    # Database schema (SQL)
│   ├── 001_ai_tables.sql          # Core AI tables (embeddings, faces, etc.)
│   └── 002_processing_queue.sql   # Pull-based job queue
├── scripts/
│   └── download_models.py         # Model downloader
├── model_cache/                   # Downloaded model weights
├── run_worker.py                  # Local worker entry point
├── docker-compose.yml             # Legacy stack (with Redis/Celery)
├── docker-compose.local.yml       # Recommended: local worker stack
└── docker-compose.prod.yml        # Production stack
```

## AI Models

All models are fully open source with no external API dependencies.

| Model | Implementation | Dimension | Use Case |
|-------|---------------|-----------|----------|
| **OpenCLIP ViT-B/32** | `CLIPEmbedder` | 512 | Semantic image search, text-to-image matching |
| **YOLOv8-m** | `YOLODetector` | N/A | Object detection (80 COCO classes) |
| **InsightFace buffalo_l** | `InsightFaceModel` | 512 | Face detection, recognition, clustering |
| **EasyOCR** | `EasyOCRModel` | N/A | Text extraction (15+ languages) |
| **LLaVA 1.5 7B** | `LLaVAModel` | N/A | Image descriptions (optional, resource-intensive) |

## API Endpoints

### Processing (`/api/v1/process`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/image` | Process single image (sync) |
| POST | `/batch` | Process multiple images (async job) |
| POST | `/webhook` | Real-time processing on upload |
| POST | `/reindex` | Reindex assets without embeddings |

**Operations**: `embedding`, `faces`, `objects`, `ocr`, `describe`, `all`

### Search (`/api/v1/search`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/` | Natural language semantic search |
| POST | `/by-face` | Find images containing a person |
| POST | `/by-object` | Find images with specific objects |
| GET | `/objects` | List detected object classes |

### Faces (`/api/v1/faces`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/cluster` | Trigger face clustering |
| GET | `/clusters` | List all face clusters |
| GET | `/clusters/{id}` | Get cluster details |
| POST | `/clusters/{id}/assign` | Link cluster to contact |
| POST | `/clusters/merge` | Merge multiple clusters |
| POST | `/clear-clustering` | Reset all clustering data |
| POST | `/backfill-thumbnails` | Generate missing face thumbnails |

### System (`/api/v1`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check with loaded models |
| GET | `/models` | Current model configuration |
| GET | `/models/available` | List registered models |
| POST | `/models/load/{type}/{name}` | Manually load a model |
| POST | `/models/unload/{type}/{name}` | Unload model to free memory |
| POST | `/models/unload-all` | Unload all models |

### Jobs (`/api/v1/jobs`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | List user's jobs |
| GET | `/{id}` | Get job status |
| DELETE | `/{id}` | Cancel pending job |

## Configuration

Environment variables (`.env`):

```bash
# Required
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# Models
MODEL_CACHE_DIR=./model_cache
CLIP_MODEL=ViT-B-32              # ViT-B-32, ViT-L-14, etc.
CLIP_PRETRAINED=openai           # openai, laion2b, etc.
YOLO_SIZE=m                      # n, s, m, l, x
FACE_MODEL_NAME=buffalo_l        # buffalo_l, buffalo_s
FACE_THRESHOLD=0.6               # Face match threshold

# Processing
MAX_IMAGE_SIZE=2048              # Resize images larger than this
BATCH_SIZE=8                     # Batch processing size

# Infrastructure
REDIS_URL=redis://localhost:6379/0
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Privacy (always enforced)
ALLOW_EXTERNAL_APIS=false        # External AI APIs disabled
```

## Adding New Models

The registry pattern enables swapping models without code changes.

### Step 1: Create Abstract Base (if new type)

```python
# api/core/abstractions/base_newtype.py
from abc import ABC, abstractmethod

class BaseNewType(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def unload_model(self) -> None:
        pass

    # Add type-specific abstract methods
```

### Step 2: Implement Concrete Model

```python
# api/models/newtypes/my_model.py
from api.core.abstractions import BaseNewType

class MyModel(BaseNewType):
    def __init__(self, model_variant: str = "default"):
        self._model_name = f"mymodel-{model_variant}"
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def load_model(self) -> None:
        if self._model is not None:
            return
        # Load model weights
        self._model = ...

    def unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
```

### Step 3: Register Model

```python
# api/models/newtypes/__init__.py
from api.core import ModelRegistry
from api.core.registry import ModelType
from .my_model import MyModel

ModelRegistry.register(
    ModelType.NEW_TYPE,
    "mymodel",
    MyModel,
    is_default=True
)
```

### Step 4: Use Model

```python
from api.core import ModelRegistry
from api.core.registry import ModelType

# Get default model (lazy loaded)
model = ModelRegistry.get(ModelType.NEW_TYPE)

# Get specific variant
model = ModelRegistry.get(ModelType.NEW_TYPE, "mymodel-large")

# Unload when done
ModelRegistry.unload(ModelType.NEW_TYPE, "mymodel")
```

## Local Worker

The recommended way to run Kizu-AI is using the pull-based local worker.

### Running the Worker

```bash
# Direct Python
python run_worker.py

# With options
python run_worker.py --worker-id my-nas --poll-interval 10 --debug

# Docker
docker-compose -f docker-compose.local.yml up worker
```

### How It Works

1. **Supabase Realtime subscription**: Worker subscribes to `processing_queue` INSERT events
2. **Atomic job claiming**: Uses `claim_processing_job()` RPC with `FOR UPDATE SKIP LOCKED`
3. **Polling fallback**: Also polls every N seconds in case Realtime misses events
4. **Auto-retry**: Failed jobs retry up to `max_attempts` (default: 3)
5. **Stale job recovery**: `reset_stale_processing_jobs()` resets stuck jobs

### Processing Queue Schema

```sql
processing_queue
├── id              UUID PRIMARY KEY
├── asset_id        UUID (references assets)
├── user_id         UUID (references auth.users)
├── status          TEXT ('pending'|'processing'|'completed'|'failed')
├── operations      TEXT[] (e.g., ['embedding','objects','faces'])
├── worker_id       TEXT (claims ownership)
├── priority        INT (1=highest, 10=lowest)
├── attempts        INT
├── result          JSONB
├── error           TEXT
└── timestamps      (created_at, started_at, completed_at)
```

### Why Pull-Based?

| Push (Webhook) | Pull (Realtime) |
|----------------|-----------------|
| Requires public endpoint | Works behind NAT/firewall |
| Single point of failure | Multiple workers anywhere |
| Complex networking | Simple: just outbound HTTPS |

## Performance Considerations

### Memory Management

- **Single-threaded processing**: One job at a time to avoid OOM
- **Lazy loading**: Models load on first use, not at startup
- **Explicit unload**: `/models/unload-all` frees GPU/CPU memory
- **Async I/O**: Network operations don't block model inference

### Hardware Requirements

| Configuration | RAM | GPU VRAM | Models Supported |
|--------------|-----|----------|------------------|
| Minimum (MacBook M1) | 8GB | N/A | CLIP, YOLO, InsightFace, EasyOCR |
| Recommended | 16GB+ | 8GB+ | All models including LLaVA |
| Production | 32GB+ | 12GB+ | All models with batch processing |

### Device Detection

The system auto-detects available hardware:

```python
# api/utils/device_utils.py
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"
```

InsightFace uses `CPUExecutionProvider` for cross-platform compatibility (M1/AMD/Intel).

## Privacy Philosophy

1. **All AI runs locally** - No data leaves the device
2. **No cloud AI providers** - No OpenAI, Google, AWS AI services
3. **Open source models only** - Fully auditable, no proprietary weights
4. **No external API calls** - `ALLOW_EXTERNAL_APIS=false` enforced
5. **User data isolation** - All queries filtered by `user_id`

## Integration Points

### Supabase Tables

| Table | Purpose |
|-------|---------|
| `image_embeddings` | CLIP vectors (pgvector) |
| `face_embeddings` | Face vectors with bounding boxes |
| `face_clusters` | Clustered face groups |
| `detected_objects` | YOLO detection results |
| `assets` | Source image metadata |
| `album_assets` | Thumbnail references |

### Processing Flow (Pull-Based)

1. User uploads image to Kizu Mobile/Web
2. Image stored in Supabase Storage
3. Database trigger inserts job into `processing_queue`
4. Local worker receives Realtime notification (or polls)
5. Worker claims job atomically, downloads image
6. AI processes image (embedding, faces, objects)
7. Results stored in Supabase pgvector tables
8. Job marked completed, search immediately available

### Search Flow

1. User enters natural language query
2. NLP parser extracts dates, locations, semantic terms
3. CLIP embeds semantic query
4. pgvector similarity search
5. Post-filter by date/location
6. Return ranked results with thumbnails
