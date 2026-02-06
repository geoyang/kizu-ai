# Kizu AI - Personal Image Intelligence Engine

A Docker-based, locally-hosted AI system that provides intelligent image search, automatic tagging, face recognition with clustering, photo moments generation, and image optimization. Integrates with Kizu/Supabase backend.

## Features

- **Natural Language Search**: "photos of John at the beach"
- **Face Recognition**: Automatic face detection, clustering, and identification
- **Object Detection**: Find images containing specific objects
- **OCR**: Search for text within images
- **AI Descriptions**: Auto-generate image descriptions (Florence-2)
- **Photo Moments**: Auto-generated photo collections (location, people, events, "on this day")
- **Image Optimization**: Thumbnail and web version generation with EXIF orientation handling
- **Video Transcoding**: Background video processing with FFmpeg
- **100% Local**: All AI models run locally, no external APIs

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Docker Compose Stack                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐   ┌─────────────────────────┐   ┌──────────────────────┐  │
│  │   FastAPI     │   │    Unified AI Worker     │   │    Video Worker      │  │
│  │   Gateway     │   │                          │   │                      │  │
│  │   :8000       │   │  3 Job Types:            │   │  FFmpeg transcoding  │  │
│  │               │   │  • Image AI processing   │   │                      │  │
│  │  REST API for │   │  • Moments generation    │   └──────────────────────┘  │
│  │  testing &    │   │  • Image optimization    │                             │
│  │  management   │   │                          │                             │
│  └──────────────┘   └─────────────────────────┘                              │
│                                                                              │
│  AI Models: CLIP • YOLOv8 • InsightFace • EasyOCR • Florence-2              │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │ Pull-based (Realtime + polling)
                                     ▼
                           ┌──────────────────┐
                           │    Supabase       │
                           │  (pgvector, jobs) │
                           └──────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Supabase project

### Setup

```bash
cd ~/Development/Kizu-AI

# Edit configuration
cp .env.example .env
nano .env  # Add Supabase credentials

# Run database migrations
# Copy contents of migrations/*.sql to Supabase SQL editor

# Start with Docker (all services)
docker-compose up -d

# Or run components locally
uvicorn api.main:app --reload          # API server
python run_worker.py                    # Unified AI worker
python run_video_worker.py              # Video transcoding worker
```

### Running the Worker

```bash
# Default
python run_worker.py

# With custom options
python run_worker.py --worker-id my-nas --poll-interval 10 --debug
```

The unified worker handles all 3 job types automatically. See [ARCHITECTURE.md](ARCHITECTURE.md) for details.

## API Documentation

Interactive Swagger docs available at `http://localhost:8000/docs` when the API server is running.

### Key Endpoints

| Category | Endpoint | Description |
|----------|----------|-------------|
| Process | `POST /api/v1/process/image` | Process single image (sync) |
| Process | `POST /api/v1/process/batch` | Batch process images (async) |
| Search | `POST /api/v1/search` | Natural language semantic search |
| Search | `POST /api/v1/search/by-face` | Find images of a person |
| Faces | `POST /api/v1/faces/cluster` | Trigger face clustering |
| Faces | `GET /api/v1/faces/clusters` | List face clusters |
| Jobs | `GET /api/v1/jobs/{id}` | Check job status |
| System | `GET /health` | Health check |

See [AI-PROCESSING.md](AI-PROCESSING.md) for full endpoint documentation.

## AI Models

All models are fully open source with no external API dependencies.

| Component | Model | Dimension | Purpose |
|-----------|-------|-----------|---------|
| Embeddings | OpenCLIP ViT-B/32 | 512 | Semantic image search |
| Objects | YOLOv8-m | N/A | Object detection (80 COCO classes) |
| Faces | InsightFace buffalo_l | 512 | Face detection & recognition |
| OCR | EasyOCR | N/A | Text extraction (15+ languages) |
| VLM | Florence-2 | N/A | Image descriptions & captions |

### Model Abstraction

All models are behind abstract interfaces, making them easy to swap:

```python
# Current
ModelRegistry.register(ModelType.EMBEDDER, "clip", CLIPEmbedder, is_default=True)

# Future upgrade
ModelRegistry.register(ModelType.EMBEDDER, "siglip", SigLIPEmbedder, is_default=True)
```

## Hardware Requirements

| Configuration | RAM | GPU VRAM | Models Supported |
|--------------|-----|----------|------------------|
| Minimum (MacBook M1) | 8GB | N/A | CLIP, YOLO, InsightFace, EasyOCR |
| Recommended | 16GB+ | 8GB+ | All models including Florence-2 |
| Production | 32GB+ | 12GB+ | All models with batch processing |

## Project Structure

```
Kizu-AI/
├── api/
│   ├── core/
│   │   ├── abstractions/    # Base classes for all model types
│   │   └── registry.py      # Model registry (lazy loading)
│   ├── models/              # Concrete AI model implementations
│   │   ├── embedders/       # CLIP
│   │   ├── detectors/       # YOLO
│   │   ├── faces/           # InsightFace
│   │   ├── ocr/             # EasyOCR
│   │   └── vlm/             # Florence-2
│   ├── services/            # Business logic
│   │   ├── process_service.py     # Image processing orchestration
│   │   ├── search_service.py      # Semantic search with NLP
│   │   ├── clustering_service.py  # Face clustering (HDBSCAN)
│   │   ├── face_service.py        # Face operations
│   │   └── moments_service.py     # Photo moments generation
│   ├── routers/             # API endpoints
│   ├── stores/              # Vector storage (pgvector)
│   ├── workers/
│   │   ├── local_worker.py  # Unified worker (3 job types)
│   │   └── video_worker.py  # Video transcoding worker
│   ├── schemas/             # Pydantic DTOs
│   └── utils/               # Shared utilities
├── migrations/              # Database schema (SQL)
├── run_worker.py            # Unified worker entry point
├── run_video_worker.py      # Video worker entry point
├── docker-compose.yml       # Docker stack
├── ARCHITECTURE.md          # System architecture
└── AI-PROCESSING.md         # AI processing documentation
```

## Configuration

See `.env.example` for all configuration options:

- Supabase connection (URL, anon key, service role key)
- Model selection (CLIP variant, YOLO size, face model, VLM)
- Processing settings (image size, batch size)
- Privacy enforcement (`ALLOW_EXTERNAL_APIS=false`)

## License

Private - Kizu Project
