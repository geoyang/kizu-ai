# Kizu AI - Personal Image Intelligence Engine

A Docker-based, locally-hosted AI system that provides intelligent image search, automatic tagging, face recognition with clustering, and natural language queries. Integrates with Kizu/Supabase backend.

## Features

- **Natural Language Search**: "photos of John at the beach"
- **Face Recognition**: Automatic face detection, clustering, and identification
- **Object Detection**: Find images containing specific objects
- **OCR**: Search for text within images
- **AI Descriptions**: Auto-generate image descriptions
- **100% Local**: All AI models run locally, no external APIs

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   FastAPI    │    │    Redis     │    │  Celery Workers  │  │
│  │   Gateway    │◄──►│   (Queue)    │◄──►│  (Processing)    │  │
│  │   :8000      │    │   :6379      │    │                  │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│                                                                  │
│  AI Models: CLIP • YOLOv8 • InsightFace • EasyOCR • LLaVA      │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Supabase        │
                    │  (pgvector)      │
                    └──────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Supabase project

### Setup

```bash
# Clone and setup
cd ~/Development/Kizu-AI
./scripts/setup.sh

# Edit configuration
cp .env.example .env
nano .env  # Add Supabase credentials

# Run database migrations
# Copy contents of migrations/001_ai_tables.sql to Supabase SQL editor

# Start with Docker
docker-compose up -d

# Or run locally
uvicorn api.main:app --reload
```

### Download Models

```bash
# Download all required models
python scripts/download_models.py --all

# Or download specific models
python scripts/download_models.py --clip --yolo --face --ocr

# Download LLaVA (optional, large)
python scripts/download_models.py --llava
```

## API Endpoints

### Search

```bash
# Natural language search
curl -X POST http://localhost:8000/api/v1/search \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "sunset photos from Hawaii", "limit": 20}'
```

### Process Images

```bash
# Process single image
curl -X POST http://localhost:8000/api/v1/process/image \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"asset_id": "uuid", "image_url": "https://...", "operations": ["all"]}'

# Batch processing
curl -X POST http://localhost:8000/api/v1/process/batch \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"asset_ids": ["uuid1", "uuid2"], "operations": ["embedding", "faces"]}'
```

### Face Clustering

```bash
# Trigger clustering
curl -X POST http://localhost:8000/api/v1/faces/cluster \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.6}'

# List clusters
curl http://localhost:8000/api/v1/faces/clusters \
  -H "Authorization: Bearer <token>"

# Assign cluster to contact
curl -X POST http://localhost:8000/api/v1/faces/clusters/<cluster_id>/assign \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"knox_contact_id": "uuid", "name": "John Smith"}'
```

## AI Models

| Component | Model | Purpose |
|-----------|-------|---------|
| Embeddings | OpenCLIP ViT-B/32 | Semantic image search |
| Objects | YOLOv8-m | Object detection (80 classes) |
| Faces | InsightFace buffalo_l | Face detection & recognition |
| OCR | EasyOCR | Text extraction |
| VLM | LLaVA 1.5 7B | Image descriptions (optional) |

### Model Abstraction

All models are behind abstract interfaces, making them easy to swap:

```python
# Current
ModelRegistry.register(ModelType.EMBEDDER, "clip", CLIPEmbedder, is_default=True)

# Future upgrade
ModelRegistry.register(ModelType.EMBEDDER, "siglip", SigLIPEmbedder, is_default=True)
```

## Hardware Requirements

### Minimum (MacBook M1)
- 8GB RAM
- Models: CLIP, YOLO, InsightFace, EasyOCR
- No LLaVA

### Recommended (Ryzen 7 + GPU)
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM (or AMD Radeon with ROCm)
- All models including LLaVA

## Project Structure

```
Kizu-AI/
├── api/
│   ├── core/
│   │   ├── abstractions/    # Base classes for models
│   │   └── registry.py      # Model registry
│   ├── models/              # Concrete implementations
│   │   ├── embedders/       # CLIP, SigLIP
│   │   ├── detectors/       # YOLO
│   │   ├── faces/           # InsightFace
│   │   ├── ocr/             # EasyOCR
│   │   └── vlm/             # LLaVA
│   ├── services/            # Business logic
│   ├── routers/             # API endpoints
│   ├── stores/              # Vector storage
│   └── workers/             # Celery tasks
├── migrations/              # Database schema
├── scripts/                 # Setup utilities
└── docker-compose.yml
```

## Configuration

See `.env.example` for all configuration options:

- Supabase connection
- Model selection
- Processing settings
- Redis configuration

## Extending

### Adding a New Embedder

```python
# api/models/embedders/siglip_embedder.py
from api.core.abstractions import BaseEmbedder

class SigLIPEmbedder(BaseEmbedder):
    @property
    def model_name(self) -> str:
        return "siglip"

    # Implement abstract methods...
```

### Adding a New Detector

```python
# api/models/detectors/detr_detector.py
from api.core.abstractions import BaseDetector

class DETRDetector(BaseDetector):
    # Implement abstract methods...
```

## Integration with Kizu App

1. **On Upload**: Call `/api/v1/process/webhook` from Supabase Edge Function
2. **Batch Processing**: Trigger `/api/v1/process/batch` for existing assets
3. **Search**: Use `/api/v1/search` for natural language queries
4. **Face Clustering**: Run periodically or on-demand

## License

Private - Kizu Project
