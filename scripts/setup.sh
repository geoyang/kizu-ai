#!/bin/bash
# Kizu AI - Setup Script

set -e

echo "=========================================="
echo "  Kizu AI - Personal Image Intelligence  "
echo "=========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ "$python_version" < "3.10" ]]; then
    echo "Error: Python 3.10+ required (found $python_version)"
    exit 1
fi
echo "✓ Python $python_version"

# Check if in virtual environment
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created and activated"
else
    echo "✓ Using existing virtual environment"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r api/requirements.txt
echo "✓ Dependencies installed"

# Check for .env file
if [[ ! -f ".env" ]]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  Please edit .env and add your Supabase credentials!"
fi

# Create model cache directory
mkdir -p model_cache
echo "✓ Model cache directory created"

# Download models (optional)
echo ""
read -p "Download AI models now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading models (this may take a while)..."
    python scripts/download_models.py --all
    echo "✓ Models downloaded"
fi

echo ""
echo "=========================================="
echo "  Setup Complete!                        "
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env with your Supabase credentials"
echo "2. Run migrations in Supabase SQL editor"
echo "3. Start the API: uvicorn api.main:app --reload"
echo "4. Or use Docker: docker-compose up"
echo ""
echo "API docs will be available at: http://localhost:8000/docs"
