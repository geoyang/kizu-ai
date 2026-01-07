#!/usr/bin/env python3
"""Quick test script for Kizu AI API."""

import requests
import base64
import sys
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    r = requests.get(f"{API_URL}/health")
    print("Health:", r.json())
    return r.json()["status"] == "healthy"

def test_models():
    """List available models."""
    r = requests.get(f"{API_URL}/models/available")
    print("\nAvailable models:")
    for model_type, models in r.json().items():
        print(f"  {model_type}: {models}")

def process_image(image_path: str, features: list = None):
    """Process an image through the AI pipeline."""
    features = features or ["objects", "faces", "embedding"]

    # Read and encode image
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()

    print(f"\nProcessing: {image_path}")
    print(f"Features: {features}")

    r = requests.post(
        f"{API_URL}/api/v1/process/image",
        json={
            "image_base64": image_base64,
            "features": features,
        },
        timeout=120,  # Models may take time to load
    )

    if r.status_code == 200:
        result = r.json()
        print("\nResults:")

        if "objects" in result and result["objects"]:
            print(f"\n  Objects detected ({len(result['objects'])}):")
            for obj in result["objects"][:10]:
                print(f"    - {obj['class_name']}: {obj['confidence']:.1%}")

        if "faces" in result and result["faces"]:
            print(f"\n  Faces detected: {len(result['faces'])}")
            for i, face in enumerate(result["faces"]):
                print(f"    - Face {i+1}: confidence {face.get('confidence', 'N/A')}")

        if "text" in result and result["text"]:
            print(f"\n  Text extracted: {result['text'][:200]}...")

        if "embedding" in result:
            print(f"\n  Embedding: {len(result['embedding'])} dimensions")

        if "description" in result and result["description"]:
            print(f"\n  Description: {result['description']}")

        return result
    else:
        print(f"Error: {r.status_code}")
        print(r.text)
        return None

def search_images(query: str):
    """Search images by text query."""
    print(f"\nSearching for: '{query}'")
    r = requests.post(
        f"{API_URL}/api/v1/search",
        json={"query": query, "limit": 5},
    )
    if r.status_code == 200:
        results = r.json()
        print(f"Found {len(results.get('results', []))} results")
        return results
    else:
        print(f"Error: {r.status_code}")
        return None

if __name__ == "__main__":
    print("=" * 50)
    print("Kizu AI API Test")
    print("=" * 50)

    if not test_health():
        print("API not healthy!")
        sys.exit(1)

    test_models()

    # Process image if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if Path(image_path).exists():
            # All features
            features = ["objects", "faces", "ocr", "embedding"]
            if "--describe" in sys.argv:
                features.append("description")
            process_image(image_path, features)
        else:
            print(f"File not found: {image_path}")
    else:
        print("\n" + "-" * 50)
        print("Usage: python test_api.py <image_path> [--describe]")
        print("\nExample:")
        print("  python test_api.py ~/Photos/vacation.jpg")
        print("  python test_api.py ~/Photos/vacation.jpg --describe")
