#!/usr/bin/env python3
"""Download and cache AI models for offline use."""

import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "./model_cache")


def download_clip(model_name: str = "ViT-B-32", pretrained: str = "openai"):
    """Download OpenCLIP model."""
    logger.info(f"Downloading CLIP: {model_name}/{pretrained}")
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        cache_dir=CACHE_DIR
    )
    logger.info("CLIP model downloaded")


def download_yolo(model_size: str = "m"):
    """Download YOLOv8 model."""
    logger.info(f"Downloading YOLOv8-{model_size}")
    from ultralytics import YOLO

    model = YOLO(f"yolov8{model_size}.pt")
    logger.info("YOLO model downloaded")


def download_insightface(model_name: str = "buffalo_l"):
    """Download InsightFace model."""
    logger.info(f"Downloading InsightFace: {model_name}")
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name=model_name, root=CACHE_DIR)
    app.prepare(ctx_id=-1, det_size=(640, 640))
    logger.info("InsightFace model downloaded")


def download_easyocr(languages: list = None):
    """Download EasyOCR model."""
    languages = languages or ["en"]
    logger.info(f"Downloading EasyOCR: {languages}")
    import easyocr

    reader = easyocr.Reader(languages, gpu=False)
    logger.info("EasyOCR model downloaded")


def download_llava():
    """Download LLaVA model (large, optional)."""
    logger.info("Downloading LLaVA (this may take a while)...")
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

    model_id = "llava-hf/llava-1.5-7b-hf"
    LlavaNextProcessor.from_pretrained(model_id, cache_dir=CACHE_DIR)
    LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir=CACHE_DIR,
        low_cpu_mem_usage=True
    )
    logger.info("LLaVA model downloaded")


def main():
    parser = argparse.ArgumentParser(description="Download AI models")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all models"
    )
    parser.add_argument(
        "--clip",
        action="store_true",
        help="Download CLIP model"
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Download YOLO model"
    )
    parser.add_argument(
        "--face",
        action="store_true",
        help="Download InsightFace model"
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Download EasyOCR model"
    )
    parser.add_argument(
        "--llava",
        action="store_true",
        help="Download LLaVA model (large)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=CACHE_DIR,
        help="Cache directory for models"
    )

    args = parser.parse_args()

    global CACHE_DIR
    CACHE_DIR = args.cache_dir
    os.makedirs(CACHE_DIR, exist_ok=True)

    if not any([args.all, args.clip, args.yolo, args.face, args.ocr, args.llava]):
        parser.print_help()
        sys.exit(1)

    if args.all or args.clip:
        download_clip()

    if args.all or args.yolo:
        download_yolo()

    if args.all or args.face:
        download_insightface()

    if args.all or args.ocr:
        download_easyocr()

    if args.llava:  # Not included in --all due to size
        download_llava()

    logger.info("Model download complete!")


if __name__ == "__main__":
    main()
