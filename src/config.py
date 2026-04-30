"""All magic numbers, paths, prompts, and policy live here.

Stages and tasks import from this module so we never reach into another file
to discover what budget we're holding to or which CLIP prompt drives Stage 3.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
OUTPUT_DIR = DATA_DIR / "outputs"

INSIDE_AIRBNB_INDEX_URL = "https://insideairbnb.com/get-the-data/"

VALIDATION_REPORT_PATH = OUTPUT_DIR / "validation_report.json"
LISTINGS_CLEAN_PATH = INTERIM_DIR / "listings_clean.parquet"
PHOTO_MANIFEST_PATH = INTERIM_DIR / "photo_manifest.parquet"
IMAGES_CPU_PATH = INTERIM_DIR / "images_cpu.parquet"
IMAGES_GPU_PATH = INTERIM_DIR / "images_gpu.parquet"
REVIEWS_SCORED_PATH = INTERIM_DIR / "reviews_scored.parquet"
CORRELATIONS_PATH = INTERIM_DIR / "correlations.parquet"
RUNTIME_LOG_PATH = OUTPUT_DIR / "runtime_log.json"

SHARED_ROOT = "/workspace/shared/airbnb"
SHARED_LISTINGS = f"{SHARED_ROOT}/listings_v2"
SHARED_PHOTOS = f"{SHARED_ROOT}/photos_v2"
SHARED_REVIEWS = f"{SHARED_ROOT}/reviews_v2"
SHARED_CALENDAR = f"{SHARED_ROOT}/calendar_v2"
SHARED_IMAGES_CPU = f"{SHARED_ROOT}/images_cpu_v2"
SHARED_IMAGES_GPU = f"{SHARED_ROOT}/images_gpu_v2"
SHARED_REVIEWS_TIER1 = f"{SHARED_ROOT}/reviews_tier1_v3"
SHARED_REVIEWS_TIER2 = f"{SHARED_ROOT}/reviews_tier2_v3"
SHARED_REVIEWS_TIER3 = f"{SHARED_ROOT}/reviews_tier3_v3"
SHARED_WTF_HAIKU = f"{SHARED_ROOT}/wtf_haiku_v1"

MAX_SNAPSHOTS_PER_CITY = 4
MIN_LISTINGS_PER_CITY = 0
MAX_SAMPLE_IMAGE_FAIL_RATIO = 1.0
SAMPLE_IMAGES_PER_CITY = 5

LISTINGS_BATCH_SIZE = 1000
SCRAPE_REQ_PER_SEC_PER_WORKER = 0.5
SCRAPE_MAX_PARALLELISM = 1200
SCRAPE_MIN_SUCCESS_RATE = 0.50
SCRAPE_RETRY_LIMIT = 2

# When True, pre-merge picture_urls from every snapshot into the photo
# manifest so we cover hero photos for listings that hosts later changed.
PHOTO_MANIFEST_INCLUDE_HISTORY = True

CPU_IMAGE_BATCH_SIZE = 700
CPU_IMAGE_MAX_PARALLELISM = 800

GPU_BATCH_SIZE = 32
GPU_MAX_PARALLELISM = 12  # GCP A100 quota in us-central1 is 16, leave headroom

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_PROMPTS = {
    "messy_room": "a messy cluttered room with stuff everywhere",
    "tv_above_fireplace": "a television mounted above a fireplace",
    "lots_of_plants": "a room full of houseplants",
    "bright_natural_light": "a bright sunny room with large windows and natural light",
    "bathroom": "a bathroom with toilet shower and sink",
    "kitchen": "a kitchen with stove counters and appliances",
    "bedroom": "a bedroom with a bed and pillows",
    "living_room": "a living room with a couch and tv",
    "minimalist": "a minimalist clean empty room",
    "pet_dog": "a dog visible in a vacation rental photo",
    "pet_cat": "a cat visible in a vacation rental photo",
    "pet_on_furniture": "a pet sleeping on the bed or sofa",
    "wtf_absurd_object": "a strange or absurd object in a vacation rental",
    "wtf_unsettling_decor": "an unsettling or creepy decoration",
    "wtf_unusual_scene": "an unusual or surprising scene for a rental",
    "wtf_does_not_belong": "something that does not belong in an Airbnb",
}

TOP_N_PER_AXIS = {
    "clip_tv_above_fireplace": 25_000,
    "clip_messy_room": 20_000,
    "clip_lots_of_plants": 8_000,
    "clip_pet_dog": 8_000,
    "clip_pet_cat": 8_000,
    "clip_pet_on_furniture": 4_000,
}

YOLO_MODEL = "yolov8x.pt"
YOLO_DEVICE = "cuda"
YOLO_TARGET_CLASSES = {
    "tv": 62,
    "person": 0,
    "potted plant": 58,
    "couch": 57,
    "bed": 59,
    "toilet": 61,
    "cat": 15,
    "dog": 16,
}

WTF_CLIP_PROMPT_KEYS = (
    "wtf_absurd_object", "wtf_unsettling_decor",
    "wtf_unusual_scene", "wtf_does_not_belong",
)
WTF_TOP_K_FOR_HAIKU = 10_000
WTF_HAIKU_BATCH_SIZE = 10
WTF_HAIKU_MAX_PARALLELISM = 200
WTF_MIN_LABEL_CLUSTER_SIZE = 6
WTF_TOP_PHOTOS_PER_CLUSTER = 24

PETS_TOP_K = 200
PETS_MIN_COMBINED_SCORE = 0.0

REVIEW_TIER1_BATCH_SIZE = 5000
REVIEW_TIER1_MAX_PARALLELISM = 2000
REVIEW_TIER2_TOP_K = 250_000
REVIEW_TIER2_NUM_CLUSTERS = 40
REVIEW_TIER3_TOP_K = 12_000
REVIEW_TIER3_BATCH_SIZE = 50
REVIEW_TIER3_MAX_PARALLELISM = 250

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_MAX_TOKENS = 200

REVIEW_HEURISTIC_KEYWORDS = [
    "cockroach", "roach", "bedbug", "rat", "mouse", "mice",
    "ghost", "haunted", "scary", "terrifying",
    "police", "cops", "arrest", "fight", "blood",
    "toilet", "plumbing", "leak", "flood", "smell",
    "fire", "smoke alarm", "alarm",
    "neighbor", "loud", "argue",
    "dog", "cat", "snake", "spider",
    "weed", "drugs", "needles",
    "host showed up", "knocked on the door", "midnight",
    "however", " but ", "BUT ",
]

REVIEW_HUMOR_CATEGORIES = [
    "This escalated quickly",
    "Five stars but terrifying",
    "Passive aggressive poetry",
    "Host said what now",
    "Pets and wildlife",
    "Bugs and pests",
    "Plumbing and smells",
    "Noise complaint hall of fame",
    "Cleanliness mystery",
    "Weather and building drama",
    "Lost in translation",
    "Photo did not match",
    "Honest disaster",
    "Not funny",
]

HYPOTHESES = [
    ("brightness_quartile", "demand_proxy"),
    ("plant_count_bucket", "demand_proxy"),
    ("messiness_quartile", "demand_proxy"),
    ("tv_too_high", "demand_proxy"),
    ("has_pet", "demand_proxy"),
    ("is_wtf", "demand_proxy"),
]
BOOTSTRAP_RESAMPLES = 1000
MIN_BUCKET_N = 100

OUTPUT_TOP_K = {
    "worst_tv_placements": 60,
    "pets_in_photos": 60,
    "hectic_kitchens": 40,
    "drug_den_vibes": 40,
    "funniest_reviews": 250,
}

PIPELINE_HARD_CAP_USD = float(os.environ.get("PIPELINE_HARD_CAP_USD", 2500.0))
PIPELINE_HARD_CAP_SOFT_USD = float(os.environ.get("PIPELINE_HARD_CAP_SOFT_USD", 2000.0))
PIPELINE_HARD_CAP_HOURS = float(os.environ.get("PIPELINE_HARD_CAP_HOURS", 40.0))
PIPELINE_PHASE1_CAP_USD = float(os.environ.get("PIPELINE_PHASE1_CAP_USD", 1200.0))

STAGE_BUDGETS = {
    "s00_validate": {"hours": 0.5, "usd": 10.0},
    "s01_listings": {"hours": 1.5, "usd": 50.0},
    "s02a_scrape": {"hours": 8.0, "usd": 600.0},
    "s02b_images_cpu": {"hours": 6.0, "usd": 400.0},
    "s03_images_gpu": {"hours": 6.0, "usd": 250.0},
    "s04_reviews": {"hours": 10.0, "usd": 500.0},
    "s05_correlate": {"hours": 1.0, "usd": 20.0},
    "s06_artifacts": {"hours": 1.0, "usd": 20.0},
    "s07_calendar": {"hours": 2.0, "usd": 80.0},
    "s07_wtf_haiku": {"hours": 3.0, "usd": 200.0},
}
