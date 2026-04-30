PY ?= /Users/josephperry/.burla/joeyper23/.venv/bin/python
SSL_CERT_FILE := $(shell $(PY) -c 'import certifi; print(certifi.where())' 2>/dev/null)
ENV := SSL_CERT_FILE=$(SSL_CERT_FILE)
RUN := $(ENV) $(PY) -m

.PHONY: all stage00 stage01 stage02a stage02a_sample stage02b stage02b_sample stage03 stage04 stage05 stage05b stage05c stage06 stage07 site site_serve site_data test clean help

help:
	@echo "Pipeline (each stage is resume-aware via /workspace/shared):"
	@echo "  make all              run every stage in order"
	@echo "  make stage00          validate every Inside Airbnb city"
	@echo "  make stage01          download + clean per-city listings + calendar"
	@echo "  make stage02a_sample  scrape ~5k listings, sanity check"
	@echo "  make stage02a         scrape every listing's photo manifest"
	@echo "  make stage02b_sample  CLIP-score ~10k images, sanity check"
	@echo "  make stage02b         CLIP-score every photo (~1.7M)"
	@echo "  make stage03          YOLOv8 GPU detection (deprecated, kept for completeness)"
	@echo "  make stage04          score reviews (heuristic + SBERT + Haiku)"
	@echo "  make stage05          bootstrap 95% CI correlations"
	@echo "  make stage05b         Haiku Vision validates WTF photo shortlist"
	@echo "  make stage05c         Haiku Vision validates TV / kitchen / drug-den / pet shortlists"
	@echo "  make stage06          build site/data/*.json + apply manual blocklist"
	@echo "  make stage07          derive occupancy_365 calendar demand proxy"
	@echo "  make site             serve site/ on localhost:8000"
	@echo "  make test             run pytest"
	@echo "  make clean            wipe data/interim and data/outputs"

stage00:
	$(RUN) src.stages.s00_validate_cities

stage01:
	$(RUN) src.stages.s01_download_listings

stage02a_sample:
	$(RUN) src.stages.s02a_scrape_photo_urls --sample 5000

stage02a:
	$(RUN) src.stages.s02a_scrape_photo_urls

stage02b_sample:
	$(RUN) src.stages.s02b_clip_score_photos --sample 10000

stage02b:
	$(RUN) src.stages.s02b_clip_score_photos

stage03:
	$(RUN) src.stages.s03_yolo_detect_photos

stage04:
	$(RUN) src.stages.s04_score_reviews --skip-ingest --skip-tier1

stage05:
	$(RUN) src.stages.s05_bootstrap_correlations

stage05b:
	$(RUN) src.stages.s05b_haiku_validate_wtf

stage05c:
	$(RUN) src.stages.s05c_haiku_validate_photos

stage06:
	$(RUN) src.stages.s06_build_site_data

stage07:
	$(RUN) src.stages.s07_calendar_demand

all: stage00 stage01 stage02a stage02b stage03 stage07 stage05b stage05c stage04 stage05 stage06 site_data
	@echo "All stages complete. Outputs in data/outputs/. Site data at site/data/."

site_data:
	@mkdir -p site/data
	@cp -f data/outputs/*.json site/data/ 2>/dev/null || true
	@echo "Synced data/outputs/*.json -> site/data/"

site: site_data site_serve

site_serve:
	cd site && python -m http.server 8000

test:
	$(PY) -m pytest tests/

clean:
	rm -rf data/interim/* data/outputs/*
	@echo "Cleaned data/interim/ and data/outputs/."
