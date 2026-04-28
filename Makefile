PY ?= /Users/josephperry/.burla/joeyper23/.venv/bin/python
SSL_CERT_FILE := $(shell $(PY) -c 'import certifi; print(certifi.where())' 2>/dev/null)
ENV := SSL_CERT_FILE=$(SSL_CERT_FILE)
RUN := $(ENV) $(PY) -m

.PHONY: all stage00 stage01 stage02a stage02a_sample stage02b stage02b_sample stage03 stage04 stage05 stage05b stage06 stage07 site site_serve site_data test clean help

help:
	@echo "Targets:"
	@echo "  make all              run every stage in order, resume-aware"
	@echo "  make stage00          validate every Inside Airbnb city"
	@echo "  make stage01          download + clean per-city listings + calendar"
	@echo "  make stage02a_sample  scrape ~5k listings, sanity check"
	@echo "  make stage02a         scrape ~3M listings (slow, hours)"
	@echo "  make stage02b_sample  CLIP-score ~10k images, sanity check"
	@echo "  make stage02b         CLIP-score ~25-35M images"
	@echo "  make stage03          GPU object detection on top candidates"
	@echo "  make stage04          three-tier review scoring"
	@echo "  make stage05          correlations with bootstrap CIs"
	@echo "  make stage05b         WTF photo Haiku rerank + cluster"
	@echo "  make stage06          write site JSON artifacts"
	@echo "  make stage07          derive occupancy_365 + price volatility"
	@echo "  make site             serve site/ on localhost:8000"
	@echo "  make test             run pytest"
	@echo "  make clean            wipe data/interim and data/outputs"

stage00:
	$(RUN) src.stages.s00_validate

stage01:
	$(RUN) src.stages.s01_listings

stage02a_sample:
	$(RUN) src.stages.s02a_scrape --sample 5000

stage02a:
	$(RUN) src.stages.s02a_scrape

stage02b_sample:
	$(RUN) src.stages.s02b_images_cpu --sample 10000

stage02b:
	$(RUN) src.stages.s02b_images_cpu

stage03:
	$(RUN) src.stages.s03_images_gpu

stage04:
	$(RUN) src.stages.s04_reviews --skip-ingest --skip-tier1

stage05:
	$(RUN) src.stages.s05_correlate

stage05b:
	$(RUN) src.stages.s05b_wtf_haiku

stage05c:
	$(RUN) src.stages.s05c_categories

stage06:
	$(RUN) src.stages.s06_artifacts

stage07:
	$(RUN) src.stages.s07_calendar

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
