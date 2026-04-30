# airbnb-burla

Looking at every public Airbnb listing in Inside Airbnb's open data dump,
all at once, on Burla.

Live site: https://burla-cloud.github.io/airbnb-burla/

## What we did

Every public listing in Inside Airbnb's open dump, **119 cities, 4
quarterly snapshots**. We CLIP-scored **1.7M photos**, took the most
suspicious shortlists, and had **Claude Haiku Vision** double-check each
one. We also heuristic-scored every review and reranked the weirdest 12K
through Haiku.

Everything was parallelized on **Burla** on a single dynamic cluster:

- ~1.7K CPU workers for photo download and CLIP scoring
- 20 A100 GPUs running embedding clusters in parallel on the same cluster
- Claude Haiku validation rate-limited on top

## What's on the site

- **Listings with drug-den vibes** - CLIP shortlist, Haiku Vision picks
  the photos that look less like an Airbnb and more like an opium den.
- **Most hectic kitchens** - same funnel, kitchen edition.
- **Real cats and dogs** - CLIP finds pet-shaped pixels, Haiku Vision
  rejects throw pillows and rugs that just looked vaguely animal.
- **Worst TV placements** - TVs mounted way too high, validated by Haiku.
- **Funniest reviews** - 50.7M reviews -> regex shortlist -> 200K SBERT
  embeddings clustered for diversity -> Haiku scores the top 12K.
- **Findings** - four hypotheses (TV height, brightness, pet visible,
  absurd photo) tested against 365-night calendar occupancy with
  bootstrap 95% CIs.
- **World map** - every flagged listing on a Leaflet map.

## Quickstart

```bash
~/.burla/<your-account>/.venv/bin/pip install -e .

cp .env.example .env
# edit .env, drop in ANTHROPIC_API_KEY

make all
```

Each stage is independently runnable and resume-aware (checkpointed to
`/workspace/shared` on Burla):

```bash
make stage00          # validate every Inside Airbnb city
make stage02b_sample  # 10K-image CLIP sanity check
make stage04          # 3-tier review scoring (50.7M -> 200K -> 12K)
make stage05c         # Haiku Vision validates CLIP shortlists
make stage06          # write site/data/*.json
```

## Pipeline

| Stage | What | Where it runs |
|---|---|---|
| 00 | Validate every Inside Airbnb city | local |
| 01 | Download + clean per-city listings + calendars | Burla CPU |
| 02a | Scrape extra photo URLs from `airbnb.com/rooms/<id>` | Burla CPU |
| 02b | CLIP-score every photo | Burla CPU (~1.7K workers) |
| 03 | YOLOv8 GPU stage (deprecated, kept for completeness) | Burla GPU |
| 04 | 3-tier review scoring (heuristic + SBERT + Claude) | Burla CPU + 20 A100s for the embedding tier |
| 05 | Bootstrap 95% CI correlations | Burla CPU |
| 05b | Haiku rerank weirdest 12K reviews | Burla CPU (rate-limited) |
| 05c | Haiku Vision validates CLIP shortlists for TVs / kitchens / drug-den / pets | Burla CPU (rate-limited) |
| 06 | Build `site/data/*.json` and apply manual blocklist | local |
| 07 | Derive `occupancy_365` (calendar occupancy demand proxy) | Burla CPU |

## Layout

```
src/
  config.py          # cities, top-N, prompts, budgets
  stages/            # one orchestrator per pipeline stage
  tasks/             # Burla-serialized worker functions
  lib/               # io, budget, retries, Inside Airbnb client
site/                # static HTML/CSS/JS, fed by data/outputs
data/
  manual_blocklist.json   # human review: dropped IDs + pinned-top order
  outputs/                # final JSON the site reads (committed)
  raw/, interim/          # gitignored
scripts/
  apply_manual_blocklist.py   # post-process s06 outputs against the blocklist
  preload_clip_weights.py     # pre-stage CLIP weights to /workspace/shared
  preload_st_weights.py       # pre-stage SBERT weights to /workspace/shared
```

## How it talks to Burla

Every stage is a small script that calls `remote_parallel_map` once or
twice. The three biggest fan-outs:

```python
from burla import remote_parallel_map

# s02b: CLIP-score every photo on CPU (1K parallelism at peak)
remote_parallel_map(
    score_batch, batch_args,
    func_cpu=2, func_ram=8,
    max_parallelism=1000, grow=True,
)

# s04 tier 2: SBERT-embed top 200K reviews on GPU (20 A100s)
remote_parallel_map(
    embed_batch, embed_args,
    func_cpu=2, func_ram=8, max_parallelism=200,
    grow=True,
)

# s05c: Haiku Vision validates CLIP shortlists (rate-limited)
remote_parallel_map(
    validate_pet, pet_batches,
    func_cpu=2, func_ram=8, max_parallelism=64,
    grow=True,
)
```

Burla pickles each worker, ships it to the cluster, and runs N copies in
parallel against a shared `/workspace/shared` filesystem. No Docker, no
Kubernetes, no orchestration glue.

## Manual review

`data/manual_blocklist.json` is the human override layer. Two parts:

- **`by_city_name`** - listings flagged as not passing visual review.
  `scripts/apply_manual_blocklist.py` resolves them to listing IDs,
  removes them from `site/data/*.json` and `data/outputs/*.json`,
  rebuilds `world_map.json`, and persists the IDs so re-runs stay clean.
- **`pinned_top`** - per-section ordered listing IDs that should appear
  first in the grid (e.g. the three best TV-too-high listings).

The s06 stage runs the apply script as a final post-process, so a fresh
`make all` always reflects the latest manual review.

## Caveats

- The demand proxy is `occupancy_365` (median calendar occupancy over
  the next 365 nights). It counts blocked nights as well as booked
  ones, which is the standard Inside Airbnb caveat.
- Inside Airbnb anonymizes locations within ~150m, so the world map
  shows neighborhoods, not exact addresses.
- The original GPU stage (`s03_images_gpu`, YOLOv8) discovered a
  `libGL.so.1` packaging issue mid-run. We pivoted to Claude Haiku
  Vision (`s05c_categories`) for the final TV / kitchen / drug-den /
  pet validation. The s03 stage is still wired in `make all` but its
  output is not consumed by the live site.

## License

MIT.
