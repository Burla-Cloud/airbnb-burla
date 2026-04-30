# Every public Airbnb, looked at all at once

A short writeup of what we built, what we found, and where the bodies
are buried. The live demo is at
[burla-cloud.github.io/airbnb-burla](https://burla-cloud.github.io/airbnb-burla/).

## What we did

Inside Airbnb publishes quarterly snapshots of every city it tracks:
listings, reviews, calendars, the works. We pulled the latest snapshot
for **119 cities** and then went back four quarters per city to see
what changed. That gave us:

- ~2.6M listing-snapshot rows
- 1.7M unique photo URLs
- 50.7M reviews

Every photo got CLIP-scored on CPU. The most suspicious shortlists for
each category (TVs mounted way too high, hectic kitchens, drug-den
vibes, real cats and dogs) were re-scored by **Claude Haiku Vision**
with strict JSON output. Reviews went through a three-tier funnel:
heuristic on every review, SBERT embedding cluster on the top 200K, and
Haiku scoring the top 12K for humor.

All of it ran on a **single dynamic Burla cluster** that scaled to ~1.7K
CPU workers for photo download and CLIP scoring, with **20 A100 GPUs**
in the same cluster running the SBERT embedding tier in parallel.
Anthropic API calls were rate-limited at 64 concurrent workers.

## Findings

Each finding card sorts every listing into a few groups, plots the
median 365-night calendar occupancy per group with a bootstrap 95% CI
(1000 resamples, n >= 100 per bucket), and accepts the hypothesis only
if no two bucket CIs overlap.

| Hypothesis | Verdict | Notes |
|---|---|---|
| brightness_quartile | accepted | Brighter photos correlate monotonically with higher occupancy. The boring expected one. |
| messiness_quartile | accepted | The CLIP-messiest quartile is *more* booked than the tidiest. Likely a confound: messy = lived-in stock photos, dorm-style listings, calendar-blocked-by-policy properties. |
| has_pet | accepted | Listings where Haiku confirmed a real cat or dog visible in the hero shot are more booked than those without. |
| absurd_photos | accepted | Listings with a Haiku-flagged "absurd photo" are more booked. Probably a personality signal. |
| tv_too_high | (qualitative only on the live site) | The visual section is the deliverable, not a numeric finding. |

The takeaway is that **photo style correlates with demand** in
directions that aren't always intuitive. Messiness in particular is a
"this is more likely a methodology artifact than a preference signal"
result, which is why the live site only reports it with a caveat.

## How the pipeline runs

Each stage is a small script that calls `remote_parallel_map` once or
twice and merges Parquet shards on `/workspace/shared`.

```
00 validate -> 01 listings + calendars -> 02a scrape extra photos ->
02b CLIP score -> 04 reviews 3-tier -> 05 correlate ->
05b Haiku review rerank -> 05c Haiku Vision photo validation ->
06 site artifacts (+ apply manual blocklist) -> 07 calendar
```

The unifying pattern is:

```python
from burla import remote_parallel_map

@dataclass
class WorkerArgs:
    batch_id: int
    input_path: str
    output_path: str

def worker(args: WorkerArgs) -> dict:
    rows = read_my_slice(args.input_path, args.batch_id)
    out  = do_real_work(rows)
    out.to_parquet(f"{args.output_path}/batch_{args.batch_id:05d}.parquet")
    return {"batch_id": args.batch_id, "n_rows": len(out)}

remote_parallel_map(
    worker, [WorkerArgs(...) for batch_id in range(N_BATCHES)],
    func_cpu=2, func_ram=8, max_parallelism=1000, grow=True,
)
merge_parquet_dir(...)
```

Three quirks that matter at this scale:

### Row-group-aligned reviews parquet

The 50.7M-row reviews dump is huge. Naively, every worker reads the
whole file just to take its slice. We rechunk reviews_raw.parquet with
`row_group_size = REVIEW_TIER1_BATCH_SIZE`, then each worker does
`pq.ParquetFile(path).read_row_group(batch_id)`. Worker memory drops by
~100x and tier-1 finishes in single-digit minutes wall time.

### Pre-staging model weights

CLIP, SBERT, and Haiku-validated downstream code all want to download
weights on first use. With 1000 cold workers that's 1000 simultaneous
HF downloads, which gets you rate-limited fast. Each worker:

1. Downloads weights once into `/workspace/shared/...` if missing.
2. Uses `fcntl.flock` to serialize a single per-node copy to local SSD.
3. Pins `torch.set_num_threads(1)` to avoid CPU thrash inside an
   already-saturated pod.

`scripts/preload_clip_weights.py` and `scripts/preload_st_weights.py`
do the warmup pass.

### Three-tier review funnel

Cost-aware top-K cascade.

```
50,686,612  raw reviews
       |    heuristic score on every review (CPU, 1K workers, ~5 min)
       v
   200,000  keep top by heuristic
       |    SBERT MiniLM embed + KMeans cluster
       |    (GPU, 20 A100s, ~5 min)
       v
    12,000  sampled across clusters for diversity
       |    Claude Haiku scoring -> humor_score, category, one_line
       |    (CPU rate-limited at 250, ~3 min, ~$0.65 in API calls)
       v
       250  final picks for the site
```

Saves ~99.98% of LLM cost vs running Claude on every review. Heuristic
surfaces phrases like "do not stay" and emoji clusters, embedding
clustering buys diversity (so the top 250 isn't 200 noise complaints
and 50 cleaning-fee complaints), and Haiku does the actual taste-tier
sort.

## What broke

This was an iteratively-debugged run. Things that bit us:

- **Datadome blocking** during photo scraping. We fell back to the
  hero photos already present in the listings dump, plus retried-with-
  backoff for a chunk of cities, and ended up with 1.7M photo URLs
  spread across the 119 cities and 4 snapshots.
- **GPU image stage was broken** by a missing `libGL.so.1` in the
  worker base image, so YOLOv8 returned all-zero detections. We pivoted
  to Claude Haiku Vision (`s05c_categories.py`) for the final TV /
  kitchen / drug-den / pet validation. The original `s03_images_gpu`
  stage is still wired in but its output is not consumed by the site.
- **HuggingFace rate limits** when 200+ workers simultaneously pulled
  the SBERT model. Fixed by pre-staging.
- **Burla "job not found" 500s** during long jobs. We added stage-level
  early-exit idempotency so re-running picks up where it left off
  without re-doing completed work.
- **Manual review.** Haiku Vision was good enough to make the photo
  galleries publishable, but a human still had to scroll through them
  and prune. `data/manual_blocklist.json` is the durable record.

## Reproducing

```bash
~/.burla/<your-account>/.venv/bin/pip install -e .
cp .env.example .env  # add ANTHROPIC_API_KEY
make all
```

The Makefile wires the stages in order and is resume-aware. If anything
breaks mid-run, re-running the same `make` target picks up where it
left off because every stage checkpoints to `/workspace/shared`.
