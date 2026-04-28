# Airbnb x Burla -- viral summary

(All numbers below are computed from data/outputs/*.json. Regenerated every run.)

## Headline numbers

- 1,740,077 Airbnb listings worldwide (Inside Airbnb, latest snapshot per city)
- 1,945,032 photo URLs scraped from public listing pages
- 1,710,664 images CLIP-scored on Burla CPU
- 12,640 images run through YOLOv8 on Burla A100s
- 50,686,612 reviews heuristic-scored, top 250 sent through Claude

## What we found

### TVs in places no one should mount a TV

Top-60 listings where YOLO confirmed a TV in the upper half of the photo
and CLIP rated the image high on "TV mounted above a fireplace."

### Messiest photos a host actually posted

Top-50 listings, ranked by CLIP score against "a messy cluttered room
with stuff everywhere."

### Plant-maximalist Airbnbs

Top-30 listings combining CLIP "room full of houseplants" with YOLO
potted plant counts.

### Pets in photos

Top-60 listings where YOLO confirmed a cat or dog and CLIP scored high
on the pet prompts.

### Photos that made us go "wait, what"

360 photos across 32 Haiku-named clusters of genuinely
absurd, unsettling, or out-of-place hosting choices. Top clusters: closet bedroom, kitchen in bedroom, other, creepy doll, cramped bedroom.

### The funniest reviews

Top-250 reviews surfaced by 3-tier funnel (heuristic -> embedding cluster
-> Claude humor score).

## What held up under bootstrap

- has_pet
- is_wtf
- messiness_quartile

## What did not survive

- brightness_quartile
- plant_count_bucket

## Replication

Repo: airbnb-burla
Runtime: 22.9 hours wall time, peak 1741 Burla workers.
