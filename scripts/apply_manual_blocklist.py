"""Apply data/manual_blocklist.json to the per-section JSON outputs.

For each (section, city, name) entry, find a matching item in the section's
JSON, capture its listing_id into ``by_listing_id`` for permanence, then
remove the item from both ``site/data/<section>.json`` and
``data/outputs/<section>.json``. Rebuild ``world_map.json`` afterwards.

Run this from the repo root::

    python -m scripts.apply_manual_blocklist

Idempotent. Safe to re-run after adding new entries.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BLOCKLIST_PATH = REPO_ROOT / "data" / "manual_blocklist.json"
SITE_DATA = REPO_ROOT / "site" / "data"
OUT_DATA = REPO_ROOT / "data" / "outputs"

PHOTO_SECTIONS = (
    "worst_tv_placements",
    "hectic_kitchens",
    "drug_den_vibes",
    "pets_in_photos",
)


_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _WS.sub(" ", s)
    return s.casefold().strip()


def _load(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _remove_from_section(section_id: str, entries):
    """For each (city, name) in entries, remove a matching item from
    site/data/<section>.json and data/outputs/<section>.json. Return
    the list of removed listing_ids."""
    removed_ids: list[int] = []
    for path in (SITE_DATA / f"{section_id}.json", OUT_DATA / f"{section_id}.json"):
        if not path.exists():
            continue
        payload = _load(path)
        items = payload.get("items", [])
        keep = []
        block_keys = {(_norm(e["city"]), _norm(e["name"])) for e in entries}
        block_id_set = set()
        for it in items:
            key = (_norm(it.get("city", "")), _norm(it.get("name", "")))
            if key in block_keys:
                lid = it.get("listing_id")
                if lid is not None:
                    block_id_set.add(int(lid))
                continue
            keep.append(it)
        payload["items"] = keep
        payload["n"] = len(keep)
        _dump(path, payload)
        removed_ids.extend(sorted(block_id_set))
    return sorted(set(removed_ids))


def _apply_pin_top(pinned: dict[str, list[int]]) -> None:
    """For each section, reorder items so that any listing_id present in the
    pin list appears first, in the order given. Items not in the pin list keep
    their existing relative order. Idempotent."""
    for section_id, raw_ids in (pinned or {}).items():
        ordered_ids = [int(x) for x in (raw_ids or [])]
        if not ordered_ids:
            continue
        rank = {lid: i for i, lid in enumerate(ordered_ids)}
        for path in (SITE_DATA / f"{section_id}.json",
                      OUT_DATA / f"{section_id}.json"):
            if not path.exists():
                continue
            payload = _load(path)
            items = payload.get("items", [])
            pinned_items = [None] * len(ordered_ids)
            tail: list[dict] = []
            for it in items:
                lid = it.get("listing_id")
                try:
                    lid_i = int(lid) if lid is not None else None
                except (TypeError, ValueError):
                    lid_i = None
                if lid_i is not None and lid_i in rank:
                    pinned_items[rank[lid_i]] = it
                else:
                    tail.append(it)
            head = [it for it in pinned_items if it is not None]
            n_pinned = len(head)
            payload["items"] = head + tail
            payload["n"] = len(payload["items"])
            _dump(path, payload)
            print(f"[blocklist] {path.name}: pinned {n_pinned} listing(s) to top")


def _rebuild_world_map():
    points = []
    for stype in PHOTO_SECTIONS:
        path = SITE_DATA / f"{stype}.json"
        if not path.exists():
            continue
        payload = _load(path)
        for it in payload.get("items", []):
            if it.get("lat") is None or it.get("lng") is None:
                continue
            lid = str(it.get("listing_id", ""))
            points.append({
                "type": stype,
                "lat": float(it["lat"]),
                "lng": float(it["lng"]),
                "listing_id": lid,
                "listing_url": it.get("listing_url") or f"https://www.airbnb.com/rooms/{lid}",
            })
    payload = {
        "title": "Every flagged Airbnb in the demo, on a Leaflet map",
        "n": len(points),
        "points": points,
    }
    for path in (SITE_DATA / "world_map.json", OUT_DATA / "world_map.json"):
        _dump(path, payload)
    return len(points)


def main() -> int:
    if not BLOCKLIST_PATH.exists():
        print(f"[blocklist] no manual_blocklist.json at {BLOCKLIST_PATH}; nothing to do.")
        return 0
    blocklist = _load(BLOCKLIST_PATH)
    by_city_name = blocklist.get("by_city_name", [])
    by_listing_id = set(int(x) for x in blocklist.get("by_listing_id", []))

    by_section: dict[str, list[dict]] = {}
    for entry in by_city_name:
        sec = entry.get("section")
        if not sec:
            continue
        by_section.setdefault(sec, []).append(entry)

    n_removed_total = 0
    for section_id, entries in by_section.items():
        ids = _remove_from_section(section_id, entries)
        n_removed_total += len(entries)
        by_listing_id.update(ids)
        print(f"[blocklist] {section_id}: stripped {len(entries)} entries "
              f"(persisted ids: {len(ids)})")

    # Drop any item whose listing_id is in the persistent id blocklist (covers
    # cases where the city/name moved between snapshots but the listing_id
    # stays stable).
    if by_listing_id:
        for section_id in PHOTO_SECTIONS:
            for path in (SITE_DATA / f"{section_id}.json",
                          OUT_DATA / f"{section_id}.json"):
                if not path.exists():
                    continue
                payload = _load(path)
                before = len(payload.get("items", []))
                payload["items"] = [
                    it for it in payload.get("items", [])
                    if int(it.get("listing_id", -1)) not in by_listing_id
                ]
                after = len(payload["items"])
                payload["n"] = after
                _dump(path, payload)
                if after < before:
                    print(f"[blocklist] {path.name}: id-blocked {before - after} more")

    # Persist resolved ids back into the blocklist file.
    blocklist["by_listing_id"] = sorted(by_listing_id)
    _dump(BLOCKLIST_PATH, blocklist)

    # Apply pinned-top reordering after we've stripped blocked items so we
    # don't pin something we're also trying to drop.
    _apply_pin_top(blocklist.get("pinned_top") or {})

    n_points = _rebuild_world_map()
    print(f"[blocklist] world_map.json rebuilt with {n_points} points")
    print(f"[blocklist] DONE. {n_removed_total} entries applied across "
          f"{len(by_section)} sections; {len(by_listing_id)} ids persisted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
