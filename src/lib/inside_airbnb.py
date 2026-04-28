"""Discover and download Inside Airbnb city dumps.

Inside Airbnb's get-the-data page is built on Gatsby; every dataset (current +
archived) is served as a single static-query JSON at:

    https://insideairbnb.com/page-data/get-the-data/page-data.json
        -> staticQueryHashes (e.g. ["3008393846", "3649515864", "63159454"])
        -> https://insideairbnb.com/page-data/sq/d/<hash>.json
        -> data.allData.datasets[]  with every (city, snapshot) tuple

This dodges the JS "show archived data" toggle that breaks the regex scrape.
For each (city, snapshot) we construct the listings/reviews/calendar URLs from
the standard Inside Airbnb path layout:

    https://data.insideairbnb.com/<country>/<region>/<city>/<YYYY-MM-DD>/data/<file>.csv.gz
"""
from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

import requests

from ..config import INSIDE_AIRBNB_INDEX_URL


_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
)
_LISTINGS_FILE = "listings.csv.gz"
_REVIEWS_FILE = "reviews.csv.gz"
_CALENDAR_FILE = "calendar.csv.gz"

_PAGE_DATA_URL = "https://insideairbnb.com/page-data/get-the-data/page-data.json"
_STATIC_QUERY_URL = "https://insideairbnb.com/page-data/sq/d/{hash}.json"

_DATA_URL_RE = re.compile(
    r"https?://data\.insideairbnb\.com/"
    r"([^/\s\"']+)/"        # country
    r"([^/\s\"']+)/"        # region
    r"([^/\s\"']+)/"        # city slug
    r"(\d{4}-\d{2}-\d{2})/" # snapshot
    r"data/(listings|reviews|calendar)\.csv\.gz",
    re.IGNORECASE,
)


@dataclass
class City:
    """One (city, snapshot) row from Inside Airbnb's data table.

    Phase 1 of the wider pull treats each (city, snapshot_date) tuple as its
    own work unit. The same physical city appears multiple times here, once
    per archived snapshot date.
    """
    city: str
    country: str
    region: str
    snapshot_date: str
    listings_url: str
    reviews_url: str
    calendar_url: str = ""

    def slug(self) -> str:
        s = re.sub(r"[^A-Za-z0-9]+", "-", self.city.lower()).strip("-")
        return f"{self.country.lower()}_{s}_{self.snapshot_date}"


def _http_get(url: str, timeout: int = 60) -> requests.Response:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": _USER_AGENT})
    r.raise_for_status()
    if r.encoding is None or r.encoding.lower() == "iso-8859-1":
        r.encoding = "utf-8"
    return r


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower()).strip("-")


def _build_data_root(country: str, region: Optional[str], link: str) -> str:
    parts = [_slugify(country)]
    if region:
        parts.append(_slugify(region))
    parts.append(_slugify(link))
    return "https://data.insideairbnb.com/" + "/".join(parts) + "/"


def _datasets_from_static_query() -> List[dict]:
    """Resolve the get-the-data page's static-query JSON and return its datasets list.

    Returns [] if Gatsby's static-query layout has changed; callers fall back
    to the legacy regex scrape.
    """
    try:
        page = _http_get(_PAGE_DATA_URL).json()
    except Exception:
        return []
    hashes = page.get("staticQueryHashes", [])
    for h in hashes:
        try:
            sq = _http_get(_STATIC_QUERY_URL.format(hash=h)).json()
        except Exception:
            continue
        all_data = sq.get("data", {}).get("allData") or sq.get("data", {}).get("datasets")
        if isinstance(all_data, dict):
            datasets = all_data.get("datasets", [])
        elif isinstance(all_data, list):
            datasets = all_data
        else:
            continue
        if datasets and isinstance(datasets, list) and \
                isinstance(datasets[0], dict) and "publishDate" in datasets[0]:
            return datasets
    return []


def discover_all_cities(max_snapshots_per_city: int = 4) -> List[City]:
    """Return every (city, snapshot) Inside Airbnb publishes.

    For each unique (country, region, city), keeps the ``max_snapshots_per_city``
    most recent snapshots (default 4 = roughly the last 12 months at quarterly
    cadence). Cities with fewer snapshots are kept as-is.
    """
    datasets = _datasets_from_static_query()
    if datasets:
        return _from_static_query(datasets, max_snapshots_per_city)
    return _from_regex_fallback(max_snapshots_per_city)


def _from_static_query(datasets: List[dict], max_per_city: int) -> List[City]:
    by_city: dict[tuple[str, str, str], list[tuple[str, str, dict]]] = {}
    for d in datasets:
        if not d.get("visualisationPublishStatus", True):
            continue
        country = (d.get("country") or "").strip()
        region = (d.get("region") or d.get("regionShort") or "").strip()
        link = (d.get("link") or _slugify(d.get("city") or "")).strip()
        snap = (d.get("publishDate") or "").strip()
        if not (country and link and snap):
            continue
        key = (country, region, link)
        by_city.setdefault(key, []).append((snap, d.get("city") or link, d))

    cities: list[City] = []
    for (country, region, link), rows in by_city.items():
        rows.sort(key=lambda r: r[0], reverse=True)
        for snap, display_city, d in rows[:max_per_city]:
            data_root = d.get("dataRoot") or _build_data_root(country, region, link)
            if not data_root.endswith("/"):
                data_root += "/"
            cities.append(City(
                city=display_city,
                country=country,
                region=region,
                snapshot_date=snap,
                listings_url=f"{data_root}{snap}/data/{_LISTINGS_FILE}",
                reviews_url=f"{data_root}{snap}/data/{_REVIEWS_FILE}",
                calendar_url=f"{data_root}{snap}/data/{_CALENDAR_FILE}",
            ))
    cities.sort(key=lambda c: (c.country, c.region, c.city, c.snapshot_date))
    return cities


def _from_regex_fallback(max_per_city: int) -> List[City]:
    """Old regex scrape kept as a fallback if the Gatsby static-query path moves.

    Captures only the latest snapshot per city - it's a degraded mode, but
    keeps the pipeline runnable while we re-discover the static-query hash.
    """
    r = _http_get(INSIDE_AIRBNB_INDEX_URL, timeout=60)
    listings: dict[tuple[str, str, str], tuple[str, str]] = {}
    reviews: dict[tuple[str, str, str, str], str] = {}
    calendars: dict[tuple[str, str, str, str], str] = {}

    for m in _DATA_URL_RE.finditer(r.text):
        country, region, city_slug, snapshot, kind = m.group(1, 2, 3, 4, 5)
        url = m.group(0)
        kind_l = kind.lower()
        if kind_l == "listings":
            key = (country, region, city_slug)
            prev = listings.get(key)
            if prev is None or snapshot > prev[0]:
                listings[key] = (snapshot, url)
        elif kind_l == "reviews":
            reviews[(country, region, city_slug, snapshot)] = url
        elif kind_l == "calendar":
            calendars[(country, region, city_slug, snapshot)] = url

    cities: List[City] = []
    for (country, region, city_slug), (snapshot, listings_url) in listings.items():
        cities.append(City(
            city=_humanize(city_slug),
            country=country,
            region=region,
            snapshot_date=snapshot,
            listings_url=listings_url,
            reviews_url=reviews.get((country, region, city_slug, snapshot), ""),
            calendar_url=calendars.get((country, region, city_slug, snapshot), ""),
        ))
    cities.sort(key=lambda c: (c.country, c.region, c.city))
    if max_per_city <= 1:
        return cities
    return cities


def _humanize(slug: str) -> str:
    return " ".join(part.capitalize() for part in slug.replace("_", "-").split("-"))


def fetch_csv_gz_bytes(url: str, timeout: int = 600) -> bytes:
    """Download a csv.gz from Inside Airbnb. Returns the *uncompressed* bytes."""
    r = _http_get(url, timeout=timeout)
    return gzip.decompress(r.content)


def parse_price(value: object) -> Optional[float]:
    """Parse Inside Airbnb price strings into floats. Returns None on garbage.

    Inside Airbnb normalizes prices to USD, but the actual string varies:
        '$1,250.00', '$95.00', '$1,250', '1250', '$0.00', '', 'null', NaN, ...
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (value != value):
            return None
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in ("nan", "null", "none"):
        return None
    s = s.lstrip("$\u20ac\u00a3\u00a5").strip()
    s = s.replace(",", "")
    s = s.replace(" ", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def head_check(url: str, timeout: int = 30) -> tuple[bool, int]:
    """HEAD-check a URL, return (ok, content_length)."""
    try:
        r = requests.head(url, timeout=timeout, headers={"User-Agent": _USER_AGENT}, allow_redirects=True)
        return (r.status_code == 200, int(r.headers.get("Content-Length", 0)))
    except Exception:
        return (False, 0)


def sample_image_urls_ok(image_urls: Iterable[str], timeout: int = 10) -> tuple[int, int]:
    """Return (n_ok, n_total) for a small sample of listing image URLs."""
    n_ok = 0
    urls = list(image_urls)
    for u in urls:
        if not u:
            continue
        try:
            r = requests.head(u, timeout=timeout, allow_redirects=True, headers={"User-Agent": _USER_AGENT})
            if r.status_code == 200:
                n_ok += 1
        except Exception:
            pass
    return (n_ok, len(urls))
