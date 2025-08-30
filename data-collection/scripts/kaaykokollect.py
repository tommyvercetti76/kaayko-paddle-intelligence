#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAAYKO – Enhanced Seasonal Weather Collection (Monthly Batching, Adaptive Rate)
-------------------------------------------------------------------------------
• Uses WeatherAPI History with dt…end_dt (month-at-a-time, ≤31 days)
• Global token-bucket + adaptive RPM (backs off on 429; ramps on sustained success)
• aqi=no&alerts=no (smaller payloads)
• Per lake-per-month CSV files to avoid locks & speed I/O
• Thread pool across lakes (default 12), month chunks per lake are processed sequentially
• Rich, color-coded terminal output + structured logs
• Clean resume: skips months that are already written

ENV OVERRIDES:
  KAAYKO_API_KEY, KAAYKO_RPM, KAAYKO_THREADS, KAAYKO_START, KAAYKO_END,
  KAAYKO_LAKES_CSV, KAAYKO_OUTPUT_DIR

Author: Kaayko
"""

import os
import sys
import csv
import math
import json
import time
import signal
import logging
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ────────────────────────────────────────────────────────────────────────────────
#  Import seasonal functions (your existing file)
# ────────────────────────────────────────────────────────────────────────────────
sys.path.append('/path/to/your/scripts')
from massive_scale_analysis import get_regional_season, classify_geographic_region  # noqa: E402

# ────────────────────────────────────────────────────────────────────────────────
#  ANSI Colors / Glyphs
# ────────────────────────────────────────────────────────────────────────────────
class C:
    R = "\033[0m"
    B = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[38;5;196m"
    GRN = "\033[38;5;46m"
    YEL = "\033[38;5;220m"
    CYA = "\033[38;5;45m"
    MAG = "\033[38;5;207m"
    BLU = "\033[38;5;69m"
    GRY = "\033[38;5;246m"

BAR = f"{C.GRY}{'─'*78}{C.R}"
OK  = f"{C.GRN}✔{C.R}"
KO  = f"{C.RED}✖{C.R}"
WA  = f"{C.YEL}⚠{C.R}"
IN  = f"{C.CYA}➜{C.R}"
SP  = f"{C.MAG}◆{C.R}"

# ────────────────────────────────────────────────────────────────────────────────
#  Config (with env overrides)
# ────────────────────────────────────────────────────────────────────────────────
API_KEY         = os.getenv("KAAYKO_API_KEY", "YOUR_WEATHERAPI_KEY_HERE")
LAKES_CSV       = os.getenv("KAAYKO_LAKES_CSV", "/path/to/your/lakes.csv")
OUTPUT_DIR      = os.getenv("KAAYKO_OUTPUT_DIR", "data_lake_monthly")
START_DATE      = os.getenv("KAAYKO_START", "2019-01-01")
END_DATE        = os.getenv("KAAYKO_END",   "2025-01-01")

MAX_THREADS     = int(os.getenv("KAAYKO_THREADS", "12"))   # guardrail: 10–15
TARGET_RPM      = int(os.getenv("KAAYKO_RPM", "40"))        # starting RPM target (adaptive)
MAX_RETRIES     = 4                                         # robust retries
TIMEOUT_SEC     = 20
HEADERS         = {"User-Agent": "Kaayko-SeasonalCollector/2.2"}

# Paddle: keep 24/7 (you can restrict later if desired)
PADDLE_HOURS    = list(range(0, 24))

# Logging
if not os.path.exists("logs"):
    os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=os.path.join("logs", "weather_seasonal_monthly.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Graceful shutdown
SHOULD_STOP = threading.Event()

def handle_signal(signum, frame):
    print(f"\n{WA} {C.B}Signal {signum} received — finishing in-flight tasks, then stopping...{C.R}")
    SHOULD_STOP.set()

for _sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_sig, handle_signal)

# ────────────────────────────────────────────────────────────────────────────────
#  Adaptive Token Bucket Limiter
# ────────────────────────────────────────────────────────────────────────────────
class AdaptiveLimiter:
    def __init__(self, start_rpm=40, min_rpm=10, max_rpm=120, burst=10):
        self.min_rpm = min_rpm
        self.max_rpm = max_rpm
        self._rpm = max(min(start_rpm, max_rpm), min_rpm)
        self.rate_per_sec = self._rpm / 60.0
        self.capacity = burst
        self.tokens = burst
        self.last = time.time()
        self.lock = threading.Lock()
        self.success_counter = 0

    @property
    def rpm(self):
        with self.lock:
            return int(self._rpm)

    def _refill(self):
        now = time.time()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate_per_sec)
        self.last = now

    def take(self, n=1):
        while not SHOULD_STOP.is_set():
            with self.lock:
                self._refill()
                if self.tokens >= n:
                    self.tokens -= n
                    return
            time.sleep(0.03)

    def on_success(self):
        # Gentle ramp up every 300 successes if we have headroom
        with self.lock:
            self.success_counter += 1
            if self.success_counter % 300 == 0 and self._rpm < self.max_rpm:
                self._rpm = min(self.max_rpm, int(self._rpm * 1.10))
                self.rate_per_sec = self._rpm / 60.0

    def on_throttle(self):
        # Immediate 20% step-down (min floor), reset success counter
        with self.lock:
            self._rpm = max(self.min_rpm, int(self._rpm * 0.80))
            self.rate_per_sec = self._rpm / 60.0
            self.success_counter = 0

    def snapshot(self):
        with self.lock:
            return {"rpm": int(self._rpm), "tokens": round(self.tokens, 2)}

limiter = AdaptiveLimiter(start_rpm=TARGET_RPM, min_rpm=10, max_rpm=120, burst=12)

# ────────────────────────────────────────────────────────────────────────────────
#  Utility
# ────────────────────────────────────────────────────────────────────────────────
def slugify(name: str) -> str:
    s = "".join(c if c.isalnum() or c in (" ", "_", "-", ".") else "_" for c in name).strip()
    s = "_".join(s.split())
    return s[:80] if len(s) > 80 else s

def month_ranges(start_date: str, end_date: str):
    """Yield tuples (start_iso, end_iso, year, month) aligned to calendar months."""
    s = datetime.strptime(start_date, "%Y-%m-%d")
    e = datetime.strptime(end_date,   "%Y-%m-%d")
    cur = datetime(s.year, s.month, 1)
    while cur <= e:
        # first day of current month to last day of current month
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        last_day = (nxt - timedelta(days=1)).day
        start_iso = cur.strftime("%Y-%m-01")
        end_iso = cur.strftime(f"%Y-%m-{last_day:02d}")
        yield start_iso, end_iso, cur.year, cur.month
        cur = nxt

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def days_in_range(start_iso: str, end_iso: str) -> int:
    s = datetime.strptime(start_iso, "%Y-%m-%d")
    e = datetime.strptime(end_iso,   "%Y-%m-%d")
    return (e - s).days + 1

# ────────────────────────────────────────────────────────────────────────────────
#  Estimators & Scoring (your logic, kept)
# ────────────────────────────────────────────────────────────────────────────────
def estimate_water_temperature(air_temp_c, latitude, month):
    water_temp = air_temp_c - 3.0
    seasonal_info = get_regional_season(latitude, 0, month, 15)
    if seasonal_info['climate_zone'] == 'tropical':
        water_temp = air_temp_c - 2.0
    elif seasonal_info['season'] == 'monsoon':
        water_temp = air_temp_c - 4.0
    elif seasonal_info['climate_zone'] == 'polar':
        water_temp = air_temp_c - 6.0
    seasonal_factor = math.sin(math.radians((month - 4) * 30)) * 2.0
    if latitude < 0:
        seasonal_factor *= -1
    water_temp += seasonal_factor
    latitude_factor = (abs(latitude) - 20) * 0.1
    water_temp -= max(0, latitude_factor)
    return round(water_temp, 1)

def calculate_paddle_score(row, seasonal_info):
    score = 3.0
    temp = row.get('temp_c')
    if 20 <= temp <= 25: score += 1.0
    elif 18 <= temp < 20 or 25 < temp <= 28: score += 0.5
    elif 15 <= temp < 18 or 28 < temp <= 32: score += 0.0
    elif 12 <= temp < 15 or 32 < temp <= 35: score -= 0.5
    elif 8 <= temp < 12 or 35 < temp <= 38: score -= 1.0
    else: score -= 2.0

    wind_kph = row.get('wind_kph')
    if 5 <= wind_kph <= 15: score += 0.5
    elif wind_kph < 5 or (15 < wind_kph <= 20): score += 0.0
    elif 20 < wind_kph <= 30: score -= 0.5
    elif 30 < wind_kph <= 40: score -= 1.0
    else: score -= 2.0

    precip_mm = row.get('precip_mm')
    if precip_mm == 0: score += 0.5
    elif precip_mm <= 1: score += 0.0
    elif 1 < precip_mm <= 3: score -= 0.5
    elif 3 < precip_mm <= 8: score -= 1.0
    else: score -= 1.5

    season = seasonal_info['season']
    climate_zone = seasonal_info['climate_zone']
    if season == 'monsoon' and precip_mm > 5: score -= 0.5
    if season in ['summer','dry_season'] and climate_zone in ['temperate','subtropical']: score += 0.2
    elif season == 'winter' and climate_zone == 'tropical': score += 0.3
    if climate_zone == 'equatorial': score += 0.1
    elif climate_zone == 'polar' and temp > 10: score += 0.3
    uv = row.get('uv')
    if uv > 8: score -= 0.5
    cloud = row.get('cloud')
    if cloud < 20: score += 0.2
    elif cloud > 80: score -= 0.2
    return max(1.0, min(5.0, round(score * 2) / 2))

def classify_skill_level(paddle_score):
    if paddle_score >= 4.5: return "Beginner"
    elif paddle_score >= 3.5: return "Intermediate"
    elif paddle_score >= 2.5: return "Advanced"
    else: return "Expert Only"

def classify_skill_level_enhanced(paddle_score, seasonal_info):
    base = classify_skill_level(paddle_score)
    if seasonal_info['season'] == 'monsoon':
        lvls = ["Beginner","Intermediate","Advanced","Expert Only"]
        return lvls[min(lvls.index(base)+1, 3)]
    if seasonal_info['climate_zone'] == 'polar':
        return "Expert Only" if paddle_score < 3.0 else "Advanced"
    return base

# ────────────────────────────────────────────────────────────────────────────────
#  Weather API call (month range). Adds aqi=no&alerts=no. Adaptive rate + retries
# ────────────────────────────────────────────────────────────────────────────────
def fetch_history_range(lat: float, lon: float, dt_iso: str, end_dt_iso: str):
    params = {
        "key": API_KEY,
        "q": f"{lat},{lon}",
        "dt": dt_iso,
        "end_dt": end_dt_iso,
        "aqi": "no",
        "alerts": "no",
    }
    url = "https://api.weatherapi.com/v1/history.json"

    for attempt in range(1, MAX_RETRIES + 1):
        if SHOULD_STOP.is_set():
            return None
        limiter.take()
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT_SEC)
            if r.status_code == 429:
                logging.warning(f"429 throttle for ({lat},{lon}) {dt_iso}→{end_dt_iso} attempt {attempt}")
                limiter.on_throttle()
                time.sleep(min(8, 2 ** attempt))
                continue
            if r.status_code >= 500:
                logging.warning(f"5xx error {r.status_code} for ({lat},{lon}) range {dt_iso}→{end_dt_iso}")
                time.sleep(min(8, 2 ** attempt))
                continue
            r.raise_for_status()
            limiter.on_success()
            return r.json()
        except requests.RequestException as e:
            logging.warning(f"Req fail ({lat},{lon}) {dt_iso}→{end_dt_iso} attempt {attempt}: {e}")
            time.sleep(min(8, 2 ** attempt))
    return None

# ────────────────────────────────────────────────────────────────────────────────
#  Processing
# ────────────────────────────────────────────────────────────────────────────────
FIELDNAMES = [
    "lake","datetime","temp_c","wind_kph","wind_dir","humidity","cloud","uv",
    "precip_mm","condition","pressure_mb","dew_point_c","feelslike_c",
    "gust_kph","is_day","will_it_rain","will_it_snow","vis_km",
    "estimated_water_temp_c","estimated_wave_height_m",
    "paddle_score","skill_level",
    "season","season_intensity","hemisphere","climate_zone","region",
    "regional_pattern","latitude","longitude","month","day_of_year",
    "lake_region","lake_type","base_lake_name"
]

def write_lake_month_file(lake_slug: str, year: int, month: int, rows: list):
    if not rows:
        return 0
    out_dir = os.path.join(OUTPUT_DIR, lake_slug)
    ensure_dir(out_dir)
    file_path = os.path.join(out_dir, f"{year:04d}-{month:02d}.csv")
    # If exists, skip (resume behavior)
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return 0
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    return len(rows)

def lake_month_exists(lake_slug: str, year: int, month: int) -> bool:
    file_path = os.path.join(OUTPUT_DIR, lake_slug, f"{year:04d}-{month:02d}.csv")
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0

def process_history_json(lake, lat, lon, data, region, lake_type, base_lake):
    """Return list of hourly rows (filtered by PADDLE_HOURS)."""
    rows = []
    if not data or "forecast" not in data or "forecastday" not in data["forecast"]:
        return rows
    for day_block in data["forecast"]["forecastday"]:
        for hour_entry in day_block.get("hour", []):
            try:
                hour_str = hour_entry["time"]
                hour = int(hour_str.split(" ")[1].split(":")[0])
                if hour not in PADDLE_HOURS:
                    continue
                dt_obj = datetime.strptime(hour_str, "%Y-%m-%d %H:%M")
                month = dt_obj.month
                seas = get_regional_season(lat, lon, dt_obj.month, dt_obj.day)

                temp_c = hour_entry["temp_c"]
                wind_kph = hour_entry["wind_kph"]
                precip_mm = hour_entry["precip_mm"]
                cloud = hour_entry["cloud"]
                uv = hour_entry["uv"]
                humidity = hour_entry["humidity"]
                pressure_mb = hour_entry["pressure_mb"]

                est_water = estimate_water_temperature(temp_c, lat, dt_obj.month)
                est_wave  = min(0.016 * (wind_kph ** 1.2) / 10, 3.0)

                weather_row = {"temp_c": temp_c, "wind_kph": wind_kph,
                               "precip_mm": precip_mm, "cloud": cloud, "uv": uv}
                paddle_score = calculate_paddle_score(weather_row, seas)
                skill_level = classify_skill_level_enhanced(paddle_score, seas)

                rows.append({
                    "lake": lake,
                    "datetime": hour_str,
                    "temp_c": temp_c,
                    "wind_kph": wind_kph,
                    "wind_dir": hour_entry["wind_dir"],
                    "humidity": humidity,
                    "cloud": cloud,
                    "uv": uv,
                    "precip_mm": precip_mm,
                    "condition": hour_entry["condition"]["text"],
                    "pressure_mb": pressure_mb,
                    "dew_point_c": hour_entry.get("dewpoint_c"),
                    "feelslike_c": hour_entry["feelslike_c"],
                    "gust_kph": hour_entry["gust_kph"],
                    "is_day": hour_entry["is_day"],
                    "will_it_rain": hour_entry["will_it_rain"],
                    "will_it_snow": hour_entry["will_it_snow"],
                    "vis_km": hour_entry["vis_km"],
                    "estimated_water_temp_c": est_water,
                    "estimated_wave_height_m": est_wave,
                    "paddle_score": paddle_score,
                    "skill_level": skill_level,
                    "season": seas['season'],
                    "season_intensity": seas['season_intensity'],
                    "hemisphere": seas['hemisphere'],
                    "climate_zone": seas['climate_zone'],
                    "region": seas['region'],
                    "regional_pattern": seas['regional_pattern'],
                    "latitude": lat,
                    "longitude": lon,
                    "month": month,
                    "day_of_year": dt_obj.timetuple().tm_yday,
                    "lake_region": region or "Unknown",
                    "lake_type": lake_type or "Lake",
                    "base_lake_name": base_lake or lake
                })
            except Exception as e:
                logging.warning(f"Hour parse error for {lake}: {e}")
    return rows

# ────────────────────────────────────────────────────────────────────────────────
#  Per-lake worker: pulls month-by-month; writes one CSV per month
# ────────────────────────────────────────────────────────────────────────────────
def collect_for_lake_monthly(name, lat, lon, region=None, lake_type=None, base_lake=None):
    if SHOULD_STOP.is_set():
        return 0, 0
    lake_slug = slugify(name)
    total_rows, months_done = 0, 0

    print(f"{SP} {C.B}{name}{C.R} {C.GRY}({lat:.4f},{lon:.4f}) • {region or 'Unknown'}, {lake_type or 'Lake'}{C.R}")

    for dt_iso, end_iso, yy, mm in month_ranges(START_DATE, END_DATE):
        if SHOULD_STOP.is_set():
            break

        # Resume: skip if we already have the month
        if lake_month_exists(lake_slug, yy, mm):
            print(f"   {OK} {C.GRY}skip {yy}-{mm:02d} (exists){C.R}")
            months_done += 1
            continue

        # Fetch month range
        print(f"   {IN} {C.BLU}GET{C.R} {dt_iso} → {end_iso}   "
              f"{C.GRY}[rpm:{limiter.rpm:>3}] {json.dumps(limiter.snapshot())}{C.R}")

        data = fetch_history_range(lat, lon, dt_iso, end_iso)
        if data is None:
            print(f"   {KO} {C.RED}failed {yy}-{mm:02d}{C.R}")
            logging.error(f"Failed month {yy}-{mm:02d} for {name}")
            # Do not write partial; continue to next month (will retry on next run)
            continue

        # Process and write
        rows = process_history_json(name, lat, lon, data, region, lake_type, base_lake)
        wrote = write_lake_month_file(lake_slug, yy, mm, rows)
        total_rows += wrote
        months_done += 1

        if wrote > 0:
            print(f"   {OK} {C.GRN}wrote {wrote:5d} rows → {lake_slug}/{yy}-{mm:02d}.csv{C.R}")
        else:
            # Either zero rows or file skipped
            if rows and wrote == 0:
                print(f"   {WA} {C.YEL}file existed, skipped write{C.R}")
            else:
                print(f"   {WA} {C.YEL}no rows (API returned empty?) {yy}-{mm:02d}{C.R}")

    return total_rows, months_done

# ────────────────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────────────────
def main():
    # Header
    print("\n" + BAR)
    print(f"{C.B}KAAYKO SEASONAL COLLECTOR — MONTHLY RANGE MODE{C.R}  {C.GRY}(aqi=no, alerts=no){C.R}")
    print(BAR)
    print(f"{C.GRY}• Lakes CSV:{C.R} {LAKES_CSV}")
    print(f"{C.GRY}• Output   :{C.R} {OUTPUT_DIR}")
    print(f"{C.GRY}• Period   :{C.R} {START_DATE} → {END_DATE}")
    print(f"{C.GRY}• Threads  :{C.R} {MAX_THREADS}   {C.GRY}• Start RPM:{C.R} {TARGET_RPM}")
    print(BAR)

    # Load lakes (expects: name, lat, lng, region, type, base_lake)
    lakes = []
    with open(LAKES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                if not row["name"].strip(): 
                    continue
                lakes.append({
                    "name": row["name"].strip(),
                    "lat": float(row["lat"]),
                    "lon": float(row["lng"]),
                    "region": row.get("region", "Unknown").strip(),
                    "type": row.get("type", "Lake").strip(),
                    "base_lake": row.get("base_lake", row["name"]).strip()
                })
            except Exception as e:
                logging.warning(f"Bad row in lakes CSV: {e}")

    total_lakes = len(lakes)
    print(f"{C.B}{OK} Loaded {total_lakes:,} lakes{C.R}\n" + BAR)

    ensure_dir(OUTPUT_DIR)

    lake_counter = 0
    grand_rows = 0
    t0 = time.time()

    # Process lakes in parallel (each lake does months sequentially)
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as ex:
        futures = []
        for L in lakes:
            if SHOULD_STOP.is_set(): break
            futures.append(ex.submit(
                collect_for_lake_monthly,
                L["name"], L["lat"], L["lon"], L["region"], L["type"], L["base_lake"]
            ))

        for fut in as_completed(futures):
            try:
                rows, months = fut.result()
                grand_rows += rows
                lake_counter += 1
                elapsed = time.time() - t0
                rate = grand_rows / max(1, elapsed)
                print(f"{SP} {C.B}{lake_counter}/{total_lakes}{C.R} lakes done  "
                      f"| rows: {C.GRN}{grand_rows:,}{C.R}  "
                      f"| avg: {C.CYA}{rate:.1f} rows/s{C.R}  "
                      f"| limiter rpm={C.B}{limiter.rpm}{C.R}")
            except Exception as e:
                logging.error(f"Lake task failed: {e}")
                print(f"{KO} {C.RED}Lake task failed:{C.R} {e}")

    elapsed = time.time() - t0
    print(BAR)
    print(f"{C.B}{OK} COMPLETE{C.R}  rows={C.GRN}{grand_rows:,}{C.R}  time={C.CYA}{elapsed/3600:.2f}h{C.R}  "
          f"avg={C.BLU}{(grand_rows/max(1, elapsed)):.1f}/s{C.R}")
    print(BAR + "\n")

if __name__ == "__main__":
    # Quick guardrails banner
    print(f"{C.B}{C.MAG}Quick Guardrails{C.R}: "
          f"{C.GRY}threads↓ to ~10–15, global rate limiter ON, aqi=no&alerts=no, per-lake-month files{C.R}")
    if not API_KEY or len(API_KEY) < 10:
        print(f"{KO} {C.RED}Missing or invalid API key. Set KAAYKO_API_KEY or edit API_KEY in this script.{C.R}")
        sys.exit(1)
    main()