#!/usr/bin/env python3
"""
Guardrailed Weather Collector (per lake-month files)
• Token-bucket limiter (API_RPM_LIMIT)
• 10–15 threads (MAX_THREADS)
• aqi=no&alerts=no
• Writes one CSV per lake-month: data_out/by_lake_month/<lake>/<YYYY-MM>.csv
• Color-coded ASCII logs
"""
import os, time, json, math, random, csv, threading, argparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd
from pathlib import Path
from utils_ansi import ok, warn, err, info, title, sect
from 00_config import (
    RAW_MONTHLY_DIR, PACKAGED_DIR, WEATHER_API_KEY, WEATHER_BASE, API_RPM_LIMIT,
    MAX_THREADS, THREAD_JITTER_S, REQUEST_TIMEOUT, MAX_RETRIES, BACKOFF_S, ADD_PARAMS
)

# ---------- Token-Bucket Rate Limiter ----------
class RateLimiter:
    def __init__(self, rpm):
        self.capacity = rpm
        self.tokens = rpm
        self.fill_rate = rpm / 60.0
        self.timestamp = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.timestamp
            self.timestamp = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    def wait(self):
        while not self.acquire():
            time.sleep(0.05)

limiter = RateLimiter(API_RPM_LIMIT)

def fetch_history(lat, lon, date_str):
    params = f"?key={WEATHER_API_KEY}&q={lat},{lon}&dt={date_str}{ADD_PARAMS}"
    url = f"{WEATHER_BASE}{params}"
    for attempt in range(MAX_RETRIES):
        limiter.wait()
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                warn(f"429 {date_str}, backoff..")
                time.sleep(3.0 + random.random())
                continue
            r.raise_for_status()
            time.sleep(random.uniform(*THREAD_JITTER_S))
            return r.json()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                warn(f"retry {attempt+1}/{MAX_RETRIES} {lat},{lon} {date_str}: {e}")
                time.sleep(BACKOFF_S[min(attempt, len(BACKOFF_S)-1)])
            else:
                err(f"failed {lat},{lon} {date_str}: {e}")
                return None

def write_lake_month_csv(lake_name, month_key, rows, out_dir):
    out_dir = Path(out_dir) / "by_lake_month" / lake_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{month_key}.csv"
    if not rows:
        return None
    cols = list(rows[0].keys())
    write_header = not fp.exists()
    with open(fp, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    return str(fp)

def collect_for_lake_month(lake_row, start_date, end_date, out_dir):
    name = lake_row["name"]
    lat  = float(lake_row["lat"])
    lon  = float(lake_row["lng"])
    region = lake_row.get("region","Unknown")
    ltype  = lake_row.get("type","Lake")

    dt = start_date
    all_rows = []
    while dt <= end_date:
        js = fetch_history(lat, lon, dt.strftime("%Y-%m-%d"))
        if js is None:
            dt += timedelta(days=1)
            continue
        try:
            fd = js["forecast"]["forecastday"][0]
            for h in fd["hour"]:
                all_rows.append({
                    "lake": name,
                    "datetime": h["time"],
                    "temp_c": h["temp_c"],
                    "wind_kph": h["wind_kph"],
                    "wind_dir": h["wind_dir"],
                    "humidity": h["humidity"],
                    "cloud": h["cloud"],
                    "uv": h.get("uv", 0.0),
                    "precip_mm": h["precip_mm"],
                    "condition": h["condition"]["text"],
                    "pressure_mb": h["pressure_mb"],
                    "dew_point_c": h["dewpoint_c"],
                    "feelslike_c": h["feelslike_c"],
                    "gust_kph": h.get("gust_kph", 0.0),
                    "is_day": h["is_day"],
                    "will_it_rain": h.get("will_it_rain", 0),
                    "will_it_snow": h.get("will_it_snow", 0),
                    "vis_km": h.get("vis_km", 0.0),
                    "latitude": lat, "longitude": lon,
                    "region": region, "lake_type": ltype
                })
        except Exception as e:
            err(f"parse error {name} {dt}: {e}")
        dt += timedelta(days=1)

    month_key = start_date.strftime("%Y-%m")
    fp = write_lake_month_csv(name, month_key, all_rows, out_dir)
    if fp:
        ok(f"{name} {month_key} -> {fp} ({len(all_rows)} rows)")
    return fp

def main(lakes_csv, year, month, out_dir="./data_out"):
    title(f"KAAYKO Guardrailed Collector — {year}-{month:02d}")
    lakes = pd.read_csv(lakes_csv)
    lakes = lakes.dropna(subset=["name","lat","lng"])
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year+1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(year, month+1, 1) - timedelta(days=1)

    sect(f"Collecting {len(lakes)} lakes for {start_date.date()}..{end_date.date()}")
    futures = []
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as ex:
        for _, row in lakes.iterrows():
            futures.append(ex.submit(collect_for_lake_month, row.to_dict(), start_date, end_date, out_dir))
        done = 0
        for fut in as_completed(futures):
            _ = fut.result()
            done += 1
            if done % 20 == 0:
                info(f"progress {done}/{len(futures)}")

    ok("Month complete.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lakes_csv", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--out_dir", default="./data_out")
    args = ap.parse_args()
    main(args.lakes_csv, args.year, args.month, args.out_dir)
