KAAYKO PIPELINE (Collector + Packaging + Training + Inference)

1) Guardrailed Collector (per lake-month CSVs)
   - file: rate_limited_collector.py
   - Use:
       python3 rate_limited_collector.py  # then call main_month(...) from code
   - Edit 00_config.py (keys/paths). The collector:
     • Enforces API rate limit with token-bucket (API_RPM_LIMIT)
     • Threads capped at MAX_THREADS (default 12)
     • Adds aqi=no&alerts=no
     • Writes one CSV per lake per month under data_out/by_lake_month/<lake>/<YYYY-MM>.csv

2) Package monthlies → parquet
   - file: 02_package_monthlies.py
   - Use:
       python3 02_package_monthlies.py
   - Produces partitioned parquet under data_out/packaged/lake=<name>/ym=<YYYY-MM>/part.parquet

3) Train (global + specialists)
   - file: 03_train_global_and_specialists.py
   - Use:
       python3 03_train_global_and_specialists.py
   - Trains:
     • Global regressor (paddle_score) and classifier (skill_level)
     • Specialist models per region (only if enough lakes)
   - Saves models/*pkl

4) Evaluate
   - file: 05_eval_models.py
   - Use:
       python3 05_eval_models.py
   - Reports GLOBAL model metrics with temporal holdout per lake.

5) Inference router
   - file: 04_inference_router.py
   - Given a CSV with forecast/current features, predicts paddle_score & skill.
   - Choose specialist for region when available; fallback to global.
