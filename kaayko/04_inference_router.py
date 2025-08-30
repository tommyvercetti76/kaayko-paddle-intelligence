#!/usr/bin/env python3
"""
Inference router:
- prefers most specific specialist available (geo_subgroup -> geo_group -> region -> climate_zone)
- caches chosen models per key
- returns predicted paddle_score & skill_level
"""
import pickle, os, pandas as pd, numpy as np
from pathlib import Path
from functools import lru_cache
from utils_ansi import title, info, ok, warn, err
from 00_config import MODELS_DIR

def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@lru_cache(maxsize=512)
def _try_load(tag, kind):
    path = Path(MODELS_DIR)/f"{tag}_{'reg' if kind=='reg' else 'clf'}.pkl"
    return _load(path) if path.exists() else None

def _candidate_tags(row):
    # Build candidates in most-specific → least order
    cands = []
    if "geo_subgroup" in row and pd.notna(row["geo_subgroup"]):
        cands.append(f"spec__geo_subgroup__{str(row['geo_subgroup']).replace(' ','_')}")
    if "geo_group" in row and pd.notna(row["geo_group"]):
        cands.append(f"spec__geo_group__{str(row['geo_group']).replace(' ','_')}")
    if "region" in row and pd.notna(row["region"]):
        cands.append(f"spec__region__{str(row['region']).replace(' ','_')}")
    if "climate_zone" in row and pd.notna(row["climate_zone"]):
        cands.append(f"spec__climate_zone__{str(row['climate_zone']).replace(' ','_')}")
    cands.append("global")
    return cands

def _choose_models_for_row(row):
    for tag in _candidate_tags(row):
        reg = _try_load(tag, "reg")
        clf = _try_load(tag, "clf")
        if reg and clf:
            return tag, reg, clf
    # Fallback to global if only one exists
    reg = _try_load("global", "reg")
    clf = _try_load("global", "clf")
    return "global", reg, clf

def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    preds_score, preds_skill, used_tags = [], [], []
    for idx, row in df.iterrows():
        tag, reg, clf = _choose_models_for_row(row)
        Xr = row.reindex(reg["features"]).to_frame().T
        Xc = row.reindex(clf["features"]).to_frame().T
        score = float(reg["pipe"].predict(Xr)[0])
        skill = str(clf["pipe"].predict(Xc)[0])
        preds_score.append(score)
        preds_skill.append(skill)
        used_tags.append(tag)
    out["pred_paddle_score"] = preds_score
    out["pred_skill_level"]  = preds_skill
    out["model_tag"]         = used_tags
    return out

def predict_csv(csv_fp):
    df = pd.read_csv(csv_fp)
    return predict_df(df)

if __name__ == "__main__":
    pass
