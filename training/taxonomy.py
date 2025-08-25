#!/usr/bin/env python3
"""
Simple hierarchy/tagging for specialists.
- geo_group: high-level region or climate group
- geo_subgroup: more specific (e.g., region+climate)
You can extend this to state-level when you add a 'state' column.
"""
import pandas as pd

def add_taxonomy_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize 'region' and 'climate_zone'
    region = df.get("region")
    climate = df.get("climate_zone")

    # geo_group prefers region if present, else climate_zone
    if region is not None:
        df["geo_group"] = region.fillna("Unknown").astype(str)
    elif climate is not None:
        df["geo_group"] = climate.fillna("Unknown").astype(str)
    else:
        df["geo_group"] = "Unknown"

    # geo_subgroup combines both when possible
    if region is not None and climate is not None:
        df["geo_subgroup"] = (region.fillna("Unknown").astype(str) + "__" +
                              climate.fillna("Unknown").astype(str))
    elif region is not None:
        df["geo_subgroup"] = region.fillna("Unknown").astype(str)
    elif climate is not None:
        df["geo_subgroup"] = climate.fillna("Unknown").astype(str)
    else:
        df["geo_subgroup"] = "Unknown"

    return df
