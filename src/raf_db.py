#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Ivan Ferre'
__email__ = 'ivan.ferre@upm.es'

import numpy as np
import pandas as pd
from pathlib import Path


def _to_float_or_none(x):
    if pd.isna(x):
        return None
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def parse_rafdb(csv_path):
    df = pd.read_csv(csv_path)

    samples = []
    for _, r in df.iterrows():
        expr_id = int(r["expression"])
        gender_id = int(r["gender"]) if pd.notna(r["gender"]) else None
        if gender_id == 2:  # To skip unsure gender values, as they do in the paper
            continue

        race_id = int(r["race"]) if pd.notna(r["race"]) else None
        age_id = int(r["age"]) if pd.notna(r["age"]) else None

        rec = {
            "path": str(r["path"]),
            "expression": expr_id,
            "gender": gender_id,
            "race": race_id,
            "age": age_id,
        }

        for k in ["yaw", "pitch", "roll", "bbox_h", "illumination"]:
            if k in df.columns:
                v = _to_float_or_none(r[k])
                if v is not None:
                    rec[k] = v

        samples.append(rec)

    return samples
