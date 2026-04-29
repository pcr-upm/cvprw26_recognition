#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Ivan Ferre'
__email__ = 'ivan.ferre@upm.es'

import numpy as np
import pandas as pd


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


def parse_affectnet(split):
    df = pd.read_csv(split)

    gender_cols = [c for c in df.columns if c.startswith("gender_")]
    race_cols = [c for c in df.columns if c.startswith("race_")]

    df[gender_cols + race_cols + ["yaw", "pitch", "roll"]] = df[
        gender_cols + race_cols + ["yaw", "pitch", "roll"]
    ].apply(pd.to_numeric, errors="coerce")

    df = df.dropna(subset=["image_path", "human_label", "yaw", "pitch", "roll"] + gender_cols + race_cols).reset_index(
        drop=True
    )

    samples = []

    for _, r in df.iterrows():
        gender_probs = r[gender_cols].to_numpy(dtype="float32")
        race_probs = r[race_cols].to_numpy(dtype="float32")

        gender_id = int(gender_probs.argmax())
        race_id = int(race_probs.argmax())
        expr_id = int(r["human_label"])

        yaw = float(r["yaw"])
        pitch = float(r["pitch"])
        roll = float(r["roll"])
        illumination = float(r["illumination"])

        samples.append(
            {
                "path": str(r["image_path"]),
                "expression": expr_id,
                "gender": gender_id,
                "race": race_id,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "illumination": illumination,
            }
        )

    return samples
