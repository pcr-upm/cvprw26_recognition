#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Ivan Ferre'
__email__ = 'ivan.ferre@upm.es'

import numpy as np
import pandas as pd


# Shared constants
GENDER_STR_TO_ID = {"Female": 0, "Male": 1}
RACE_STR_TO_ID = {"Asian": 0, "Indian": 1, "Black": 2, "White": 3}


def _safe_float(x):
    """Convert to float, return None if invalid or non-finite."""
    if pd.isna(x):
        return None
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _safe_int(x):
    """Convert to int, return None if invalid."""
    if pd.isna(x):
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def _build_sample(path, expression, gender=None, race=None, yaw=None, illumination=None):
    """Build a sample record with standard shared fields."""
    sample = {
        "path": str(path),
        "expression": expression,
    }
    if gender is not None:
        sample["gender"] = gender
    if race is not None:
        sample["race"] = race
    if yaw is not None:
        sample["yaw"] = yaw
    if illumination is not None:
        sample["illumination"] = illumination
    return sample


def parse_rafdb(split):
    """Parse RAF-DB dataset."""
    df = pd.read_csv(split)
    samples = []
    for _, row in df.iterrows():
        gender_id = _safe_int(row["gender"])
        if gender_id == 2:  # Skip unsure gender values
            continue
        yaw = _safe_float(row.get("yaw"))
        illumination = _safe_float(row.get("illumination"))
        sample = _build_sample(path=row["path"], expression=int(row["expression"]), gender=gender_id, race=_safe_int(row["race"]), yaw=yaw, illumination=illumination)
        samples.append(sample)
    return samples


def parse_affectnet(split):
    """Parse AffectNet dataset."""
    df = pd.read_csv(split)
    # Extract probability columns
    gender_cols = [c for c in df.columns if c.startswith("gender_")]
    race_cols = [c for c in df.columns if c.startswith("race_")]
    # Convert numeric columns
    df[gender_cols + race_cols + ["yaw"]] = df[gender_cols + race_cols + ["yaw"]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["image_path", "human_label", "yaw"] + gender_cols + race_cols).reset_index(drop=True)
    samples = []
    for _, row in df.iterrows():
        gender_id = int(row[gender_cols].to_numpy(dtype="float32").argmax())
        race_id = int(row[race_cols].to_numpy(dtype="float32").argmax())
        sample = _build_sample(path=row["image_path"], expression=int(row["human_label"]), gender=gender_id, race=race_id, yaw=float(row["yaw"]), illumination=float(row["illumination"]))
        samples.append(sample)
    return samples


def parse_affwild2(split):
    """Parse AffWild2 dataset."""
    df = pd.read_csv(split)
    df["expr"] = pd.to_numeric(df["expr"], errors="coerce")
    df = df.dropna(subset=["image_path", "expr"]).reset_index(drop=True)
    samples = []
    for _, row in df.iterrows():
        expr_id = _safe_int(row["expr"])
        if expr_id is None:
            continue
        gender_id = GENDER_STR_TO_ID.get(row.get("gender"))
        race_id = RACE_STR_TO_ID.get(row.get("ethnicity"))
        yaw = _safe_float(row.get("yaw"))
        illumination = _safe_float(row.get("illumination"))
        sample = _build_sample(path=row["image_path"], expression=expr_id, gender=gender_id, race=race_id, yaw=yaw, illumination=illumination)
        samples.append(sample)
    return samples


def parse_multipie(split):
    """Parse Multi-PIE dataset."""
    df = pd.read_csv(split)
    samples = []
    for _, row in df.iterrows():
        # Determine gender (prefer gender_norm if available)
        gender_str = str(row.get("gender_norm", row.get("gender"))).strip().lower()
        gender_id = 1 if gender_str == "male" else 0
        # Determine yaw from camera_id
        camera_id = str(row["camera_id"])
        yaw = 0 if camera_id in ["14_0", "05_1", "05_0"] else 1
        # Extract illumination from filename
        illumination = int(str(row["filename"]).split("_")[-1].split(".")[0])
        sample = _build_sample(path=row["abs_path"], expression=int(row["expression_id"]), gender=gender_id, race=RACE_STR_TO_ID.get(row.get("ethnicity")), yaw=yaw, illumination=illumination)
        samples.append(sample)
    return samples
