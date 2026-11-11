from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

RENAME = {
    "PatientId": "patient_id", "AppointmentID": "appointment_id", "Gender": "gender",
    "ScheduledDay": "scheduled_day", "AppointmentDay": "appointment_day", "Age": "age",
    "Neighbourhood": "neighbourhood", "Scholarship": "scholarship",
    "Hipertension": "hypertension", "Diabetes": "diabetes", "Alcoholism": "alcoholism",
    "Handcap": "handicap", "SMS_received": "sms_received", "No-show": "no_show_raw",
}

@dataclass(frozen=True)
class DataQualitySummary:
    input_rows: int
    output_rows: int
    dropped_negative_age: int
    dropped_bad_dates: int
    duplicate_rows: int
    min_appointment_date: str
    max_appointment_date: str

def load_and_clean(path: str | Path) -> tuple[pd.DataFrame, DataQualitySummary]:
    raw = pd.read_csv(path)
    missing = set(RENAME) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing expected source columns: {sorted(missing)}")
    df = raw.rename(columns=RENAME).copy()
    input_rows = len(df)
    duplicate_rows = int(df.duplicated(subset=["appointment_id"]).sum())
    df["scheduled_day"] = pd.to_datetime(df["scheduled_day"], utc=True, errors="coerce")
    df["appointment_day"] = pd.to_datetime(df["appointment_day"], utc=True, errors="coerce")
    bad_dates = df["scheduled_day"].isna() | df["appointment_day"].isna() | (df["appointment_day"] < df["scheduled_day"].dt.normalize())
    dropped_bad_dates = int(bad_dates.sum())
    negative_age = df["age"] < 0
    dropped_negative_age = int(negative_age.sum())
    df = df.loc[~bad_dates & ~negative_age].copy()
    df["no_show"] = df["no_show_raw"].map({"Yes": 1, "No": 0}).astype("int8")
    if df["no_show"].isna().any():
        raise ValueError("Unexpected values in No-show label")
    for c in ["scholarship", "hypertension", "diabetes", "alcoholism", "handicap", "sms_received"]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype("int8")
    df["patient_id"] = df["patient_id"].astype("int64")
    df["appointment_id"] = df["appointment_id"].astype("int64")
    df = df.sort_values(["appointment_day", "scheduled_day", "appointment_id"], kind="mergesort").reset_index(drop=True)
    summary = DataQualitySummary(input_rows, len(df), dropped_negative_age, dropped_bad_dates, duplicate_rows, str(df["appointment_day"].min()), str(df["appointment_day"].max()))
    return df, summary

