from __future__ import annotations
import numpy as np
import pandas as pd

MODEL_FEATURES = ["age", "gender_m", "lead_days", "day_of_week", "scholarship", "hypertension", "diabetes", "alcoholism", "handicap", "sms_received", "prior_appointments", "prior_no_shows", "prior_no_show_rate", "neighbourhood_frequency"]
GROUP_COLUMNS = ["age_band", "scholarship"]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lead_days"] = (out["appointment_day"].dt.normalize() - out["scheduled_day"].dt.normalize()).dt.days.clip(lower=0)
    out["day_of_week"] = out["appointment_day"].dt.dayofweek.astype("int8")
    out["gender_m"] = (out["gender"].str.upper() == "M").astype("int8")
    out["age_band"] = pd.cut(out["age"], bins=[-np.inf, 17, 34, 49, 64, np.inf], labels=["0-17", "18-34", "35-49", "50-64", "65+"]).astype(str)
    # Compute neighbourhood frequency strictly before the appointment date.
    # Sorting by patient_id would make "history" depend on identifier order,
    # which is not a valid temporal feature.
    chronological = out.sort_values(
        ["appointment_day", "scheduled_day", "appointment_id"], kind="mergesort"
    )
    neighbourhood_day = (
        chronological.groupby(["neighbourhood", "appointment_day"], sort=False)
        .size()
        .rename("neighbourhood_day_count")
        .reset_index()
        .sort_values(["appointment_day", "neighbourhood"], kind="mergesort")
    )
    neighbourhood_day["neighbourhood_prior_count"] = (
        neighbourhood_day.groupby("neighbourhood", sort=False)["neighbourhood_day_count"]
        .cumsum()
        .sub(neighbourhood_day["neighbourhood_day_count"])
    )
    total_day = (
        chronological.groupby("appointment_day", sort=False)
        .size()
        .rename("total_day_count")
        .reset_index()
        .sort_values("appointment_day", kind="mergesort")
    )
    total_day["total_prior_count"] = total_day["total_day_count"].cumsum().sub(total_day["total_day_count"])
    neighbourhood_day = neighbourhood_day.merge(
        total_day[["appointment_day", "total_prior_count"]],
        on="appointment_day",
        how="left",
        validate="many_to_one",
    )
    neighbourhood_day["neighbourhood_frequency"] = (
        neighbourhood_day["neighbourhood_prior_count"]
        / neighbourhood_day["total_prior_count"].replace(0, np.nan)
    ).fillna(0.0)
    out = out.merge(
        neighbourhood_day[["neighbourhood", "appointment_day", "neighbourhood_frequency"]],
        on=["neighbourhood", "appointment_day"],
        how="left",
        validate="many_to_one",
    )

    # Patient history is aggregated at patient + appointment date first. This is
    # important: two appointments on the same day must not expose one another's
    # observed outcome, even if their appointment IDs sort differently.
    day = out.groupby(["patient_id", "appointment_day"], sort=False)["no_show"].agg(day_appointments="size", day_no_shows="sum").reset_index()
    day = day.sort_values(["patient_id", "appointment_day"], kind="mergesort")
    day_g = day.groupby("patient_id", sort=False)
    day["prior_appointments"] = day_g["day_appointments"].cumsum().sub(day["day_appointments"]).astype("int32")
    day["prior_no_shows"] = day_g["day_no_shows"].cumsum().sub(day["day_no_shows"]).astype("int32")
    day["prior_no_show_rate"] = (day["prior_no_shows"] / day["prior_appointments"].replace(0, np.nan)).fillna(0.0)
    out = out.drop(columns=["prior_appointments", "prior_no_shows", "prior_no_show_rate"], errors="ignore").merge(day[["patient_id", "appointment_day", "prior_appointments", "prior_no_shows", "prior_no_show_rate"]], on=["patient_id", "appointment_day"], how="left", validate="many_to_one")
    out["prior_appointments"] = out["prior_appointments"].astype("int32")
    out["prior_no_shows"] = out["prior_no_shows"].astype("int32")
    out["prior_no_show_rate"] = (out["prior_no_shows"] / out["prior_appointments"].replace(0, np.nan)).fillna(0.0).astype(float)
    return out.sort_values(["appointment_day", "scheduled_day", "appointment_id"], kind="mergesort").reset_index(drop=True)

