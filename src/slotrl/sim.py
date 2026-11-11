from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Request:
    appointment_id: int
    p_noshow: float
    shows: int
    age_band: str
    scholarship: int

class RequestPool:
    def __init__(self, frame: pd.DataFrame, seed: int = 0, probability_column: str = "p_noshow_cal"):
        required = {"appointment_id", probability_column, "no_show", "age_band", "scholarship"}
        missing = required - set(frame.columns)
        if missing: raise ValueError(f"Request pool missing columns: {sorted(missing)}")
        self.frame = frame.reset_index(drop=True).copy()
        self.probability_column = probability_column
        self.rng = np.random.default_rng(seed)

    def sample_day(self, n_requests: int, seed: int | None = None, base_rate: float | None = None) -> list[Request]:
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        frame = self.frame
        if base_rate is not None:
            y = frame["no_show"].to_numpy()
            p = np.where(y == 1, base_rate / max(y.mean(), 1e-9), (1 - base_rate) / max(1 - y.mean(), 1e-9))
            p = p / p.sum()
            idx = rng.choice(len(frame), size=n_requests, replace=True, p=p)
        else:
            idx = rng.choice(len(frame), size=n_requests, replace=True)
        rows = frame.iloc[idx]
        return [Request(int(r.appointment_id), float(getattr(r, self.probability_column)), int(1 - r.no_show), str(r.age_band), int(r.scholarship)) for r in rows.itertuples()]

def split_rl_pools(test_frame: pd.DataFrame, fraction: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split the model test slice for tuning versus final evaluation."""
    if not 0 < fraction < 1: raise ValueError("fraction must be strictly between 0 and 1")
    dates=np.sort(pd.to_datetime(test_frame["appointment_day"],utc=True).dt.date.unique())
    cut=max(1,min(len(dates)-1,int(len(dates)*fraction)))
    train=test_frame[pd.to_datetime(test_frame["appointment_day"],utc=True).dt.date.isin(dates[:cut])].copy()
    evaluation=test_frame[pd.to_datetime(test_frame["appointment_day"],utc=True).dt.date.isin(dates[cut:])].copy()
    if train.empty or evaluation.empty: raise ValueError("Chronological pool split produced an empty pool")
    return train.reset_index(drop=True),evaluation.reset_index(drop=True)

