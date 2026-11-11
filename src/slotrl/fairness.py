from __future__ import annotations
import pandas as pd

def exposure_rates(records: pd.DataFrame, group: str) -> pd.DataFrame:
    if records.empty: return pd.DataFrame(columns=[group,"accepted","double_booked","exposure_rate"])
    accepted=records[records["accepted"]].copy();
    out=accepted.groupby(group, dropna=False).agg(accepted=("accepted","size"), double_booked=("double_booked","sum")).reset_index()
    out["exposure_rate"]=out["double_booked"]/out["accepted"].replace(0, float("nan")); overall=out["double_booked"].sum()/max(1,out["accepted"].sum()); out["gap_vs_overall"]=out["exposure_rate"]-overall; out["ratio_vs_overall"]=out["exposure_rate"]/overall if overall else float("nan")
    return out

