from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier
from .features import MODEL_FEATURES

def _split_by_date(df, cfg):
    dates = np.sort(df["appointment_day"].dt.date.unique())
    n = len(dates); a = max(1, int(n * cfg["data"]["train_fraction"])); b = max(a + 1, int(n * (cfg["data"]["train_fraction"] + cfg["data"]["calibration_fraction"])))
    return df["appointment_day"].dt.date.isin(dates[:a]), df["appointment_day"].dt.date.isin(dates[a:b]), df["appointment_day"].dt.date.isin(dates[b:])

def _preprocessor():
    numeric = [c for c in MODEL_FEATURES if c != "gender_m"]
    return ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric), ("gender", "passthrough", ["gender_m"])], remainder="drop")

def _metrics(y, p):
    prob = np.clip(p, 1e-7, 1 - 1e-7)
    return {"roc_auc": float(roc_auc_score(y, prob)), "pr_auc": float(average_precision_score(y, prob)), "brier": float(brier_score_loss(y, prob)), "log_loss": float(log_loss(y, prob))}

def train_and_save(df, cfg):
    train_m, cal_m, test_m = _split_by_date(df, cfg)
    Xtr, Xcal, Xte = df.loc[train_m, MODEL_FEATURES], df.loc[cal_m, MODEL_FEATURES], df.loc[test_m, MODEL_FEATURES]
    ytr, ycal, yte = df.loc[train_m, "no_show"], df.loc[cal_m, "no_show"], df.loc[test_m, "no_show"]
    pre = _preprocessor()
    logit = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=cfg["data"]["random_seed"]))])
    logit.fit(Xtr, ytr)
    xgb_cfg = cfg["model"]["xgb"].copy()
    xgb = Pipeline([("pre", _preprocessor()), ("model", XGBClassifier(**xgb_cfg, random_state=cfg["data"]["random_seed"], tree_method="hist"))])
    xgb.fit(Xtr, ytr)
    raw = xgb.predict_proba(Xcal)[:, 1]
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(xgb), method=cfg["model"]["calibration_method"])
    except ImportError:
        cal = CalibratedClassifierCV(xgb, method=cfg["model"]["calibration_method"], cv="prefit")
    cal.fit(Xcal, ycal)
    output = df.loc[test_m, ["appointment_id", "patient_id", "appointment_day", "age_band", "scholarship", "no_show"]].copy()
    output["p_noshow_raw"] = xgb.predict_proba(Xte)[:, 1]
    output["p_noshow_cal"] = cal.predict_proba(Xte)[:, 1]
    outdir = Path(cfg["paths"]["processed_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    output.to_parquet(outdir / "noshow_test.parquet", index=False)
    metrics = {"logistic_regression": _metrics(yte, logit.predict_proba(Xte)[:, 1]), "xgboost_raw": _metrics(yte, output["p_noshow_raw"]), "xgboost_calibrated": _metrics(yte, output["p_noshow_cal"])}
    metrics["split_sizes"] = {"train": int(train_m.sum()), "calibration": int(cal_m.sum()), "test": int(test_m.sum())}
    (outdir / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    modeldir = Path(cfg["paths"]["model_dir"]); modeldir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": xgb, "calibrator": cal, "features": MODEL_FEATURES}, modeldir / "noshow_model.joblib")
    return metrics, output

