from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def expected_calibration_error(y_true, probabilities, n_bins=10):
    y=np.asarray(y_true, dtype=float); p=np.asarray(probabilities, dtype=float)
    edges=np.linspace(0.0,1.0,n_bins+1); ece=0.0; rows=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
        count=int(mask.sum())
        if count:
            confidence=float(p[mask].mean()); accuracy=float(y[mask].mean()); ece += count/len(y)*abs(confidence-accuracy)
            rows.append({"bin_lower":lo,"bin_upper":hi,"count":count,"mean_predicted":confidence,"observed_rate":accuracy})
        else: rows.append({"bin_lower":lo,"bin_upper":hi,"count":0,"mean_predicted":np.nan,"observed_rate":np.nan})
    return float(ece), pd.DataFrame(rows)

def create_calibration_diagnostics(test_frame, output_dir, n_bins=10):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    y=test_frame["no_show"].to_numpy(); metrics={}; curves={}
    for name,col in [("raw","p_noshow_raw"),("calibrated","p_noshow_cal")]:
        p=test_frame[col].to_numpy(); ece,bins=expected_calibration_error(y,p,n_bins)
        metrics[name]={"brier":float(brier_score_loss(y,p)),"ece":ece}
        bins.to_csv(output_dir/f"calibration_bins_{name}.csv",index=False)
        frac,mean=calibration_curve(y,p,n_bins=n_bins,strategy="quantile"); curves[name]=(mean,frac)
    fig,ax=plt.subplots(figsize=(7,6)); ax.plot([0,1],[0,1],"k--",label="Perfect calibration")
    for name,(mean,frac) in curves.items(): ax.plot(mean,frac,marker="o",label=name.title())
    ax.set(xlabel="Mean predicted no-show probability",ylabel="Observed no-show rate",title="No-show probability calibration"); ax.legend(); fig.tight_layout(); fig.savefig(output_dir/"calibration_curve.png",dpi=160); plt.close(fig)
    (output_dir/"calibration_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    return metrics

