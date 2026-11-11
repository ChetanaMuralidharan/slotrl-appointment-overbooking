from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from .evaluate import bootstrap_mean_ci, evaluate_baselines, tune_baselines
from .policies import make_policy_set
from .sim import RequestPool
from .env import EnvConfig, SlotEnv
from .evaluate import run_policy

def finalize_policy_report(results_dir, bootstrap_reps=2000):
    results=Path(results_dir); daily=pd.read_csv(results/"behavior_daily_results.csv")
    rows=[]
    for policy,g in daily.groupby("policy"):
        vals=g.sort_values(["seed","day"])["reward_dollars"].to_numpy(); mean,lo,hi=bootstrap_mean_ci(vals,bootstrap_reps)
        rows.append({"policy":policy,"mean_reward":mean,"ci_low":lo,"ci_high":hi,"mean_revenue":g.revenue.mean(),"mean_overtime":g.overtime_count.mean(),"mean_declines":g.declines.mean(),"mean_double_bookings":g.double_bookings.mean(),"mean_seen":g.seen.mean()})
    summary=pd.DataFrame(rows).sort_values("mean_reward",ascending=False); summary.to_csv(results/"final_policy_summary.csv",index=False)
    plt.figure(figsize=(10,6)); x=np.arange(len(summary)); plt.bar(x,summary.mean_reward,yerr=[summary.mean_reward-summary.ci_low,summary.ci_high-summary.mean_reward],capsize=4); plt.xticks(x,summary.policy,rotation=35,ha="right"); plt.ylabel("Net reward per provider-day ($)"); plt.title("Policy net reward with 95% bootstrap intervals"); plt.tight_layout(); plt.savefig(results/"reward_by_policy.png",dpi=180); plt.close()
    plt.figure(figsize=(10,6)); sns.barplot(data=summary,x="policy",y="mean_overtime",errorbar=None); plt.xticks(rotation=35,ha="right"); plt.ylabel("Mean overtime incidents per day"); plt.title("Overtime by policy"); plt.tight_layout(); plt.savefig(results/"overtime_by_policy.png",dpi=180); plt.close()
    fairness=pd.read_csv(results/"fairness_exposure.csv"); fairness["overall_exposure"] = fairness.groupby("policy")["double_booked"].transform("sum") / fairness.groupby("policy")["accepted"].transform("sum").replace(0,np.nan); fairness["gap_vs_overall"]=fairness.exposure_rate-fairness.overall_exposure; fairness["ratio_vs_overall"]=fairness.exposure_rate/fairness.overall_exposure.replace(0,np.nan); fairness.to_csv(results/"fairness_exposure_with_gaps.csv",index=False)
    plt.figure(figsize=(11,6)); sns.barplot(data=fairness,x="label",y="exposure_rate",hue="policy",errorbar=None); plt.ylabel("Double-booking exposure rate"); plt.xlabel("Group"); plt.title("Double-booking exposure by group"); plt.xticks(rotation=30); plt.tight_layout(); plt.savefig(results/"fairness_exposure.png",dpi=180); plt.close()
    sensitivity=pd.read_csv(results/"baseline_sensitivity.csv"); pivot=sensitivity.pivot_table(index="overtime_cost",columns="base_no_show_rate",values="mean_reward",aggfunc="max"); plt.figure(figsize=(8,5)); sns.heatmap(pivot,annot=True,fmt=".0f",cmap="YlGnBu"); plt.title("Best baseline net reward sensitivity"); plt.tight_layout(); plt.savefig(results/"sensitivity_heatmap.png",dpi=180); plt.close()
    return summary

def run_calibration_ablation(train_frame, eval_frame, cfg, results_dir):
    rows=[]; paired_frames=[]; results=Path(results_dir); results.mkdir(parents=True,exist_ok=True)
    for source in ["p_noshow_cal","p_noshow_raw"]:
        tr=train_frame.copy(); ev=eval_frame.copy(); tr["p_noshow_cal"]=tr[source]; ev["p_noshow_cal"]=ev[source]
        train_pool=RequestPool(tr); eval_pool=RequestPool(ev); tuning=tune_baselines(train_pool,cfg)
        raw,summary=evaluate_baselines(eval_pool,cfg,tau=tuning["threshold_tau"],max_load=tuning["expected_load_max"])
        paired_frames.append(raw[raw.policy == "expected_load"][["seed", "day", "reward_dollars"]].rename(columns={"reward_dollars": "calibrated_reward" if source.endswith("cal") else "raw_reward"}))
        for _,r in summary.iterrows(): rows.append({"probability_source":"calibrated" if source.endswith("cal") else "raw","policy":r.policy,"mean_reward":r.mean_reward,"ci_low":r.ci_low,"ci_high":r.ci_high,"mean_overtime":r.mean_overtime,"mean_declines":r.mean_declines})
    out=pd.DataFrame(rows); out.to_csv(results/"calibration_ablation.csv",index=False)
    paired=paired_frames[0].merge(paired_frames[1], on=["seed", "day"], how="inner")
    paired["calibrated_minus_raw"] = paired["calibrated_reward"] - paired["raw_reward"]
    differences=paired["calibrated_minus_raw"].to_numpy()
    paired_result={
        "policy": "expected_load",
        "paired_days": int(len(paired)),
        "mean_calibrated_minus_raw": float(paired["calibrated_minus_raw"].mean()),
        "median_calibrated_minus_raw": float(paired["calibrated_minus_raw"].median()),
        "wilcoxon_pvalue": float(wilcoxon(differences).pvalue) if np.any(differences) else 1.0,
    }
    (results / "calibration_paired_test.json").write_text(json.dumps(paired_result, indent=2), encoding="utf-8")
    paired.to_csv(results / "calibration_paired_daily.csv", index=False)
    return out

