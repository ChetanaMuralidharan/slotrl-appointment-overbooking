from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from .env import EnvConfig, SlotEnv
from .policies import make_policy_set
from .sim import RequestPool

def bootstrap_mean_ci(values, reps=2000, seed=0):
    x=np.asarray(values, dtype=float); rng=np.random.default_rng(seed)
    means=np.mean(rng.choice(x, size=(reps, len(x)), replace=True), axis=1)
    return float(np.mean(x)), float(np.quantile(means,.025)), float(np.quantile(means,.975))

def run_policy(env, action_fn, seed, enforce_mask=True):
    obs, _=env.reset(seed=seed); records=[]
    while True:
        mask=env.action_masks(); action=action_fn(obs, mask)
        if enforce_mask and not mask[action]: raise AssertionError(f"Policy returned invalid action {action}")
        obs,reward,terminated,truncated,info=env.step(action)
        if terminated or truncated:
            return info

def evaluate_policy_set(pool, cfg, policy_set, n_seeds=None, days_per_seed=None):
    ec=EnvConfig(**cfg["environment"]); n_seeds=n_seeds or cfg["evaluation"]["n_seeds"]; days=days_per_seed or cfg["evaluation"]["days_per_seed"]
    rows=[]
    for seed in range(n_seeds):
        env=SlotEnv(pool, ec, seed=seed)
        for day in range(days):
            day_seed=seed*100000+day
            for name,fn in policy_set.items():
                info=run_policy(env, fn, day_seed)
                rows.append({"seed":seed,"day":day,"policy":name,**{k:v for k,v in info.items() if isinstance(v,(int,float))}})
    result=pd.DataFrame(rows)
    best=result.groupby("policy")["reward_dollars"].mean().idxmax()
    summary=[]
    best_values=result.loc[result.policy==best,"reward_dollars"].to_numpy()
    for policy,g in result.groupby("policy"):
        vals=g.sort_values(["seed","day"])["reward_dollars"].to_numpy(); mean,lo,hi=bootstrap_mean_ci(vals,cfg["evaluation"]["bootstrap_reps"])
        if policy==best: pvalue=1.0
        else:
            diff=vals-best_values
            pvalue=float(wilcoxon(diff).pvalue) if np.any(diff) else 1.0
        summary.append({"policy":policy,"mean_reward":mean,"ci_low":lo,"ci_high":hi,"wilcoxon_vs_best_p":pvalue,"mean_revenue":float(g["revenue"].mean()),"mean_overtime":float(g["overtime_count"].mean()),"mean_declines":float(g["declines"].mean()),"mean_double_bookings":float(g["double_bookings"].mean()),"mean_seen":float(g["seen"].mean())})
    return result,pd.DataFrame(summary)

def evaluate_baselines(pool, cfg, n_seeds=None, days_per_seed=None, tau=0.35, max_load=1.05):
    return evaluate_policy_set(pool,cfg,make_policy_set(tau,max_load,n_slots=cfg["environment"]["n_slots"]),n_seeds,days_per_seed)

def tune_baselines(train_pool, cfg):
    best_tau=(None,-np.inf); best_load=(None,-np.inf)
    # Include the full operational range. If tau=0 remains optimal, report it
    # as a boundary solution rather than implying a meaningful risk threshold.
    for tau in np.linspace(0.00, 0.95, 20):
        _,summary=evaluate_baselines(train_pool,cfg,n_seeds=5,days_per_seed=50,tau=float(tau),max_load=1.05)
        score=float(summary.loc[summary.policy=="threshold","mean_reward"].iloc[0])
        if score>best_tau[1]: best_tau=(float(tau),score)
    for load in [0.80,0.90,1.00,1.10,1.20,1.40,1.60,1.80,2.00]:
        _,summary=evaluate_baselines(train_pool,cfg,n_seeds=5,days_per_seed=50,tau=best_tau[0],max_load=float(load))
        score=float(summary.loc[summary.policy=="expected_load","mean_reward"].iloc[0])
        if score>best_load[1]: best_load=(float(load),score)
    return {"threshold_tau":best_tau[0],"expected_load_max":best_load[0],"tuning_mean_reward_threshold":best_tau[1],"tuning_mean_reward_expected_load":best_load[1]}

