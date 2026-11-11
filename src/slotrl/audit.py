from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sb3_contrib import MaskablePPO
from .env import EnvConfig, SlotEnv
from .evaluate import bootstrap_mean_ci
from .policies import make_policy_set
from .sim import RequestPool

def trace_policy(env, action_fn, seed):
    obs,_=env.reset(seed=seed); actions=[]
    while True:
        mask=env.action_masks(); action=int(action_fn(obs,mask))
        if not mask[action]: raise AssertionError(f"Invalid action {action}")
        actions.append(action); obs,_,terminated,truncated,info=env.step(action)
        if terminated or truncated: return info,actions

def audit_policies(pool, cfg, model_paths, tau, max_load, n_seeds=None, days_per_seed=None):
    ec=EnvConfig(**cfg["environment"]); n_seeds=n_seeds or cfg["evaluation"]["n_seeds"]; days=days_per_seed or cfg["evaluation"]["days_per_seed"]
    policy_set=make_policy_set(tau,max_load,n_slots=cfg["environment"]["n_slots"]); models={Path(p).stem:MaskablePPO.load(p) for p in model_paths}
    all_policies={**policy_set}
    for name,model in models.items():
        all_policies[name]=lambda obs,mask,m=model: int(m.predict(obs,action_masks=mask,deterministic=True)[0])
    rows=[]; action_rows=[]; exposure=[]
    for seed in range(n_seeds):
        for day in range(days):
            day_seed=seed*100000+day
            for name,fn in all_policies.items():
                env=SlotEnv(pool,ec,seed=seed); info,actions=trace_policy(env,fn,day_seed)
                rows.append({"policy":name,"seed":seed,"day":day,**{k:v for k,v in info.items() if isinstance(v,(int,float))}})
                action_rows.append({"policy":name,"seed":seed,"day":day,"actions":json.dumps(actions),"declines":info["declines"]})
                for group,counts in info["exposure_counts"].items():
                    for label,(accepted,double_booked) in counts.items(): exposure.append({"policy":name,"seed":seed,"day":day,"group":group,"label":label,"accepted":accepted,"double_booked":double_booked})
    daily=pd.DataFrame(rows); actions=pd.DataFrame(action_rows); exposures=pd.DataFrame(exposure)
    reference="expected_load"; base=daily[daily.policy==reference].sort_values(["seed","day"])["reward_dollars"].to_numpy(); summary=[]
    for name,g in daily.groupby("policy"):
        vals=g.sort_values(["seed","day"])["reward_dollars"].to_numpy(); mean,lo,hi=bootstrap_mean_ci(vals,cfg["evaluation"]["bootstrap_reps"]); diff=vals-base
        pvalue=1.0 if name==reference or np.allclose(diff,0) else float(wilcoxon(diff).pvalue)
        summary.append({"policy":name,"mean_reward":mean,"ci_low":lo,"ci_high":hi,"mean_revenue":g.revenue.mean(),"mean_declines":g.declines.mean(),"mean_double_bookings":g.double_bookings.mean(),"mean_overtime":g.overtime_count.mean(),"mean_idle_slots":g.idle_slots.mean(),"mean_seen":g.seen.mean(),"delta_vs_expected_load":float(diff.mean()),"wilcoxon_p_vs_expected_load":pvalue})
    fairness=[]
    for (policy,group,label),g in exposures.groupby(["policy","group","label"]):
        accepted=g.accepted.sum(); double_booked=g.double_booked.sum(); fairness.append({"policy":policy,"group":group,"label":label,"accepted":int(accepted),"double_booked":int(double_booked),"exposure_rate":float(double_booked/accepted) if accepted else np.nan})
    fairness_df=pd.DataFrame(fairness)
    if not fairness_df.empty:
        totals=fairness_df.groupby("policy")[["double_booked","accepted"]].sum()
        totals["overall_exposure"]=totals.double_booked/totals.accepted.replace(0,np.nan)
        fairness_df=fairness_df.merge(totals[["overall_exposure"]],left_on="policy",right_index=True); fairness_df["gap_vs_overall"]=fairness_df.exposure_rate-fairness_df.overall_exposure; fairness_df["ratio_vs_overall"]=fairness_df.exposure_rate/fairness_df.overall_exposure.replace(0,np.nan)
    return daily,actions,pd.DataFrame(summary),fairness_df

