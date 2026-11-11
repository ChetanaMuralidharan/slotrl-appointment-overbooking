from __future__ import annotations
import pandas as pd
from .env import EnvConfig, SlotEnv
from .evaluate import run_policy
from .policies import make_policy_set
from .sim import RequestPool

def run_baseline_sensitivity(pool, cfg, tau, max_load, overtime_costs, base_rates, n_seeds=5, days_per_seed=50):
    rows=[]
    for overtime in overtime_costs:
        for base_rate in base_rates:
            ec=EnvConfig(**{**cfg["environment"],"overtime_cost":float(overtime)})
            for name,fn in make_policy_set(tau,max_load,n_slots=cfg["environment"]["n_slots"]).items():
                values=[]
                for seed in range(n_seeds):
                    env=SlotEnv(pool,ec,seed=seed,base_rate=float(base_rate))
                    for day in range(days_per_seed): values.append(run_policy(env,fn,seed*100000+day)["reward_dollars"])
                rows.append({"policy":name,"overtime_cost":overtime,"base_no_show_rate":base_rate,"mean_reward":sum(values)/len(values),"n_days":len(values)})
    return pd.DataFrame(rows)

