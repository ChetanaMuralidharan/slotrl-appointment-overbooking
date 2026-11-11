from __future__ import annotations
import argparse, json
from pathlib import Path
from .config import load_config, ensure_dirs
from .data import load_and_clean
from .features import make_features
from .noshow_model import train_and_save
from .sim import RequestPool, split_rl_pools
from .env import SlotEnv, EnvConfig
from .diagnostics import create_calibration_diagnostics
from .evaluate import evaluate_baselines, tune_baselines
from .audit import audit_policies
from .sensitivity import run_baseline_sensitivity
from .reporting import finalize_policy_report, run_calibration_ablation
from .train import train_agent, evaluate_agent_models
from .policies import make_policy_set
from .experiments import train_experiment_suite, evaluate_experiment_suite, paired_experiment_comparisons

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="command", required=True); 
    for name in ["prepare-data","train-noshow","diagnostics","build-pools","check-env","evaluate-baselines","train-agent","evaluate-agent","audit-agent","sensitivity","calibration-ablation","finalize-report","train-experiments","evaluate-experiments"]:
        p=sub.add_parser(name); p.add_argument("--config", default="configs/default.yaml")
    args=ap.parse_args(); cfg=load_config(args.config); ensure_dirs(cfg, Path(args.config).parent.parent)
    root=Path(args.config).parent.parent; raw=root/cfg["paths"]["raw_csv"]
    df, summary=load_and_clean(raw); df=make_features(df)
    if args.command=="prepare-data":
        print(json.dumps(summary.__dict__, indent=2)); return
    if args.command=="train-noshow":
        metrics,_=train_and_save(df,cfg); print(json.dumps(metrics, indent=2)); return
    import pandas as pd
    processed=root/cfg["paths"]["processed_dir"]
    test_frame=pd.read_parquet(processed/"noshow_test.parquet")
    if args.command=="diagnostics":
        print(json.dumps(create_calibration_diagnostics(test_frame, root/cfg["paths"]["results_dir"]), indent=2)); return
    if args.command=="build-pools":
        train_pool,eval_pool=split_rl_pools(test_frame,cfg["data"]["rl_train_fraction_of_test"])
        train_pool.to_parquet(processed/"rl_train_pool.parquet",index=False); eval_pool.to_parquet(processed/"rl_eval_pool.parquet",index=False)
        metadata={"rl_train_rows":len(train_pool),"rl_eval_rows":len(eval_pool),"rl_train_dates":int(pd.to_datetime(train_pool.appointment_day,utc=True).dt.date.nunique()),"rl_eval_dates":int(pd.to_datetime(eval_pool.appointment_day,utc=True).dt.date.nunique()),"rl_train_max_date":str(train_pool.appointment_day.max()),"rl_eval_min_date":str(eval_pool.appointment_day.min()),"rl_eval_max_date":str(eval_pool.appointment_day.max())}
        (processed/"rl_pool_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
        print(json.dumps(metadata,indent=2)); return
    pool_path=processed/("rl_eval_pool.parquet" if args.command=="evaluate-baselines" else "rl_train_pool.parquet")
    pool=RequestPool(pd.read_parquet(pool_path))
    env=SlotEnv(pool, EnvConfig(**cfg["environment"])); from gymnasium.utils.env_checker import check_env; check_env(env); print("Environment passed Gymnasium checks")
    if args.command=="evaluate-baselines":
        train_pool=RequestPool(pd.read_parquet(processed/"rl_train_pool.parquet")); tuning=tune_baselines(train_pool,cfg)
        raw,summary=evaluate_baselines(pool,cfg,tau=tuning["threshold_tau"],max_load=tuning["expected_load_max"])
        results=root/cfg["paths"]["results_dir"]; results.mkdir(parents=True,exist_ok=True)
        raw.to_csv(results/"baseline_daily_results.csv",index=False); summary.to_csv(results/"baseline_summary.csv",index=False); (results/"baseline_tuning.json").write_text(json.dumps(tuning,indent=2),encoding="utf-8")
        print(json.dumps(tuning,indent=2)); print(summary.to_string(index=False))
    if args.command=="train-agent":
        train_pool=RequestPool(pd.read_parquet(processed/"rl_train_pool.parquet")); paths=train_agent(train_pool,cfg); print(json.dumps(paths,indent=2)); return
    if args.command=="train-experiments":
        print(json.dumps(train_experiment_suite(pd.read_parquet(processed/"rl_train_pool.parquet"), cfg), indent=2)); return
    if args.command=="evaluate-experiments":
        results=root/cfg["paths"]["results_dir"]
        tuning=json.loads((results/"baseline_tuning.json").read_text(encoding="utf-8"))
        daily,summary=evaluate_experiment_suite(pd.read_parquet(processed/"rl_eval_pool.parquet"),cfg,tuning)
        daily.to_csv(results/"experiment_daily_results.csv",index=False); summary.to_csv(results/"experiment_summary.csv",index=False)
        paired_experiment_comparisons(daily).to_csv(results/"experiment_comparisons.csv",index=False)
        print(summary.to_string(index=False)); return
    if args.command=="evaluate-agent":
        eval_pool=RequestPool(pd.read_parquet(processed/"rl_eval_pool.parquet")); train_pool=RequestPool(pd.read_parquet(processed/"rl_train_pool.parquet"))
        tuning_path=root/cfg["paths"]["results_dir"]/"baseline_tuning.json"
        if tuning_path.exists(): tuning=json.loads(tuning_path.read_text(encoding="utf-8"))
        else: tuning=tune_baselines(train_pool,cfg)
        model_paths=sorted(str(p) for p in Path(cfg["paths"]["model_dir"]).glob("maskable_ppo_seed_*.zip"))
        if not model_paths: raise FileNotFoundError("No PPO models found. Run train-agent first.")
        raw=evaluate_agent_models(eval_pool,cfg,model_paths,make_policy_set(tuning["threshold_tau"],tuning["expected_load_max"]))
        results=root/cfg["paths"]["results_dir"]; raw.to_csv(results/"agent_and_baseline_daily_results.csv",index=False)
        summary=raw.groupby("policy").agg(mean_reward=("reward_dollars","mean"),std_reward=("reward_dollars","std"),mean_revenue=("revenue","mean"),mean_overtime=("overtime_count","mean"),mean_declines=("declines","mean"),count=("reward_dollars","count")).reset_index(); summary.to_csv(results/"agent_and_baseline_summary.csv",index=False); print(summary.to_string(index=False))
    if args.command in {"audit-agent","sensitivity"}:
        results=root/cfg["paths"]["results_dir"]; processed=root/cfg["paths"]["processed_dir"]
        tuning=json.loads((results/"baseline_tuning.json").read_text(encoding="utf-8"))
        eval_pool=RequestPool(pd.read_parquet(processed/"rl_eval_pool.parquet")); train_pool=RequestPool(pd.read_parquet(processed/"rl_train_pool.parquet"))
        model_paths=sorted(str(p) for p in Path(cfg["paths"]["model_dir"]).glob("maskable_ppo_seed_*.zip"))
        if args.command=="audit-agent":
            daily,actions,summary,fairness=audit_policies(eval_pool,cfg,model_paths,tuning["threshold_tau"],tuning["expected_load_max"])
            daily.to_csv(results/"behavior_daily_results.csv",index=False); actions.to_csv(results/"behavior_actions.csv",index=False); summary.to_csv(results/"behavior_summary.csv",index=False); fairness.to_csv(results/"fairness_exposure.csv",index=False); print(summary.to_string(index=False)); print(fairness.to_string(index=False))
        else:
            sens=run_baseline_sensitivity(train_pool,cfg,tuning["threshold_tau"],tuning["expected_load_max"],[30,60,90,120,150],[0.10,0.20,0.30],n_seeds=5,days_per_seed=50); sens.to_csv(results/"baseline_sensitivity.csv",index=False); print(sens.to_string(index=False))
    if args.command in {"calibration-ablation","finalize-report"}:
        results=root/cfg["paths"]["results_dir"]; processed=root/cfg["paths"]["processed_dir"]
        if args.command=="calibration-ablation":
            test=pd.read_parquet(processed/"noshow_test.parquet"); train_pool,eval_pool=split_rl_pools(test,cfg["data"]["rl_train_fraction_of_test"]); print(run_calibration_ablation(train_pool,eval_pool,cfg,results).to_string(index=False))
        else: print(finalize_policy_report(results,cfg["evaluation"]["bootstrap_reps"]).to_string(index=False))

if __name__ == "__main__": main()

