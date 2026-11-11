from __future__ import annotations

"""Registered, reproducible ablations for the SlotRL study."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

from .env import EnvConfig, SlotEnv
from .evaluate import run_policy
from .policies import make_policy_set
from .sim import RequestPool
from .train import train_agent


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    probability_column: str
    algorithm: str
    allow_invalid_actions: bool
    gamma: float
    ent_coef: float
    learning_rate: float


def default_experiment_specs(cfg: dict) -> list[ExperimentSpec]:
    t = cfg["training"]
    all_specs = [
        ExperimentSpec("raw_masked_ppo", "p_noshow_raw", "maskable_ppo", False, t["gamma"], t["ent_coef"], t["learning_rate"]),
        ExperimentSpec("calibrated_unmasked_ppo", "p_noshow_cal", "ppo", True, t["gamma"], t["ent_coef"], t["learning_rate"]),
        ExperimentSpec("calibrated_gamma_1_0", "p_noshow_cal", "maskable_ppo", False, 1.0, t["ent_coef"], t["learning_rate"]),
        ExperimentSpec("calibrated_no_entropy", "p_noshow_cal", "maskable_ppo", False, t["gamma"], 0.0, t["learning_rate"]),
    ]
    enabled = cfg.get("experiments", {}).get("enabled")
    if enabled is None:
        return all_specs
    known = {spec.name for spec in all_specs}
    unknown = set(enabled) - known
    if unknown:
        raise ValueError(f"Unknown experiment names in config: {sorted(unknown)}")
    selected = [spec for spec in all_specs if spec.name in enabled]
    if not selected:
        raise ValueError("experiments.enabled must select at least one experiment")
    return selected


def train_experiment_suite(train_frame: pd.DataFrame, cfg: dict) -> dict[str, list[str]]:
    root = Path(cfg["paths"]["model_dir"]) / "experiments"
    root.mkdir(parents=True, exist_ok=True)
    trained: dict[str, list[str]] = {}
    for spec in default_experiment_specs(cfg):
        pool = RequestPool(train_frame, probability_column=spec.probability_column)
        paths = train_agent(pool, cfg, model_prefix=spec.name, algorithm=spec.algorithm,
                            allow_invalid_actions=spec.allow_invalid_actions,
                            hyperparams={"gamma": spec.gamma, "ent_coef": spec.ent_coef, "learning_rate": spec.learning_rate},
                            output_dir=root / spec.name)
        (root / spec.name / "experiment_spec.json").write_text(json.dumps(asdict(spec), indent=2), encoding="utf-8")
        trained[spec.name] = paths
    return trained


def _load_model(spec: ExperimentSpec, path: str):
    return (MaskablePPO if spec.algorithm == "maskable_ppo" else PPO).load(path)


def evaluate_experiment_suite(eval_frame: pd.DataFrame, cfg: dict, tuning: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(cfg["paths"]["model_dir"]) / "experiments"
    ec = EnvConfig(**cfg["environment"])
    rows = []
    for spec in default_experiment_specs(cfg):
        pool = RequestPool(eval_frame, probability_column=spec.probability_column)
        paths = sorted((root / spec.name).glob(f"{spec.name}_seed_*.zip"))
        expected = {f"{spec.name}_seed_{seed}.zip" for seed in cfg["training"]["seeds"]}
        actual = {p.name for p in paths}
        if actual != expected:
            raise FileNotFoundError(f"Incomplete experiment {spec.name}: expected {sorted(expected)}, found {sorted(actual)}")
        for path in paths:
            model = _load_model(spec, str(path))
            for seed in range(cfg["evaluation"]["n_seeds"]):
                env = SlotEnv(pool, ec, seed=seed, allow_invalid_actions=spec.allow_invalid_actions)
                for day in range(cfg["evaluation"]["days_per_seed"]):
                    day_seed = seed * 100000 + day
                    def action_fn(obs, mask, model=model, masked=spec.algorithm == "maskable_ppo"):
                        action, _ = (model.predict(obs, action_masks=mask, deterministic=True) if masked else model.predict(obs, deterministic=True))
                        return int(action)
                    info = run_policy(env, action_fn, day_seed, enforce_mask=spec.algorithm == "maskable_ppo")
                    rows.append({"experiment": spec.name, "policy": path.stem, "seed": seed, "day": day,
                                 **{k: v for k, v in info.items() if isinstance(v, (int, float))}})

    pool = RequestPool(eval_frame, probability_column="p_noshow_cal")
    policies = make_policy_set(tuning["threshold_tau"], tuning["expected_load_max"], n_slots=ec.n_slots)
    for name, fn in policies.items():
        for seed in range(cfg["evaluation"]["n_seeds"]):
            env = SlotEnv(pool, ec, seed=seed)
            for day in range(cfg["evaluation"]["days_per_seed"]):
                info = run_policy(env, fn, seed * 100000 + day)
                rows.append({"experiment": "reference_baselines", "policy": name, "seed": seed, "day": day,
                             **{k: v for k, v in info.items() if isinstance(v, (int, float))}})

    daily = pd.DataFrame(rows)
    summary = daily.groupby(["experiment", "policy"], as_index=False).agg(
        mean_reward=("reward_dollars", "mean"), std_reward=("reward_dollars", "std"),
        mean_revenue=("revenue", "mean"), mean_overtime=("overtime_count", "mean"),
        mean_invalid_actions=("invalid_actions", "mean"), mean_double_bookings=("double_bookings", "mean"),
        count=("reward_dollars", "count"))
    return daily, summary


def paired_experiment_comparisons(daily: pd.DataFrame, reference_policy: str = "expected_load") -> pd.DataFrame:
    """Compare every learned policy to the reference on identical seed/day pairs."""
    ref = daily[daily["policy"] == reference_policy][["seed", "day", "reward_dollars"]].rename(columns={"reward_dollars": "reference_reward"})
    rows = []
    for (experiment, policy), group in daily.groupby(["experiment", "policy"]):
        if policy == reference_policy:
            continue
        paired = group.merge(ref, on=["seed", "day"], how="inner")
        diff = paired["reward_dollars"] - paired["reference_reward"]
        rows.append({"experiment": experiment, "policy": policy, "paired_days": len(paired),
                     "mean_reward_difference_vs_reference": float(diff.mean()),
                     "median_reward_difference_vs_reference": float(diff.median()),
                     "wilcoxon_p_vs_reference": float(wilcoxon(diff).pvalue) if len(diff) and np.any(diff) else 1.0})
    return pd.DataFrame(rows)
