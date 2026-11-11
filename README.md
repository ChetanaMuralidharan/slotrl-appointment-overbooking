# SlotRL

The results below are from the corrected reviewed rerun using strict
chronological `neighbourhood_frequency` history.

SlotRL is a research-grade simulation of revenue-aware outpatient appointment overbooking. It combines a calibrated no-show model with a masked reinforcement-learning scheduler and strong heuristic baselines.

This repository is deliberately honest about scope. It uses the public Brazilian Medical Appointment No Shows dataset, simulates request arrivals and provider capacity, and uses configurable dollar assumptions. Results are simulation results, not evidence of real clinic revenue.

## Architecture

1. `src/slotrl/data.py` loads and cleans the raw file.
2. `src/slotrl/features.py` creates leakage-safe, time-aware features.
3. `src/slotrl/noshow_model.py` trains logistic regression and XGBoost, then calibrates probabilities on a disjoint chronological slice.
4. `src/slotrl/sim.py` creates seeded request pools and paired simulated days.
5. `src/slotrl/env.py` defines the Gymnasium MDP and action mask.
6. `src/slotrl/policies.py` provides four baselines.
7. `src/slotrl/train.py` trains MaskablePPO.
8. `src/slotrl/evaluate.py` evaluates all policies on identical paired days and computes bootstrap intervals.
9. `src/slotrl/fairness.py` reports double-booking exposure by group.
10. `src/slotrl/experiments.py` runs registered PPO ablations with paired evaluation and Wilcoxon comparisons.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install -e .
python -m slotrl.cli prepare-data --config configs/default.yaml
python -m slotrl.cli train-noshow --config configs/default.yaml
python -m slotrl.cli diagnostics --config configs/default.yaml
python -m slotrl.cli build-pools --config configs/default.yaml
python -m slotrl.cli check-env --config configs/default.yaml
python -m slotrl.cli evaluate-baselines --config configs/default.yaml
python -m slotrl.cli train-agent --config configs/default.yaml
python -m slotrl.cli evaluate-agent --config configs/default.yaml
python -m slotrl.cli audit-agent --config configs/default.yaml
python -m slotrl.cli sensitivity --config configs/default.yaml
python -m slotrl.cli calibration-ablation --config configs/default.yaml
python -m slotrl.cli finalize-report --config configs/default.yaml
python -m slotrl.cli train-experiments --config configs/default.yaml
python -m slotrl.cli evaluate-experiments --config configs/default.yaml
python -m pytest
```

The raw CSV must be at `data/raw/KaggleV2-May-2016.csv`. The pipeline writes only derived artifacts under `data/processed`, `models`, and `results`.
The `build-pools` command records the exact number and date range of RL-train
and RL-evaluation dates in `data/processed/rl_pool_metadata.json`. The reported
evaluation-date count must be cited with the final results.

## Reviewed rerun after feature correction

`configs/reviewed.yaml` writes to separate `data/processed_reviewed`,
`models_reviewed`, and `results_reviewed` directories. Use it when rerunning
after the chronological `neighbourhood_frequency` correction so old and new
artifacts cannot be mixed:

```text
python -m slotrl.cli prepare-data --config configs/reviewed.yaml
python -m slotrl.cli train-noshow --config configs/reviewed.yaml
python -m slotrl.cli diagnostics --config configs/reviewed.yaml
python -m slotrl.cli build-pools --config configs/reviewed.yaml
python -m slotrl.cli evaluate-baselines --config configs/reviewed.yaml
python -m slotrl.cli train-agent --config configs/reviewed.yaml
python -m slotrl.cli evaluate-agent --config configs/reviewed.yaml
python -m slotrl.cli audit-agent --config configs/reviewed.yaml
python -m slotrl.cli sensitivity --config configs/reviewed.yaml
python -m slotrl.cli calibration-ablation --config configs/reviewed.yaml
python -m slotrl.cli train-experiments --config configs/reviewed.yaml
python -m slotrl.cli evaluate-experiments --config configs/reviewed.yaml
python -m slotrl.cli finalize-report --config configs/reviewed.yaml
```

This rerun is required before replacing the current headline model metrics.

## MDP definition

An episode is one simulated provider day. The state contains each slot's booking count and expected show load, the current request's calibrated no-show probability, requests remaining, and double-booking-cap usage. Actions are slots `0..15` plus decline. A slot at capacity is invalid; once the daily double-booking cap is reached, a single-booked slot is invalid. Decline is always valid.

At the end of the day, reward is:

`visit_value * patients_seen - overtime_cost * slots_with_two_shows - decline_cost * declined_requests`

The true outcome is held out from the agent and is used only when the episode settles. Rewards are scaled by `reward_scale` during training while unscaled dollar metrics remain in `info`.

## Data limitations

The source is not dental data, covers one Brazilian region and a short period, and does not represent one provider's demand. The simulated request process, capacity, and dollar values are assumptions. There are no cancellations, rebooking, wait-time distributions, or real logging-policy data. A policy that performs well here still requires shadow-mode evaluation, off-policy evaluation, operational guardrails, and human approval before any deployment.

## Reproducibility

All random streams are seeded. Evaluation uses paired request sequences and realized outcomes for every policy. Thresholds are tuned only on the RL-training pool; the later evaluation pool remains untouched. Every result should be regenerated from the saved per-day CSV rather than copied manually.

## Research ablations

After the primary calibrated MaskablePPO run, the repository provides a fixed
15-model experiment matrix. It keeps the same chronological pools, five random
seeds, evaluation days, and baseline policies for every comparison:

- raw no-show probabilities with masked PPO;
- calibrated probabilities with ordinary, unmasked PPO and an explicit invalid-action penalty;
- calibrated masked PPO with `gamma=1.0`.

Run the suite from the project root:

```text
python -m slotrl.cli train-experiments --config configs/default.yaml
python -m slotrl.cli evaluate-experiments --config configs/default.yaml
pytest
```

Training runs independent seeds in parallel using `training.parallel_workers`
(default: 3). Completed seed files are skipped, so an interrupted run resumes
without overwriting finished models. Increase this only if CPU temperature,
memory, and responsiveness remain acceptable.

Training also records one learning-curve CSV per seed. These curves show the
rolling terminal episode reward during training. They are diagnostic evidence,
not an independent evaluation result. To test whether the primary PPO is
undertrained, create a copy of the configuration with
`training.total_timesteps: 1000000`, train a separately named control, and
compare its untouched evaluation performance before changing the headline
conclusion.

The training command writes one directory per registered variant under
`models/experiments/`. Evaluation writes `results/experiment_daily_results.csv`,
`results/experiment_summary.csv`, and the paired same-day comparison file
`results/experiment_comparisons.csv`. Do not change the evaluation pool or
seeds after inspecting results; all tuning remains restricted to the RL-train
pool.

## Final result

The primary evaluation used 4,000 paired simulated days per policy and measured
mean net reward per provider-day. The expected-load baseline achieved
**$2,332.36/day** with 2.10 overtime incidents per day. Across the five primary
PPO seeds, mean reward was **$2,305.30/day**. PPO did not outperform the tuned
heuristic, but it substantially outperformed single-book and always-overbook
in the tested simulator.

Those 4,000 simulated days were generated from a three-calendar-date held-out
evaluation pool (`2016-06-06` through `2016-06-08`), not from 4,000 real clinic
days.

Across the three ablation variants, mean reward was $2,305.19/day for
calibrated masked PPO with `gamma=1.0`, $2,303.84/day for raw masked PPO, and
$2,303.96/day for calibrated unmasked PPO. These results support the conclusion
that the simpler expected-load policy is the recommended policy under the
documented assumptions.

The raw probability model had slightly better held-out Brier score and ECE
than the calibrated model. On the same 4,000 paired simulated days, calibration
increased expected-load reward by **$7.75/day**; the paired Wilcoxon p-value was
`3.75e-30`. The median daily difference was $0, so the statistically detectable
effect is small and concentrated in a subset of days.

With 20 requests, 16 slots, and a cap of 8 double-bookings, policies that
accept all requests earn the same visit revenue on each realized day. The
primary decision problem is therefore how requests are paired into double-booked
slots. Expected-load earned more reward, but its exposure was less uniform:
18–34 patients were double-booked at approximately 1.37x the overall rate,
scholarship patients at approximately 1.31x, and patients 65+ at approximately
0.49x. PPO was closer to uniform exposure, approximately 0.69–1.22x across the
reported groups, at an estimated reward cost of about $27/day (1.2%).

These are offline simulation results. The project does not establish causal
revenue improvement or deployment safety. The fairness analysis measures
double-booking exposure only and does not establish equal access, outcomes, or
absence of disparate impact. See `FINAL_REPORT.md` for the complete research
summary and limitations.
