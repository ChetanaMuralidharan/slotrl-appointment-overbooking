# SlotRL Final Research Report

> This report reflects the corrected reviewed rerun using strict chronological
> `neighbourhood_frequency` history and paired calibration testing.

## Executive summary

SlotRL studies revenue-aware outpatient appointment overbooking using a
time-aware no-show model, a constrained Gymnasium scheduling environment, and
MaskablePPO. The project compares learned policies with operational heuristics
under an explicitly documented offline simulator.

The main result is that the tuned expected-load heuristic achieved the highest
mean net reward: **$2,332.36 per provider-day**. Across the five primary PPO
seeds, mean reward was **$2,305.30 per day**, approximately 1.2% lower. PPO
reduced overtime relative to always-overbooking, but did not outperform the
simpler heuristic.

This negative result is retained. Under the current state, reward, demand
process, and cost assumptions, reinforcement learning did not provide enough
benefit to justify its additional complexity.

## Experimental design

The raw appointment data were cleaned using explicit date and age checks. The
no-show model used chronological training, calibration, and held-out test
slices. XGBoost was compared with logistic regression, and raw and calibrated
probabilities were evaluated separately.

The environment represents one provider-day with 16 appointment slots, a slot
capacity of two, 20 incoming requests, and a daily double-booking cap. The
reward is:

```text
visit value × patients seen
- overtime cost × slots with two shows
- decline cost × declined requests
```

The primary PPO used action masking. The research ablation suite contained 15
models: five raw-probability masked PPO seeds, five calibrated-probability
unmasked PPO seeds with an explicit invalid-action penalty, and five calibrated
masked PPO seeds with `gamma=1.0`.

Every model was evaluated on the same chronological evaluation pool and the
same paired simulated days. Each policy received 4,000 simulated evaluation
days generated from three held-out calendar dates, 2016-06-06 through
2016-06-08. These are repeated simulations, not 4,000 real clinic days.

## Primary policy results

| Policy | Mean net reward/day | 95% bootstrap interval | Mean overtime/day | Mean double bookings/day |
|---|---:|---:|---:|---:|
| Expected-load | $2,332.36 | $2,325.57–$2,339.39 | 2.10 | 4.06 |
| Primary PPO, mean across five seeds | $2,305.30 | See per-seed results | 2.57 | 4.00 |
| Threshold | $2,297.01 | $2,289.92–$2,304.36 | 2.69 | 4.00 |
| Always-overbook | $2,136.71 | $2,130.59–$2,142.83 | 5.37 | 8.00 |
| Single-book | $1,885.75 | $1,878.66–$1,892.95 | 0.00 | 0.00 |

All primary PPO seeds were significantly below expected-load in paired Wilcoxon
comparisons. The mean primary PPO policy was approximately $169/day better than
always-overbook and approximately $420/day better than single-book.

## Ablation results

| Variant | Mean across five seeds | Best seed |
|---|---:|---:|
| Calibrated masked PPO, gamma=1.0 | $2,305.19 | $2,318.13 |
| Raw masked PPO | $2,303.84 | $2,316.00 |
| Calibrated unmasked PPO | $2,303.96 | $2,322.20 |

The unmasked models generated very few invalid actions. This shows that the
agent learned the action structure under the explicit penalty, not that action
masking is universally unnecessary.

## Calibration findings

The held-out XGBoost model achieved ROC-AUC 0.7322 and PR-AUC 0.3424. Raw
probabilities had Brier score 0.13647 and ECE 0.03155. Sigmoid calibration
preserved ranking metrics but produced Brier score 0.13675 and ECE 0.03405.

Calibration therefore did not improve formal probability-quality metrics on
this split. On the same 4,000 paired simulated days, calibration increased
expected-load reward by $7.7525/day with a paired Wilcoxon p-value of 3.75e-30.
The median daily difference was $0, so the statistically detectable effect is
small and concentrated in a subset of days.

## Fairness, robustness, and limitations

The fairness audit measured double-booking exposure by age band and scholarship
indicator. Always-overbooking was the most uniform because it ignores risk.
Expected-load maximized reward but produced less uniform exposure: 18–34
patients were double-booked at approximately 1.37x the overall rate,
scholarship patients at approximately 1.31x, and patients 65+ at approximately
0.49x. PPO was closer to uniform exposure, approximately 0.69–1.22x across the
reported groups, at an estimated reward cost of about $27/day (1.2%). This is
an exposure audit only. It does not measure equal service outcomes, access,
wait times, clinical priority, or disparate impact.

The sensitivity analysis varied overtime cost from 30 to 150 and base no-show
rate from 0.10 to 0.30. Expected-load remained the strongest tested baseline
across the grid.

The project is an offline simulation, not a deployment or causal experiment.
The request process, capacity, visit value, overtime cost, and decline cost are
assumptions. The simulator does not model cancellations, rebooking, wait
times, appointment priority, provider heterogeneity, or staffing constraints.
The historical no-show label may also reflect the original clinic's policies.

## Final conclusion

With 20 requests, 16 slots, and a cap of 8 double-bookings, policies that
accept all requests earn the same visit revenue on each realized day. The core
problem is therefore how patients are paired into double-booked slots. The
threshold grid selected tau=0.0, a boundary solution equivalent to accepting
everything, so it should not be presented as evidence of a useful risk
threshold.

The project is complete as an offline research prototype. Expected-load is the
reward-maximizing policy under the current simulator, while PPO offers a more
balanced exposure profile at a small reward cost. These results do not justify
claiming that reinforcement learning is superior.

The appropriate production next step is validation with real scheduling logs,
reviewed cost assumptions, shadow-mode testing, operational guardrails, and a
pre-registered policy evaluation plan.
