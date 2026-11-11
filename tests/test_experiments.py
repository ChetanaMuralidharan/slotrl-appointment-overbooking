import pandas as pd
from slotrl.env import EnvConfig, SlotEnv
from slotrl.experiments import default_experiment_specs
from slotrl.sim import RequestPool


def test_experiment_matrix_is_declared_and_distinct():
    cfg = {"training": {"gamma": 0.99, "ent_coef": 0.01, "learning_rate": 3e-4}}
    specs = default_experiment_specs(cfg)
    assert {s.name for s in specs} == {"raw_masked_ppo", "calibrated_unmasked_ppo", "calibrated_gamma_1_0", "calibrated_no_entropy"}
    assert any(s.algorithm == "ppo" and s.allow_invalid_actions for s in specs)
    assert any(s.probability_column == "p_noshow_raw" for s in specs)


def test_experiment_selection_is_configurable():
    cfg = {"training": {"gamma": 0.99, "ent_coef": 0.01, "learning_rate": 3e-4}, "experiments": {"enabled": ["raw_masked_ppo"]}}
    assert [s.name for s in default_experiment_specs(cfg)] == ["raw_masked_ppo"]


def test_invalid_actions_are_penalized_in_unmasked_mode():
    frame = pd.DataFrame({"appointment_id": range(10), "p_noshow_cal": [.2] * 10, "no_show": [0] * 10, "age_band": ["35-49"] * 10, "scholarship": [0] * 10})
    env = SlotEnv(RequestPool(frame), EnvConfig(n_slots=2, slot_capacity=1, requests_per_day=2, double_booking_cap=1, invalid_action_penalty=50.0), allow_invalid_actions=True)
    env.reset(seed=1); env.step(0); _, _, terminated, _, info = env.step(0)
    assert terminated and info["invalid_actions"] == 1 and info["reward_dollars"] == 80.0
