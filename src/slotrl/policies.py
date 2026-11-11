from __future__ import annotations
import numpy as np

def _valid(mask): return np.flatnonzero(mask[:-1])
def single_book(obs, mask, n_slots=16):
    valid=_valid(mask); empty=[i for i in valid if obs[2*i] == 0]
    return int(empty[0]) if empty else n_slots
def always_overbook(obs, mask, n_slots=16):
    valid=_valid(mask); return int(valid[0]) if len(valid) else n_slots
def threshold(obs, mask, tau=0.35, n_slots=16):
    valid=_valid(mask); empty=[i for i in valid if obs[2*i] == 0]
    if empty: return int(empty[0])
    candidates=[i for i in valid if obs[2*i] > 0]
    return int(candidates[0]) if candidates and obs[2*n_slots] >= tau else n_slots
def expected_load(obs, mask, max_load=1.05, n_slots=16):
    valid=_valid(mask); p_show=1-obs[2*n_slots]; scored=[]
    for i in valid:
        load=obs[2*i+1]*2 + p_show
        scored.append((abs(load-1), load, i))
    if not scored: return n_slots
    _,load,i=min(scored)
    return int(i) if load <= max_load else n_slots

POLICIES={"single_book":single_book,"always_overbook":always_overbook,"threshold":threshold,"expected_load":expected_load}

def make_policy_set(tau=0.35, max_load=1.05, n_slots=16):
    """Create policies bound to the configured environment action layout."""
    return {
        "single_book": lambda obs, mask: single_book(obs, mask, n_slots=n_slots),
        "always_overbook": lambda obs, mask: always_overbook(obs, mask, n_slots=n_slots),
        "threshold": lambda obs, mask: threshold(obs, mask, tau=tau, n_slots=n_slots),
        "expected_load": lambda obs, mask: expected_load(obs, mask, max_load=max_load, n_slots=n_slots),
    }

