from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .sim import Request, RequestPool

@dataclass(frozen=True)
class EnvConfig:
    n_slots: int = 16; slot_capacity: int = 2; requests_per_day: int = 20; double_booking_cap: int = 8
    visit_value: float = 150.0; overtime_cost: float = 60.0; decline_cost: float = 20.0; reward_scale: float = 1000.0; invalid_action_penalty: float = 50.0

class SlotEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, pool: RequestPool, config: EnvConfig | None = None, seed: int = 0, base_rate: float | None = None, allow_invalid_actions: bool = False):
        super().__init__(); self.pool = pool; self.cfg = config or EnvConfig(); self._seed = seed; self.base_rate = base_rate; self.allow_invalid_actions = allow_invalid_actions
        n = self.cfg.n_slots
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2*n + 3,), dtype=np.float32)
        self.action_space = spaces.Discrete(n + 1); self.slots: list[list[Request]] = []; self.requests: list[Request] = []
        self.index = 0; self.declines = 0; self.double_bookings = 0; self.invalid_actions = 0; self._terminated = False
    def _obs(self):
        values=[]
        for slot in self.slots:
            values += [len(slot)/self.cfg.slot_capacity, min(1.0, sum(1-r.p_noshow for r in slot)/self.cfg.slot_capacity)]
        req = self.requests[self.index] if self.index < len(self.requests) else None
        return np.asarray(values + [req.p_noshow if req else 0.0, max(0, len(self.requests)-self.index)/self.cfg.requests_per_day, self.double_bookings/max(1,self.cfg.double_booking_cap)], dtype=np.float32)
    def action_masks(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        if self._terminated: return mask
        for i, slot in enumerate(self.slots):
            if len(slot) < self.cfg.slot_capacity and not (len(slot) == 1 and self.double_bookings >= self.cfg.double_booking_cap): mask[i] = True
        mask[-1] = True
        return mask
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); actual_seed = self._seed if seed is None else seed
        self.requests = self.pool.sample_day(self.cfg.requests_per_day, seed=actual_seed, base_rate=self.base_rate)
        self.slots = [[] for _ in range(self.cfg.n_slots)]; self.index=0; self.declines=0; self.double_bookings=0; self.invalid_actions=0; self._terminated=False
        return self._obs(), {"action_mask": self.action_masks()}
    def _settle(self):
        seen=sum(sum(r.shows for r in slot) for slot in self.slots); overtime=sum(sum(r.shows for r in slot)==2 for slot in self.slots)
        revenue=self.cfg.visit_value*seen; reward=revenue-self.cfg.overtime_cost*overtime-self.cfg.decline_cost*self.declines-self.cfg.invalid_action_penalty*self.invalid_actions
        exposure={"age_band": {}, "scholarship": {}}
        for attr in exposure:
            for slot in self.slots:
                is_double=len(slot)==2
                for r in slot:
                    key=str(getattr(r,attr)); exposure[attr].setdefault(key,[0,0]); exposure[attr][key][0]+=1; exposure[attr][key][1]+=int(is_double)
        info={"revenue":revenue,"overtime_count":overtime,"idle_slots":sum(sum(r.shows for r in slot)==0 for slot in self.slots),"declines":self.declines,"invalid_actions":self.invalid_actions,"double_bookings":self.double_bookings,"exposure_counts":exposure,"seen":seen,"reward_dollars":reward}
        return reward, info
    def step(self, action):
        if self._terminated: raise RuntimeError("step() called after episode termination; call reset()")
        action=int(action)
        if action < 0 or action >= self.action_space.n: raise ValueError(f"Action outside action space: {action}")
        valid=bool(self.action_masks()[action])
        if not valid and not self.allow_invalid_actions: raise ValueError(f"Invalid action {action}")
        req=self.requests[self.index]
        if not valid:
            self.invalid_actions += 1; self.declines += 1
        elif action == self.cfg.n_slots: self.declines += 1
        else:
            if len(self.slots[action]) == 1: self.double_bookings += 1
            self.slots[action].append(req)
        self.index += 1; terminated=self.index >= len(self.requests); self._terminated=terminated
        if terminated:
            dollars, info=self._settle(); return self._obs(), float(dollars/self.cfg.reward_scale), True, False, info
        return self._obs(), 0.0, False, False, {"action_mask": self.action_masks()}

