import numpy as np
import pandas as pd
from slotrl.env import EnvConfig, SlotEnv
from slotrl.policies import make_policy_set
from slotrl.sim import RequestPool

def test_seed_reproducibility_and_policy_validity():
    frame=pd.DataFrame({"appointment_id":range(20),"p_noshow_cal":[.2]*20,"no_show":[0,1]*10,"age_band":["35-49"]*20,"scholarship":[0]*20})
    p=RequestPool(frame); cfg=EnvConfig(n_slots=4,requests_per_day=6,double_booking_cap=2)
    e1=SlotEnv(p,cfg); e2=SlotEnv(p,cfg); o1,_=e1.reset(seed=123); o2,_=e2.reset(seed=123); assert np.array_equal(o1,o2)
    for name,fn in make_policy_set(n_slots=cfg.n_slots).items():
        env=SlotEnv(p,cfg); obs,_=env.reset(seed=7)
        while True:
            mask=env.action_masks(); action=fn(obs,mask); assert mask[action]; obs,reward,terminated,_,info=env.step(action)
            if terminated: assert info["reward_dollars"] <= cfg.visit_value*cfg.requests_per_day; break
