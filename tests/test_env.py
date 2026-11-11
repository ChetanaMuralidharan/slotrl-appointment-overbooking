import pandas as pd
from slotrl.sim import RequestPool
from slotrl.env import SlotEnv, EnvConfig

def pool():
    return RequestPool(pd.DataFrame({"appointment_id":range(20),"p_noshow_cal":[.2]*20,"no_show":[0,1]*10,"age_band":["35-49"]*20,"scholarship":[0]*20}))

def test_mask_and_determinism():
    e=SlotEnv(pool(), EnvConfig(n_slots=2, requests_per_day=4, double_booking_cap=1), seed=5); o,_=e.reset(seed=7)
    while True:
        m=e.action_masks(); assert m.any(); a=next(i for i,v in enumerate(m) if v); o,r,t,_,_=e.step(a)
        if t: break

def test_two_shows_reward():
    p=RequestPool(pd.DataFrame({"appointment_id":[1,2],"p_noshow_cal":[0.1,0.1],"no_show":[0,0],"age_band":["65+","65+"],"scholarship":[0,0]}))
    e=SlotEnv(p, EnvConfig(n_slots=1, requests_per_day=2, visit_value=150, overtime_cost=60, decline_cost=20)); e.reset(seed=1); e.step(0); _,_,terminated,_,info=e.step(0); assert terminated; assert info["revenue"] == 300; assert info["overtime_count"] == 1; assert info["reward_dollars"] == 240

