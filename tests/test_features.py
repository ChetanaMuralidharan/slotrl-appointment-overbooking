import pandas as pd
from slotrl.features import make_features

def test_same_day_patient_history_is_not_label_order_dependent():
    df=pd.DataFrame({
        "patient_id":[1,1,1],"appointment_id":[1,2,3],
        "gender":["F"]*3,"scheduled_day":pd.to_datetime(["2016-01-01"]*3,utc=True),
        "appointment_day":pd.to_datetime(["2016-01-10","2016-01-10","2016-01-20"],utc=True),
        "age":[30]*3,"neighbourhood":["A"]*3,"scholarship":[0]*3,
        "hypertension":[0]*3,"diabetes":[0]*3,"alcoholism":[0]*3,"handicap":[0]*3,"sms_received":[0]*3,
        "no_show":[1,0,0],
    })
    out=make_features(df)
    same=out[out.appointment_day==pd.Timestamp("2016-01-10",tz="UTC")]
    assert same.prior_appointments.tolist()==[0,0]
    assert same.prior_no_shows.tolist()==[0,0]
    later=out[out.appointment_day==pd.Timestamp("2016-01-20",tz="UTC")].iloc[0]
    assert later.prior_appointments==2 and later.prior_no_shows==1


def test_neighbourhood_frequency_uses_strictly_earlier_dates():
    df = pd.DataFrame({
        "patient_id": [3, 1, 2], "appointment_id": [3, 1, 2],
        "gender": ["F"] * 3,
        "scheduled_day": pd.to_datetime(["2016-01-01"] * 3, utc=True),
        "appointment_day": pd.to_datetime(["2016-01-12", "2016-01-10", "2016-01-11"], utc=True),
        "age": [30] * 3, "neighbourhood": ["A", "A", "B"], "scholarship": [0] * 3,
        "hypertension": [0] * 3, "diabetes": [0] * 3, "alcoholism": [0] * 3,
        "handicap": [0] * 3, "sms_received": [0] * 3, "no_show": [0] * 3,
    })
    out = make_features(df).sort_values("appointment_day")
    assert out.iloc[0].neighbourhood_frequency == 0.0
    assert out.iloc[1].neighbourhood_frequency == 0.0
    assert out.iloc[2].neighbourhood_frequency == 0.5
