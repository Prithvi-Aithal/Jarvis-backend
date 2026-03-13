import joblib
import numpy as np
import os

model = joblib.load(os.path.join(os.path.dirname(__file__), "stress_model.pkl"))

def predict_stress_from_tracker(
    screen_time,
    continuous_usage,
    night_usage,
    app_switches,
    breaks,
    productive_ratio
):

    features = np.array([[ 
        screen_time,
        continuous_usage,
        night_usage,
        app_switches,
        breaks,
        productive_ratio
    ]])

    prediction = model.predict(features)[0]

    mapping = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    return mapping.get(prediction, "Medium")