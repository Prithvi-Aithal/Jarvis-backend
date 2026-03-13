import joblib
import numpy as np
import os

# load trained model
model = joblib.load(os.path.join(os.path.dirname(__file__), "stress_model.pkl"))

def predict_stress(screen_time, continuous_usage, night_usage, app_switches, breaks, productive_ratio):
    features = np.array([[screen_time,
                          continuous_usage,
                          night_usage,
                          app_switches,
                          breaks,
                          productive_ratio]])

    prediction = model.predict(features)[0]
    return prediction


# test run
if __name__ == "__main__":
    result = predict_stress(420, 90, 60, 50, 2, 0.3)
    print("Predicted Stress Level:", result)