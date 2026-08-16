import joblib 
import pandas as pd
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "ml"
    / "saved_models"
    / "calorie_model.pkl"
)
model=joblib.load(MODEL_PATH)
print("Model loaded successfully.")
def predict_calories(
    age,
    gender,
    height_cm,
    weight_kg,
    activity_level
):

    input_data = pd.DataFrame([
        {
            "Age": age,
            "Gender": gender,
            "Height_cm": height_cm,
            "Weight_kg": weight_kg,
            "Activity_Level": activity_level
        }
    ])

    prediction = model.predict(input_data)

    return float(prediction[0])
if __name__ == "__main__":

    result = predict_calories(
        age=19,
        gender="Male",
        height_cm=182,
        weight_kg=85,
        activity_level="Moderate"
    )

    print(f"Predicted Daily Calories: {result:.2f} kcal")