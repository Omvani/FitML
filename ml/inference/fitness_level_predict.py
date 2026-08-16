import joblib
import pandas as pd

from pathlib import Path


# ==========================================
# 1. MODEL PATH
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "ml"
    / "saved_models"
    / "fitness_level_model.pkl"
)


# ==========================================
# 2. LOAD MODEL
# ==========================================

model = joblib.load(MODEL_PATH)

print("Fitness level model loaded successfully!")


# ==========================================
# 3. CALCULATE BMI
# ==========================================

def calculate_bmi(height_cm, weight_kg):

    height_m = height_cm / 100

    bmi = weight_kg / (height_m ** 2)

    return bmi


# ==========================================
# 4. PREDICT FITNESS LEVEL
# ==========================================

def predict_fitness_level(
    age,
    gender,
    height_cm,
    weight_kg,
    activity_level,
    exercise_frequency,
    exercise_duration,
    resting_heart_rate
):

    # Calculate BMI from height and weight
    bmi = calculate_bmi(
        height_cm,
        weight_kg
    )

    # Create input DataFrame
    input_data = pd.DataFrame([
        {
            "age": age,
            "gender": gender,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "activity_level": activity_level,
            "exercise_frequency": exercise_frequency,
            "exercise_duration": exercise_duration,
            "resting_heart_rate": resting_heart_rate
        }
    ])

    # Make prediction
    prediction = model.predict(input_data)

    return {
        "fitness_level": prediction[0],
        "bmi": round(bmi, 2)
    }


# ==========================================
# 5. TEST
# ==========================================

if __name__ == "__main__":

    result = predict_fitness_level(
        age=22,
        gender="Male",
        height_cm=175,
        weight_kg=70,
        activity_level="Moderate",
        exercise_frequency=4,
        exercise_duration=45,
        resting_heart_rate=65
    )

    print("\nPrediction Result:")
    print("BMI:", result["bmi"])
    print("Fitness Level:", result["fitness_level"])