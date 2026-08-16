import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "ml"
    / "data"
    / "calorie_health"
    / "healthy_diet_calorie_intake.csv"
)

MODEL_DIR = BASE_DIR / "ml" / "saved_models"

MODEL_PATH = MODEL_DIR / "calorie_model.pkl"

df=pd.read_csv(DATA_PATH)
features=[
    "Age",
    "Gender",
    "Height_cm",
    "Weight_kg",
    "Activity_Level"
]
target="Daily_Calorie_Requirement"

X=df[features]
y=df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

numerical_features=["Age","Height_cm","Weight_kg"]
categorical_features=["Gender","Activity_Level"]
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)
model=Pipeline(
    steps=[
        ("preprocesser",preprocessor),
        ("regressor",LinearRegression())
    ]
)
model.fit(X_train,y_train)
# print("\nModel Training Completed.")

y_pred=model.predict(X_test)
# print("\nFirst 10 Prediction.")
# print(y_pred[:10])
mae=mean_absolute_error(y_test,y_pred)
rmse=mean_squared_error(y_test,y_pred)**0.5
r2 = r2_score(y_test,y_pred)
print("\nModel Performance:")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))
from sklearn.tree import DecisionTreeRegressor
tree_model=Pipeline(
    steps=[
        ("preprocessing",preprocessor),
        ("regressor",DecisionTreeRegressor(random_state=42,max_depth=8))
    ]
)
tree_model.fit(X_train,y_train)
tree_pred=tree_model.predict(X_test)
tree_mae = mean_absolute_error(y_test, tree_pred)
tree_rmse = mean_squared_error(y_test, tree_pred) ** 0.5
tree_r2 = r2_score(y_test, tree_pred)

print("\nDecision Tree Performance:")
print("MAE :", round(tree_mae, 2))
print("RMSE:", round(tree_rmse, 2))
print("R²  :", round(tree_r2, 4))
from sklearn.ensemble import RandomForestRegressor
forest_model=Pipeline(
    steps=[
        ("preprocessor",preprocessor),
        ("regressor",RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=10,
            n_jobs=-1))
    ]
)
forest_model.fit(X_train, y_train)

forest_pred = forest_model.predict(X_test)

forest_mae = mean_absolute_error(y_test, forest_pred)
forest_rmse = mean_squared_error(y_test, forest_pred) ** 0.5
forest_r2 = r2_score(y_test, forest_pred)

print("\nRandom Forest Performance:")
print("MAE :", round(forest_mae, 2))
print("RMSE:", round(forest_rmse, 2))
print("R²  :", round(forest_r2, 4))
features_bmi=[
    "Age",
    "Gender",
    "BMI",
    "Height_cm",
    "Weight_kg",
    "Activity_Level"
]
X_bmi = df[features_bmi]
y_bmi = df[target]
X_train_bmi, X_test_bmi, y_train_bmi, y_test_bmi = train_test_split(
    X_bmi,
    y_bmi,
    test_size=0.20,
    random_state=42
)
numerical_features_bmi = [
    "Age",
    "Height_cm",
    "Weight_kg",
    "BMI"
]

categorical_features_bmi = [
    "Gender",
    "Activity_Level"
]
preprocessor_bmi = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features_bmi
        )
    ],
    remainder="passthrough"
)
forest_bmi_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor_bmi),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=10,
            n_jobs=-1
        ))
    ]
)
forest_bmi_model.fit(X_train_bmi, y_train_bmi)
forest_bmi_pred = forest_bmi_model.predict(X_test_bmi)
forest_bmi_mae = mean_absolute_error(
    y_test_bmi,
    forest_bmi_pred
)

forest_bmi_rmse = mean_squared_error(
    y_test_bmi,
    forest_bmi_pred
) ** 0.5

forest_bmi_r2 = r2_score(
    y_test_bmi,
    forest_bmi_pred
)

print("\nRandom Forest WITH BMI:")
print("MAE :", round(forest_bmi_mae, 2))
print("RMSE:", round(forest_bmi_rmse, 2))
print("R²  :", round(forest_bmi_r2, 4))
print("\n===================================")
print("FINAL MODEL: RANDOM FOREST")
print("BMI excluded because it did not improve performance.")
print("===================================")
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    forest_model,
    MODEL_PATH
)

print("\nFinal model saved successfully!")
print("Model location:", MODEL_PATH)