import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "ml"
    / "data"
    / "fitness"
    / "fitness_dataset.csv"
)

MODEL_DIR = BASE_DIR / "ml" / "saved_models"

MODEL_PATH = MODEL_DIR / "fitness_level_model.pkl"


# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)


# ==========================================
# 3. FEATURES AND TARGET
# ==========================================

features = [
    "age",
    "gender",
    "height_cm",
    "weight_kg",
    "bmi",
    "activity_level",
    "exercise_frequency",
    "exercise_duration",
    "resting_heart_rate"
]

target = "fitness_level"

X = df[features]
y = df[target]


print("\nFeatures:")
print(features)

print("\nTarget:")
print(target)


# ==========================================
# 4. TARGET DISTRIBUTION
# ==========================================

print("\nFitness Level Distribution:")
print(y.value_counts())


# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ==========================================
# 6. PREPROCESSING
# ==========================================

numerical_features = [
    "age",
    "height_cm",
    "weight_kg",
    "bmi",
    "exercise_frequency",
    "exercise_duration",
    "resting_heart_rate"
]

categorical_features = [
    "gender",
    "activity_level"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ==========================================
# 7. LOGISTIC REGRESSION
# ==========================================

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)


logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)

logistic_precision = precision_score(
    y_test,
    logistic_pred,
    average="macro",
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    logistic_pred,
    average="macro",
    zero_division=0
)

logistic_f1 = f1_score(
    y_test,
    logistic_pred,
    average="macro",
    zero_division=0
)


print("\n==========================================")
print("LOGISTIC REGRESSION PERFORMANCE")
print("==========================================")

print("Accuracy :", round(logistic_accuracy, 4))
print("Precision:", round(logistic_precision, 4))
print("Recall   :", round(logistic_recall, 4))
print("Macro F1 :", round(logistic_f1, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        logistic_pred,
        zero_division=0
    )
)


# ==========================================
# 8. DECISION TREE
# ==========================================

tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=8,
                random_state=42
            )
        )
    ]
)

tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)


tree_accuracy = accuracy_score(
    y_test,
    tree_pred
)

tree_precision = precision_score(
    y_test,
    tree_pred,
    average="macro",
    zero_division=0
)

tree_recall = recall_score(
    y_test,
    tree_pred,
    average="macro",
    zero_division=0
)

tree_f1 = f1_score(
    y_test,
    tree_pred,
    average="macro",
    zero_division=0
)


print("\n==========================================")
print("DECISION TREE PERFORMANCE")
print("==========================================")

print("Accuracy :", round(tree_accuracy, 4))
print("Precision:", round(tree_precision, 4))
print("Recall   :", round(tree_recall, 4))
print("Macro F1 :", round(tree_f1, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        tree_pred,
        zero_division=0
    )
)


# ==========================================
# 9. RANDOM FOREST
# ==========================================

forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

forest_model.fit(X_train, y_train)

forest_pred = forest_model.predict(X_test)


forest_accuracy = accuracy_score(
    y_test,
    forest_pred
)

forest_precision = precision_score(
    y_test,
    forest_pred,
    average="macro",
    zero_division=0
)

forest_recall = recall_score(
    y_test,
    forest_pred,
    average="macro",
    zero_division=0
)

forest_f1 = f1_score(
    y_test,
    forest_pred,
    average="macro",
    zero_division=0
)


print("\n==========================================")
print("RANDOM FOREST PERFORMANCE")
print("==========================================")

print("Accuracy :", round(forest_accuracy, 4))
print("Precision:", round(forest_precision, 4))
print("Recall   :", round(forest_recall, 4))
print("Macro F1 :", round(forest_f1, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        forest_pred,
        zero_division=0
    )
)


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

print("\n==========================================")
print("RANDOM FOREST CONFUSION MATRIX")
print("==========================================")

print(
    confusion_matrix(
        y_test,
        forest_pred
    )
)


# ==========================================
# 11. MODEL COMPARISON
# ==========================================

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        logistic_accuracy,
        tree_accuracy,
        forest_accuracy
    ],
    "Precision": [
        logistic_precision,
        tree_precision,
        forest_precision
    ],
    "Recall": [
        logistic_recall,
        tree_recall,
        forest_recall
    ],
    "Macro_F1": [
        logistic_f1,
        tree_f1,
        forest_f1
    ]
})


print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

print(
    results.to_string(
        index=False
    )
)


# ==========================================
# 12. SELECT BEST MODEL
# ==========================================

best_model_name = results.loc[
    results["Macro_F1"].idxmax(),
    "Model"
]

print("\nBest Model based on Macro F1:")
print(best_model_name)


# ==========================================
# 13. SELECT FINAL MODEL
# ==========================================

if best_model_name == "Logistic Regression":

    final_model = logistic_model

elif best_model_name == "Decision Tree":

    final_model = tree_model

else:

    final_model = forest_model


# ==========================================
# 14. SAVE FINAL MODEL
# ==========================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    final_model,
    MODEL_PATH
)

print("\n==========================================")
print("FINAL MODEL SAVED")
print("==========================================")

print("Model:", MODEL_PATH)