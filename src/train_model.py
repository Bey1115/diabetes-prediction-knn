"""
Train and evaluate KNN models for diabetes classification.

Usage:
    python src/train_model.py
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Dataset_Diabetes_Prediction.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET = "Diagnosis"


def evaluate(model, X_test, y_test, name):
    pred = model.predict(X_test)
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
    }
    print(f"\n{name}")
    print("-" * len(name))
    print(classification_report(
        y_test, pred,
        target_names=["No Diabetes", "Diabetes"],
        zero_division=0
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred))
    return metrics


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place Dataset_Diabetes_Prediction.csv in the data/ folder."
        )

    df = pd.read_csv(DATA_PATH)

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])

    baseline_search = GridSearchCV(
        baseline,
        {"knn__n_neighbors": range(1, 31)},
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )
    baseline_search.fit(X_train, y_train)

    f1_model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])

    f1_search = GridSearchCV(
        f1_model,
        {
            "knn__n_neighbors": range(1, 31),
            "knn__weights": ["uniform", "distance"],
        },
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )
    f1_search.fit(X_train, y_train)

    results = [
        evaluate(
            baseline_search.best_estimator_,
            X_test, y_test,
            "Baseline KNN"
        ),
        evaluate(
            f1_search.best_estimator_,
            X_test, y_test,
            "F1-Tuned KNN"
        ),
    ]

    results_df = pd.DataFrame(results)
    print("\nModel comparison:")
    print(results_df.round(4).to_string(index=False))

    joblib.dump(
        f1_search.best_estimator_,
        MODEL_DIR / "knn_diabetes_model.joblib"
    )

    results_df.to_csv(
        ROOT / "reports" / "model_comparison.csv",
        index=False
    )

    print("\nSaved model:", MODEL_DIR / "knn_diabetes_model.joblib")


if __name__ == "__main__":
    main()
