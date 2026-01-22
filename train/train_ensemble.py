"""
SEPE - Student Enrollment Prediction Engine
Training Script with Ensemble & Boosting Models

Models:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

Features:
✔ Pipelines
✔ Train/Val/Test split
✔ Argparse
✔ Logging
✔ Model comparison table

Running:
From project root (sepe/): uv run python -m train.train --data_path <file_path>
from train dir (sepe/train/): uv run python train.py --data_path <file_path>

"""

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

# from xgboost import XGBClassifier


# =====================================================
# Path handling (root or train/)
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


# ======================
# Feature Definitions
# ======================

NUMERIC_FEATURES = [
    "child_age",
    "response_time_hours",
    "demo_feedback_score",
    "follow_up_calls"
]

CATEGORICAL_FEATURES = [
    "parent_tech_background",
    "demo_attended",
    "prior_coding_experience",
    "course_interest",
    "ad_source",
    "Living_city"
]

TARGET = "conversion"


# ======================
# Logging Setup
# ======================

def setup_logging():
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / "training.log"),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger("SEPE-Training")


# ======================
# Argument Parser
# ======================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SEPE models including XGBoost"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to CSV dataset"
    )

    return parser.parse_args()


# ======================
# Preprocessor
# ======================

def build_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
        ]
    )


# ======================
# Training Logic
# ======================

def train(data_path: str, logger):
    logger.info("🚀 Training started")

    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    logger.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # ----------------------
    # Train / Val / Test split
    # ----------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    logger.info(
        f"Split sizes → Train: {len(X_train)}, "
        f"Val: {len(X_val)}, Test: {len(X_test)}"
    )

    preprocessor = build_preprocessor()

    # ======================
    # Models
    # ======================

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators= 300, #200
            max_depth=8, #10
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, #150
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=42
        )

        # "XGBoost": XGBClassifier(
        #     n_estimators=200,
        #     learning_rate=0.05,
        #     max_depth=4,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     eval_metric="logloss",
        #     random_state=42,
        #     use_label_encoder=False
        # )
    }

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    # Store results for table
    results = []

    # ======================
    # Train & Evaluate
    # ======================

    for name, model in models.items():
        logger.info(f"🔹 Training model: {name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)

        val_probs = pipeline.predict_proba(X_val)[:, 1]
        val_preds = pipeline.predict(X_val)

        roc_auc = roc_auc_score(y_val, val_probs)
        acc = accuracy_score(y_val, val_preds)

        logger.info(
            f"{name} | Validation ROC-AUC: {roc_auc:.4f}, Accuracy: {acc:.4f}"
        )

        # Save model
        model_file = name.lower().replace(" ", "_") + "_pipeline.joblib"
        model_path = models_dir / model_file
        joblib.dump(pipeline, model_path)

        logger.info(f"✅ Saved model: {model_path}")

        # Store metrics
        results.append({
            "Model": name,
            "Validation ROC-AUC": round(roc_auc, 4),
            "Validation Accuracy": round(acc, 4)
        })

    # ======================
    # Results Table
    # ======================

    results_df = pd.DataFrame(results).sort_values(
        by="Validation ROC-AUC", ascending=False
    )

    logger.info("\n📊 MODEL PERFORMANCE SUMMARY (Validation Set)")
    logger.info("\n" + results_df.to_string(index=False))

    # print("\nMODEL PERFORMANCE SUMMARY (Validation Set)")
    # print(results_df.to_string(index=False))

    logger.info("🎉 Training completed successfully")


# ======================
# Entry Point
# ======================

def main():
    args = parse_args()
    logger = setup_logging()

    try:
        train(args.data_path, logger)
    except Exception:
        logger.exception("❌ Training failed")
        raise


if __name__ == "__main__":
    main()
