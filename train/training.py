

import logging
import joblib
from pathlib import Path
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score







# ======================
# Import bootstrap
# ======================
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]  # sepe/

# Add project root to PYTHONPATH if not already there
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from data_eda_preprocess.data_preprocessing import (
    load_data,
    split_features_target,
    build_preprocessor,
    train_val_test_split
)

# ======================
# Path Configuration
# ======================

DATA_PATH = PROJECT_ROOT / "data" / "synthetic_lead_conversion_2000.csv"
MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

MODELS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)



# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# DATA_PATH = PROJECT_ROOT / "data" / "synthetic_lead_conversion_2000.csv"
# MODELS_DIR = PROJECT_ROOT / "models"
# LOGS_DIR = PROJECT_ROOT / "logs"

# MODELS_DIR.mkdir(exist_ok=True)
# LOGS_DIR.mkdir(exist_ok=True)


# ======================
# Logging Configuration
# ======================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SEPE-Training")


# ======================
# Training Function
# ======================

def train():
    try:
        logger.info("🚀 Training pipeline started")

        # ----------------------
        # Load data
        # ----------------------
        logger.info("Loading raw dataset")
        df = load_data(DATA_PATH)
        logger.info(f"Dataset loaded with shape: {df.shape}")

        # ----------------------
        # Split features/target
        # ----------------------
        X, y = split_features_target(df)
        logger.info("Features and target separated")
        logger.info(f"Feature shape: {X.shape}, Target shape: {y.shape}")

        # ----------------------
        # Train / Val / Test split
        # ----------------------
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
        logger.info("Data split into train/val/test")
        logger.info(
            f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
        )

        # ----------------------
        # Preprocessing
        # ----------------------
        logger.info("Building preprocessing pipeline")
        preprocessor = build_preprocessor()

        logger.info("Fitting preprocessor on training data")
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)

        logger.info(
            f"Processed shapes -> Train: {X_train_proc.shape}, Val: {X_val_proc.shape}"
        )

        # ----------------------
        # Model training
        # ----------------------
        logger.info("Initializing Logistic Regression model")
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear"
        )

        logger.info("Training model")
        model.fit(X_train_proc, y_train)

        # ----------------------
        # Validation
        # ----------------------
        logger.info("Evaluating model on validation set")
        val_probs = model.predict_proba(X_val_proc)[:, 1]
        val_preds = model.predict(X_val_proc)

        roc_auc = roc_auc_score(y_val, val_probs)
        accuracy = accuracy_score(y_val, val_preds)

        logger.info(f"Validation ROC-AUC: {roc_auc:.4f}")
        logger.info(f"Validation Accuracy: {accuracy:.4f}")

        # ----------------------
        # Save artifacts
        # ----------------------
        Path("models").mkdir(exist_ok=True)

        joblib.dump(preprocessor, "models/preprocessor.joblib")
        joblib.dump(model, "models/logistic_regression.joblib")

        logger.info("Model and preprocessor saved successfully")
        logger.info("✅ Training pipeline completed")

    except Exception as e:
        logger.exception("❌ Training pipeline failed due to an error")
        raise e


# ======================
# Entry Point
# ======================

if __name__ == "__main__":
    train()
