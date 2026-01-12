# src/data/preprocess.py

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


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

# --------------------
# Path handling
# --------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_lead_conversion_2000.csv"



# --------------------
# Functions
# --------------------

def load_data(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing pipeline"""

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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor


def split_features_target(df: pd.DataFrame):
    """Split features and target"""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def train_val_test_split(
    X,
    y,
    test_size=0.2,
    val_size=0.1,
    random_state=42
):
    """Create train/val/test splits"""

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size + val_size,
        stratify=y,
        random_state=random_state
    )

    val_ratio = val_size / (test_size + val_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - val_ratio,
        stratify=y_temp,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Example usage (local testing)

    
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)

    preprocessor = build_preprocessor()

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    print("Preprocessing complete!")
    print(f"#Training samples: {X_train_processed.shape[0]}")
    print(f"#Validation samples: {X_val_processed.shape[0]}")
    print(f"#Test samples: {X_test_processed.shape[0]}")
    print("Train shape:", X_train_processed.shape)
