# src/model_building.py

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier


# ----------------------------
# Paths & logging
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = LOG_DIR / "model_building.log"
file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Avoid duplicate handlers if re-imported
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ----------------------------
# Config helpers
# ----------------------------
def load_params(params_path: Path) -> Dict[str, Any]:
    """Load parameters from YAML."""
    with params_path.open("r", encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}
    logger.debug("Parameters retrieved from %s", params_path)
    return params


def resolve_config() -> Dict[str, Any]:
    """
    Build a configuration with defaults, optionally overridden by params.yaml.
    """
    cfg: Dict[str, Any] = {}

    # Defaults
    cfg.setdefault("paths", {})
    cfg["paths"].setdefault("train_tfidf", str(REPO_ROOT / "data" / "processed" / "train_tfidf.csv"))
    cfg["paths"].setdefault("models_dir", str(REPO_ROOT / "models"))
    cfg.setdefault("model_building", {})
    cfg["model_building"].setdefault("n_estimators", 200)
    cfg["model_building"].setdefault("random_state", 42)

    # params.yaml overrides (if present)
    params_file = REPO_ROOT / "params.yaml"
    if params_file.exists():
        try:
            params = load_params(params_file)
            mb = params.get("model_building", {})
            for k in ["n_estimators", "random_state"]:
                if k in mb and mb[k] is not None:
                    cfg["model_building"][k] = mb[k]

            p = params.get("paths", {})
            for k in ["train_tfidf", "models_dir"]:
                if k in p and p[k]:
                    cfg["paths"][k] = p[k]

            logger.debug("Config resolved from params.yaml with overrides.")
        except Exception as e:
            logger.warning("Unable to parse params.yaml; using defaults. Error: %s", e)
    else:
        logger.warning("params.yaml not found at %s; using default configuration.", params_file)

    return cfg


# ----------------------------
# Data & model helpers
# ----------------------------
def load_data(file_path: Path) -> pd.DataFrame:
    """Load TF-IDF CSV and ensure label column exists."""
    df = pd.read_csv(file_path, encoding="utf-8")
    logger.debug("Data loaded from %s with shape %s", file_path, df.shape)

    # Ensure a label column exists; prefer 'label', else last column
    label_col = "label" if "label" in df.columns else df.columns[-1]
    if label_col not in df.columns:
        raise KeyError(f"Label column not found in {file_path}. Columns: {list(df.columns)}")

    # Split features/labels
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Make sure features are numeric and handle NaNs
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = y.values

    logger.debug("Features dtype coerced to numeric. Final shapes -> X: %s, y: %s", X.shape, y.shape)
    return X.values, y


def train_model(X_train: np.ndarray, y_train: np.ndarray, params: Dict[str, Any]) -> RandomForestClassifier:
    """Train RandomForest model with basic validation."""
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("The number of samples in X_train and y_train must be the same.")
    if X_train.size == 0:
        raise ValueError("Training features are empty.")
    if y_train.size == 0:
        raise ValueError("Training labels are empty.")

    n_estimators = int(params.get("n_estimators", 200))
    random_state = int(params.get("random_state", 42))

    logger.debug("Initializing RandomForest with n_estimators=%s, random_state=%s", n_estimators, random_state)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    logger.debug("Model training started with %d samples", X_train.shape[0])
    clf.fit(X_train, y_train)
    logger.debug("Model training completed")
    return clf


def save_model(model: RandomForestClassifier, file_path: Path) -> None:
    """Persist the trained model to disk."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(model, f)
    logger.debug("Model saved to %s", file_path)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    try:
        cfg = resolve_config()
        train_fp = Path(cfg["paths"]["train_tfidf"])
        models_dir = Path(cfg["paths"]["models_dir"])

        if not train_fp.exists():
            raise FileNotFoundError(
                f"Training file not found at {train_fp}. "
                f"Make sure your feature engineering step produced train_tfidf.csv."
            )

        X_train, y_train = load_data(train_fp)
        clf = train_model(X_train, y_train, cfg["model_building"])

        model_path = models_dir / "model.pkl"
        save_model(clf, model_path)

        logger.info("Model building completed successfully. Saved to %s", model_path)

    except Exception as e:
        logger.error("Failed to complete the model building process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
