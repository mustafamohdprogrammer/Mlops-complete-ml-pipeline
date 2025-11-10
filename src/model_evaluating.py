import os
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml
from dvclive import Live

# ------------------------------------------------------
#  Setup paths
# ------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# ------------------------------------------------------
# Logging configuration
# ------------------------------------------------------
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(LOG_DIR / "model_evaluation.log", encoding="utf-8")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ------------------------------------------------------
# Helper functions
# ------------------------------------------------------
def load_params():
    """Load params.yaml if available; else return empty dict."""
    params_path = REPO_ROOT / "params.yaml"
    if not params_path.exists():
        logger.warning("params.yaml not found. Continuing with default settings.")
        return {}

    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f) or {}
            logger.debug("Parameters loaded from params.yaml")
            return params
    except Exception as e:
        logger.error("Error reading params.yaml: %s", e)
        return {}


def load_model(model_path: Path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.debug(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def load_csv(path: Path):
    try:
        df = pd.read_csv(path)
        logger.debug(f"Data loaded: {path} Shape={df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV {path}: {e}")
        raise


# ------------------------------------------------------
# Evaluation Logic
# ------------------------------------------------------
def evaluate(clf, X_test, y_test):
    """Compute accuracy, precision, recall, AUC."""
    try:
        y_pred = clf.predict(X_test)

        # Only compute AUC if probability is available
        if hasattr(clf, "predict_proba"):
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
        else:
            auc = None

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "auc": auc
        }

        logger.debug("Evaluation metrics computed successfully.")
        return metrics

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise


def save_metrics(metrics: dict, file_path: Path):
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.debug(f"Metrics saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        raise


# ------------------------------------------------------
# Main
# ------------------------------------------------------
def main():
    try:
        params = load_params()

        model_path = REPO_ROOT / "models" / "model.pkl"
        test_csv_path = REPO_ROOT / "data" / "processed" / "test_tfidf.csv"

        clf = load_model(model_path)

        test_df = load_csv(test_csv_path)
        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values

        metrics = evaluate(clf, X_test, y_test)

        # -------- DVCLive Logging ----------
        with Live(save_dvc_exp=True) as live:
            for k, v in metrics.items():
                if v is not None:   # Avoid logging None values
                    live.log_metric(k, v)

            live.log_params(params)

        # -------- Save JSON metrics --------
        save_metrics(metrics, REPO_ROOT / "reports" / "metrics.json")

        logger.info("✅ Model evaluation completed successfully.")

    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
