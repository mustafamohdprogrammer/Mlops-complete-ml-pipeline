# src/data_ingestion.py

import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


# ----------------------------
# Logging setup (console + file)
# ----------------------------
LOG_NAME = "data_ingestion"
logger = logging.getLogger(LOG_NAME)
logger.setLevel(logging.DEBUG)

# Ensure the "logs" directory exists at repo root (one level above this file)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = LOG_DIR / f"{LOG_NAME}.log"
file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Avoid duplicate handlers if this module is imported multiple times
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ----------------------------
# Helpers
# ----------------------------
def load_params(params_path: Path) -> Dict[str, Any]:
    """Load parameters from a YAML file. Raises if file not found/invalid."""
    try:
        with params_path.open("r", encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error reading params: %s", e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from CSV (supports remote URLs)."""
    try:
        # spam.csv often requires latin-1 encoding
        df = pd.read_csv(data_url, encoding="latin-1")
        logger.debug("Data loaded from %s (shape=%s)", data_url, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data:
      - drop unnamed junk columns (ignore if not present)
      - rename 'v1'->'target', 'v2'->'text'
    """
    try:
        df = df.copy()
        df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], errors="ignore", inplace=True)
        df.rename(columns={"v1": "target", "v2": "text"}, inplace=True)

        # Basic sanity: keep only needed columns if they exist
        wanted = [c for c in ["target", "text"] if c in df.columns]
        if wanted:
            df = df[wanted]

        logger.debug("Data preprocessing completed (columns=%s, shape=%s)", list(df.columns), df.shape)
        return df
    except Exception as e:
        logger.error("Unexpected error during preprocessing: %s", e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_root: Path) -> None:
    """Save the train and test datasets under data/raw."""
    try:
        raw_path = data_root / "raw"
        raw_path.mkdir(parents=True, exist_ok=True)
        train_fp = raw_path / "train.csv"
        test_fp = raw_path / "test.csv"
        train_data.to_csv(train_fp, index=False, encoding="utf-8")
        test_data.to_csv(test_fp, index=False, encoding="utf-8")
        logger.debug("Train and test data saved to %s", raw_path)
    except Exception as e:
        logger.error("Unexpected error occurred while saving the data: %s", e)
        raise


def split_data(df: pd.DataFrame, test_size: float, seed: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/test."""
    try:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
        logger.debug("Data split into train/test with test_size=%s (train=%s, test=%s)",
                     test_size, train_df.shape, test_df.shape)
        return train_df, test_df
    except Exception as e:
        logger.error("Error during train/test split: %s", e)
        raise


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    try:
        # Resolve repo root-relative paths
        params_file = REPO_ROOT / "params.yaml"
        data_dir = REPO_ROOT / "data"

        # Try to read test_size from params.yaml; fallback to 0.2 if not present/missing
        if params_file.exists():
            params = load_params(params_file)
            test_size = float(params.get("data_ingestion", {}).get("test_size", 0.21))
        else:
            logger.warning("params.yaml not found at %s; using default test_size=0.2", params_file)
            test_size = 0.2

        # Source data
        data_url = "https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv"

        # Pipeline
        df = load_data(data_url)
        df = preprocess_data(df)
        train_df, test_df = split_data(df, test_size=test_size, seed=2)
        save_data(train_df, test_df, data_root=data_dir)

        logger.info("Data ingestion completed successfully.")

    except Exception as e:
        logger.error("Failed to complete the data ingestion process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
