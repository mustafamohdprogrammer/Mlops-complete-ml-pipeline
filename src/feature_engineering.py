# src/feature_engineering.py

import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump


# ----------------------------
# Paths & logging
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = LOG_DIR / "feature_engineering.log"
file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ----------------------------
# Helpers
# ----------------------------
def load_params(params_path: Path) -> Dict[str, Any]:
    """Load parameters from YAML. Raises if YAML invalid; caller can handle missing file."""
    with params_path.open("r", encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}
    logger.debug("Parameters retrieved from %s", params_path)
    return params


def resolve_config() -> Dict[str, Any]:
    """
    Build a config dict using params.yaml if available, else use defaults.
    Defaults target the outputs from the earlier data_ingestion step.
    """
    params_file = REPO_ROOT / "params.yaml"
    cfg: Dict[str, Any] = {}

    # Defaults
    cfg.setdefault("paths", {})
    cfg["paths"].setdefault("train_csv", str(REPO_ROOT / "data" / "interim" / "train_processed.csv"))
    cfg["paths"].setdefault("test_csv", str(REPO_ROOT / "data" / "interim" / "test_processed.csv"))
    cfg["paths"].setdefault("fallback_train_csv", str(REPO_ROOT / "data" / "raw" / "train.csv"))
    cfg["paths"].setdefault("fallback_test_csv", str(REPO_ROOT / "data" / "raw" / "test.csv"))
    cfg["paths"].setdefault("processed_dir", str(REPO_ROOT / "data" / "processed"))
    cfg["paths"].setdefault("artifacts_dir", str(REPO_ROOT / "artifacts"))

    cfg.setdefault("feature_engineering", {})
    cfg["feature_engineering"].setdefault("max_features", 5000)
    cfg["feature_engineering"].setdefault("ngram_min", 1)
    cfg["feature_engineering"].setdefault("ngram_max", 1)
    cfg["feature_engineering"].setdefault("use_idf", True)
    cfg["feature_engineering"].setdefault("norm", "l2")
    # Set to "english" to remove English stopwords; keep None to keep all tokens
    cfg["feature_engineering"].setdefault("stop_words", None)

    if params_file.exists():
        try:
            params = load_params(params_file)
            fe = params.get("feature_engineering", {})
            # merge FE params
            for k in ["max_features", "ngram_min", "ngram_max", "use_idf", "norm", "stop_words"]:
                if k in fe and fe[k] is not None:
                    cfg["feature_engineering"][k] = fe[k]

            p = params.get("paths", {})
            for k in ["train_csv", "test_csv", "processed_dir", "artifacts_dir"]:
                if k in p and p[k]:
                    cfg["paths"][k] = p[k]

            logger.debug("Config resolved from params.yaml with overrides.")
        except Exception as e:
            logger.warning("Unable to parse params.yaml; using defaults. Error: %s", e)
    else:
        logger.warning("params.yaml not found; using default configuration.")

    return cfg


def load_data(file_path: Path) -> pd.DataFrame:
    """Load CSV and ensure expected columns exist."""
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        logger.debug("Data loaded from %s (shape=%s)", file_path, df.shape)
        # Standardize column names if needed
        cols = {c.lower(): c for c in df.columns}
        # Handle common variants
        for candidate in ["text", "message", "content", "body"]:
            if candidate in cols and "text" not in df.columns:
                df.rename(columns={cols[candidate]: "text"}, inplace=True)
                break
        for candidate in ["target", "label", "class", "y"]:
            if candidate in cols and "target" not in df.columns:
                df.rename(columns={cols[candidate]: "target"}, inplace=True)
                break

        if "text" not in df.columns or "target" not in df.columns:
            raise KeyError(
                f"Expected columns 'text' and 'target' not found in {file_path}. "
                f"Got columns: {list(df.columns)}"
            )

        # Fill NA text with empty string to avoid vectorizer errors
        df["text"] = df["text"].fillna("")
        return df
    except Exception as e:
        logger.error("Error loading %s: %s", file_path, e)
        raise


def pick_inputs(cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    """
    Choose input CSVs. Prefer interim; if missing, fall back to raw.
    """
    train_p = Path(cfg["paths"]["train_csv"])
    test_p = Path(cfg["paths"]["test_csv"])
    if not train_p.exists() or not test_p.exists():
        logger.warning(
            "Interim files not found (%s, %s). Falling back to raw dataset.",
            train_p, test_p
        )
        train_p = Path(cfg["paths"]["fallback_train_csv"])
        test_p = Path(cfg["paths"]["fallback_test_csv"])

    if not train_p.exists() or not test_p.exists():
        raise FileNotFoundError(
            f"Input CSVs not found. Checked:\n"
            f"- Interim: {cfg['paths']['train_csv']}, {cfg['paths']['test_csv']}\n"
            f"- Raw:     {cfg['paths']['fallback_train_csv']}, {cfg['paths']['fallback_test_csv']}\n"
            f"Run your data_ingestion (and preprocessing, if any) first."
        )
    return train_p, test_p


def apply_tfidf(train_df: pd.DataFrame, test_df: pd.DataFrame, fe_cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, TfidfVectorizer]:
    """Fit TF-IDF on train text and transform both splits; return dense DataFrames with label column appended."""
    vectorizer = TfidfVectorizer(
        max_features=int(fe_cfg["max_features"]),
        ngram_range=(int(fe_cfg["ngram_min"]), int(fe_cfg["ngram_max"])),
        use_idf=bool(fe_cfg["use_idf"]),
        norm=fe_cfg["norm"] if fe_cfg["norm"] else None,
        stop_words=fe_cfg["stop_words"] if fe_cfg["stop_words"] not in (None, "None", "null") else None,
    )

    X_train = train_df["text"].astype(str).values
    y_train = train_df["target"].values
    X_test = test_df["text"].astype(str).values
    y_test = test_df["target"].values

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    train_out = pd.DataFrame(X_train_tfidf.toarray())
    train_out["label"] = y_train

    test_out = pd.DataFrame(X_test_tfidf.toarray())
    test_out["label"] = y_test

    logger.debug(
        "TF-IDF applied: max_features=%s, ngram_range=(%s,%s), use_idf=%s, norm=%s, stop_words=%s",
        fe_cfg["max_features"], fe_cfg["ngram_min"], fe_cfg["ngram_max"],
        fe_cfg["use_idf"], fe_cfg["norm"], fe_cfg["stop_words"]
    )
    logger.debug("Shapes -> train: %s, test: %s", train_out.shape, test_out.shape)
    return train_out, test_out, vectorizer


def save_csv(df: pd.DataFrame, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False, encoding="utf-8")
    logger.debug("Saved %s", file_path)


def save_vectorizer(vectorizer: TfidfVectorizer, artifacts_dir: Path) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    fp = artifacts_dir / "vectorizer.joblib"
    dump(vectorizer, fp)
    logger.debug("Vectorizer saved to %s", fp)
    return fp


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    try:
        cfg = resolve_config()

        # Choose inputs
        train_csv, test_csv = pick_inputs(cfg)

        # Load
        train_df = load_data(train_csv)
        test_df = load_data(test_csv)

        # Vectorize
        fe_cfg = cfg["feature_engineering"]
        train_tfidf, test_tfidf, vec = apply_tfidf(train_df, test_df, fe_cfg)

        # Save outputs
        processed_dir = Path(cfg["paths"]["processed_dir"])
        save_csv(train_tfidf, processed_dir / "train_tfidf.csv")
        save_csv(test_tfidf, processed_dir / "test_tfidf.csv")

        # Save artifacts
        artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
        save_vectorizer(vec, artifacts_dir)

        logger.info("Feature engineering completed successfully.")

    except Exception as e:
        logger.error("Failed to complete the feature engineering process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
