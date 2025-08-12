import os
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

def load_market_config(market: str) -> dict:
    """Load a per-market configuration YAML file.

    Args:
        market (str): Market identifier (e.g., 'AB', 'CD'). If None, uses 'default'.

    Returns:
        dict: Configuration dictionary loaded from the YAML file.
    """
    config_path = f"src/config/markets/{market or 'default'}.yaml"
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}

def build_pipeline(cat_cols, num_cols, model_params) -> tuple[Pipeline, dict]:
    """Construct a pipeline with OneHotEncoder and XGBoost classifier.

    Args:
        cat_cols (List[str]): List of categorical column names.
        num_cols (List[str]): List of numerical column names.
        model_params (dict): Parameters for the XGBoost classifier.

    Returns:
        Pipeline: Full scikit-learn pipeline.
        model_params (dict): Custom parameters for the XGBoost classifier.
    """
    if not model_params:
        model_params = {
            "n_estimators": 500,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "max_depth": 6,
            "tree_method": "hist",
            "eval_metric": "logloss",
            "random_state": 1,
            "n_jobs": 0
        }

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    clf = XGBClassifier(**model_params)
    return Pipeline([("pre", pre), ("clf", clf)]), model_params
