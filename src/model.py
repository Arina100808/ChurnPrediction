import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from data import load_data, select_market, split_X_y, split_data, infer_columns
from pipeline import load_market_config, build_pipeline
from utils import ensure_dir

def cmd_train(args) -> str:
    """Train a churn prediction model using a train/test split.

    This function loads the dataset, filters by market if specified, splits the data into
    training and test sets, trains a preprocessing + XGBoost pipeline on the training set,
    evaluates it on the test set, and saves the trained model, metrics, and test set
    predictions.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        str: Path to the saved model file.
    """
    df = load_data(args.data)
    df = select_market(df, args.market, args.market_col)

    config = load_market_config(args.market)
    features = config.get("features")
    threshold = config.get("threshold", 0.5)
    model_params = config.get("model_params", {})

    X, y = split_X_y(df, args.target)
    if features:
        X = X[features]

    cats, nums = infer_columns(X)
    pipeline, model_params = build_pipeline(cats, nums, model_params)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=1)

    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    market_name = str(args.market) if args.market else "default"
    metrics = {
        "market": market_name,
        "n_rows_train": int(X_train.shape[0]),
        "n_rows_test": int(X_test.shape[0]),
        "n_features": int(X_test.shape[1]),
        "cat_cols": cats,
        "num_cols": nums,
        "target": args.target,
        "test_positive_rate": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, preds)),
        "auc": float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else None,
    }

    outdir = os.path.join(args.outdir, market_name)
    ensure_dir(outdir)
    model_path = os.path.join(outdir, f"model_{market_name}.joblib")
    joblib.dump(pipeline, model_path)

    metrics_path = os.path.join(outdir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    params_path = os.path.join(outdir, "params.json")
    with open(params_path, "w") as f:
        json.dump(model_params, f, indent=2)

    pred_path = os.path.join(outdir, f"pred_{market_name}.csv")
    pred_df = pd.DataFrame(proba, index=X_test.index, columns=["pred"])
    pred_df = pred_df.sort_index()
    pred_df.to_csv(pred_path, index=True, index_label="")

    metrics["model_path"] = model_path
    print(json.dumps(metrics, indent=2))

    return str(model_path)
