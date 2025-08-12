import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from data import load_data, select_market
from utils import ensure_dir, check_market_model_match

def cmd_predict(args) -> str:
    """Load model and predict churn probabilities on new data.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        str: Path to prediction output CSV.
    """
    check_market_model_match(args.model, args.market)
    pipe: Pipeline = joblib.load(args.model)

    df = load_data(args.data)
    if args.target in df.columns:
        df = df.drop(columns=[args.target])
    df = select_market(df, args.market, args.market_col)

    proba = pipe.predict_proba(df)[:, 1]

    ensure_dir(args.outdir)
    market_name = str(args.market) if args.market else "default"
    out_path = os.path.join(args.outdir, f"scored_{market_name}.csv")
    pred_df = pd.DataFrame(proba, index=df.index, columns=["pred"])
    pred_df = pred_df.sort_index()
    pred_df.to_csv(out_path, index=True, index_label="")

    print(json.dumps({
        "status": "ok",
        "scored_path": out_path,
        "n_scored": int(len(proba))
    }, indent=2))

    return str(out_path)