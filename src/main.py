import argparse
import json
import sys

from model import cmd_train
from predict import cmd_predict

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        argparse.ArgumentParser: CLI parser object.
    """
    p = argparse.ArgumentParser(description="Churn prediction (multi-market)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="Train model")
    pt.add_argument("--data", required=True)
    pt.add_argument("--market", required=False, help="e.g., AB, CD")
    pt.add_argument("--market-col", default="country")
    pt.add_argument("--target", default="churn")
    pt.add_argument("--outdir", default="artifacts")
    pt.set_defaults(func=cmd_train)

    pp = sub.add_parser("predict", help="Score new data")
    pp.add_argument("--data", required=True)
    pp.add_argument("--model", required=True)
    pp.add_argument("--market", required=False)
    pp.add_argument("--market-col", default="country")
    pp.add_argument("--target", default="churn")
    pp.add_argument("--outdir", default="artifacts")
    pp.set_defaults(func=cmd_predict)
    return p

def main():
    try:
        args = build_parser().parse_args()
        args.func(args)
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": str(e)
        }, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()
