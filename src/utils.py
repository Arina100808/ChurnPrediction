from pathlib import Path

def ensure_dir(p):
    """Create a directory if it doesn't exist."""
    Path(p).mkdir(parents=True, exist_ok=True)

def check_market_model_match(model_path: str, market: str):
    """Check if the model filename matches the target market.

    Args:
        model_path (str): Path to the trained model file.
        market (str): Market to check against.

    Raises:
        ValueError: If trained and current markets do not match.
    """
    if "model_" in model_path:
        trained_market = model_path.split("model_")[-1].split(".")[0]
        current_market = market or "default"
        if trained_market != current_market:
            raise ValueError(
                f"Model trained on market='{trained_market}', "
                f"but predicting on market='{current_market}'. "
                f"Cross-market prediction is not supported."
            )
