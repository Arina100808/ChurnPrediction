import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str) -> pd.DataFrame:
    """Load data from the CSV file. Automatically handle unnamed index column if present.

    Args:
        path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        ValueError: If the loaded DataFrame is empty.
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No data found in file: {path}")

    first_col = df.columns[0]
    if first_col.lower().startswith("unnamed"):
        df.set_index(first_col, inplace=True)
        df.index.name = None

    return df

def select_market(df: pd.DataFrame, market: str, market_col: str) -> pd.DataFrame:
    """Filters the DataFrame for a specific market using a given market column.
    If no market is specified, returns the unfiltered DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        market (str): The market value to filter by. If None, returns unfiltered DataFrame.
        market_col (str): The column name used to identify the market.

    Returns:
        pd.DataFrame: Filtered DataFrame for the specified market.

    Raises:
        ValueError: If the specified market_col is not in the DataFrame.
        ValueError: If market is provided but no matching rows are found.
    """
    if market_col not in df.columns:
        raise ValueError(f"Market column '{market_col}' not found in dataset.")
    elif market:
        df = df[df[market_col] == market]
        if df.empty:
            raise ValueError(f"No data found for market='{market}' in column '{market_col}'.")
        df = df.drop(columns=[market_col])
    return df

def process_target_column(df: pd.DataFrame, target: str) -> pd.Series:
    """Convert a binary target column to 0 and 1 using known positive indicators.

    Only allows known positive and negative categories.
    Raises an error if the column has unexpected values or is not binary.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target (str): Name of the target column.

    Returns:
        pd.Series: Binary target column with 1 for positive class, 0 otherwise.

    Raises:
        ValueError: If a column has unsupported values or is not binary.
    """
    y_normalized = df[target].astype(str).str.strip().str.lower()

    positive_values = {"yes", "true", "1"}
    negative_values = {"no", "false", "0"}
    allowed_values = positive_values.union(negative_values)

    unique_values = set(y_normalized.unique())

    if len(unique_values) != 2:
        raise ValueError(
            f"Target column '{target}' must be binary. Found values: {unique_values}"
        )

    if not unique_values.issubset(allowed_values):
        invalid = unique_values - allowed_values
        raise ValueError(
            f"Target column '{target}' contains unsupported values: {invalid}"
        )

    return y_normalized.isin(positive_values).astype(int)

def split_X_y(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features and binary target column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target (str): Name of the target column.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix X and binary target y.
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in data. Available: {list(df.columns)}")
    y = process_target_column(df, target)
    X = df.drop(columns=[target])
    return X, y

def infer_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify categorical and numerical columns from DataFrame.

    Args:
        X (pd.DataFrame): Feature matrix.

    Returns:
        Tuple[List[str], List[str]]: Categorical and numerical columns.
    """
    cats = X.select_dtypes(include=["object", "category"]).columns.tolist()
    nums = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    return cats, nums

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 1) -> tuple:
    """Split data into training and test sets.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Training and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
