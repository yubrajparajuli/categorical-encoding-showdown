"""
data_loader.py
--------------
Handles loading and preprocessing of the Adult Income dataset.
All cleaning steps are centralized here to ensure consistency
across the project.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Constants
DATA_PATH = Path("data/adult.csv")
MISSING_INDICATOR = "?"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the Adult Income dataset from a CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        Raw DataFrame with no modifications.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to snake_case.
    Strips whitespace, replaces dots and spaces with underscores,
    and converts to lowercase.

    Args:
        df: Raw DataFrame.

    Returns:
        DataFrame with cleaned column names.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(".", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and remove rows with hidden missing values represented as '?'.
    Also strips leading/trailing whitespace from all string columns.

    Args:
        df: DataFrame with potential '?' missing values.

    Returns:
        Cleaned DataFrame with missing rows removed.
    """
    df = df.copy()

    # Strip whitespace from all string columns
    str_cols = df.select_dtypes(include="str").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # Report missing values before dropping
    missing = {}
    for col in str_cols:
        count = (df[col] == MISSING_INDICATOR).sum()
        if count > 0:
            pct = (count / len(df)) * 100
            missing[col] = (count, pct)
            print(f"[INFO] '{col}': {count} missing values ({pct:.2f}%)")

    # Replace '?' with NaN then drop
    df.replace(MISSING_INDICATOR, np.nan, inplace=True)
    rows_before = len(df)
    df.dropna(inplace=True)
    rows_after = len(df)

    print(f"[INFO] Dropped {rows_before - rows_after} rows with missing values")
    print(f"[INFO] Remaining rows: {rows_after}")
    return df


def get_column_types(df: pd.DataFrame) -> dict:
    """
    Separate columns into categorical and numerical groups.

    Args:
        df: Cleaned DataFrame.

    Returns:
        Dictionary with 'categorical' and 'numerical' column lists.
    """
    return {
        "categorical": df.select_dtypes(include="str").columns.tolist(),
        "numerical": df.select_dtypes(include="number").columns.tolist()
    }


def load_and_prepare(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Full pipeline: load → clean columns → handle missing values.
    This is the main entry point for loading data.

    Args:
        path: Path to the CSV file.

    Returns:
        Fully cleaned and prepared DataFrame.
    """
    df = load_data(path)
    df = clean_column_names(df)
    df = handle_missing_values(df)
    return df