"""
encoders.py
-----------
Contains all categorical encoding techniques used in this project.
Each encoder is implemented as a standalone function that takes a
DataFrame and column name, and returns the encoded Series or DataFrame.

Encoding techniques covered:
    1. Label Encoding
    2. One-Hot Encoding
    3. Dummy Encoding
    4. Ordinal Encoding
    5. Frequency Encoding
    6. Target Encoding
    7. Binary Encoding
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import category_encoders as ce
from typing import Optional


def label_encode(
    df: pd.DataFrame,
    col: str
) -> pd.Series:
    """
    Encode categories as integers using sklearn LabelEncoder.
    Order is assigned alphabetically — not meaningful.

    Best for: Tree-based models (Decision Tree, Random Forest, XGBoost)
    Avoid for: Linear models — implies false ordinal relationship.

    Args:
        df:  Input DataFrame.
        col: Column name to encode.

    Returns:
        Encoded Series with suffix '_label'.
    """
    le = LabelEncoder()
    encoded = le.fit_transform(df[col])
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    print(f"[Label Encoding] '{col}' mapping: {mapping}")
    return pd.Series(encoded, index=df.index, name=f"{col}_label")


def onehot_encode(
    df: pd.DataFrame,
    cols: list[str]
) -> pd.DataFrame:
    """
    Create binary columns for each unique category.
    n categories → n binary columns.

    Best for: Low cardinality columns (< 10 unique values)
    Avoid for: High cardinality — causes dimensionality explosion.

    Args:
        df:   Input DataFrame.
        cols: List of column names to encode.

    Returns:
        DataFrame with one-hot encoded columns.
    """
    encoded = pd.get_dummies(df[cols], dtype=int)
    print(f"[One-Hot Encoding] {cols} → {encoded.shape[1]} new columns")
    return encoded


def dummy_encode(
    df: pd.DataFrame,
    col: str
) -> pd.DataFrame:
    """
    One-hot encoding with first category dropped to avoid
    multicollinearity (dummy variable trap).
    n categories → n-1 binary columns.

    Best for: Linear and regression models.
    Avoid for: Tree-based models — dropping a column loses no info
               for trees but may matter for linear models.

    Args:
        df:  Input DataFrame.
        col: Column name to encode.

    Returns:
        DataFrame with dummy encoded columns.
    """
    encoded = pd.get_dummies(df[[col]], drop_first=True, dtype=int)
    print(f"[Dummy Encoding] '{col}' → {encoded.shape[1]} columns (1 dropped)")
    return encoded


def ordinal_encode(
    df: pd.DataFrame,
    col: str,
    order: list[str]
) -> pd.Series:
    """
    Encode categories as integers based on a meaningful,
    manually defined order.

    Best for: Columns with natural ranking (education, rating, size)
    Avoid for: Nominal categories with no meaningful order.

    Args:
        df:    Input DataFrame.
        col:   Column name to encode.
        order: List of categories in ascending order.

    Returns:
        Encoded Series with suffix '_ordinal'.

    Raises:
        ValueError: If categories in data don't match provided order.
    """
    # Validate all categories are covered
    data_cats = set(df[col].unique())
    order_cats = set(order)
    if not data_cats.issubset(order_cats):
        missing = data_cats - order_cats
        raise ValueError(
            f"[Ordinal Encoding] Categories not in order list: {missing}"
        )

    oe = OrdinalEncoder(categories=[order])
    encoded = oe.fit_transform(df[[col]])
    print(f"[Ordinal Encoding] '{col}' → {len(order)} levels defined")
    return pd.Series(
        encoded.flatten(),
        index=df.index,
        name=f"{col}_ordinal"
    )


def frequency_encode(
    df: pd.DataFrame,
    col: str
) -> pd.Series:
    """
    Replace each category with its frequency count in the dataset.
    Useful for capturing popularity or rarity of a category.

    Best for: High cardinality columns
    Caveat:   Two categories with same frequency get same value.

    Args:
        df:  Input DataFrame.
        col: Column name to encode.

    Returns:
        Encoded Series with suffix '_freq'.
    """
    freq_map = df[col].value_counts()
    encoded = df[col].map(freq_map)
    print(f"[Frequency Encoding] '{col}' → top value: "
          f"'{freq_map.index[0]}' ({freq_map.iloc[0]} occurrences)")
    return encoded.rename(f"{col}_freq")


def target_encode(
    df: pd.DataFrame,
    col: str,
    target_col: str
) -> pd.Series:
    """
    Replace each category with the mean of the target variable.
    Captures the relationship between category and outcome.

    Best for: High cardinality columns in supervised learning.
    Caveat:   Risk of data leakage — use cross-validation in production.

    Args:
        df:         Input DataFrame.
        col:        Column name to encode.
        target_col: Binary target column name (0/1).

    Returns:
        Encoded Series with suffix '_target'.
    """
    target_map = df.groupby(col)[target_col].mean()
    encoded = df[col].map(target_map)
    print(f"[Target Encoding] '{col}' → highest: "
          f"'{target_map.idxmax()}' ({target_map.max():.3f}), "
          f"lowest: '{target_map.idxmin()}' ({target_map.min():.3f})")
    return encoded.rename(f"{col}_target")


def binary_encode(
    df: pd.DataFrame,
    col: str
) -> pd.DataFrame:
    """
    Convert categories to integer codes then represent as binary digits.
    n categories → log2(n) columns. Efficient middle ground between
    label and one-hot encoding.

    Best for: Medium-high cardinality columns.
    Caveat:   Less interpretable than one-hot encoding.

    Args:
        df:  Input DataFrame.
        col: Column name to encode.

    Returns:
        DataFrame with binary encoded columns.
    """
    be = ce.BinaryEncoder(cols=[col])
    encoded = be.fit_transform(df[[col]])
    print(f"[Binary Encoding] '{col}' → {df[col].nunique()} categories "
          f"→ {encoded.shape[1]} binary columns")
    return encoded