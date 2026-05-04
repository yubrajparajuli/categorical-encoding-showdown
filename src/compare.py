"""
compare.py
----------
Generates summary and comparison data for all encoding techniques.
Provides a centralized place to compare encodings by:
    - Number of columns created
    - Cardinality handling
    - Order preservation
    - Risk factors
    - Best use cases

Used for the final comparison table and before/after summary
in both the notebook and the Medium blog.
"""

import pandas as pd


def get_encoding_comparison() -> pd.DataFrame:
    """
    Build a comparison table for all 7 encoding techniques.

    Returns:
        DataFrame summarizing each encoding technique with:
            - Columns Created
            - High Cardinality handling
            - Order Preservation
            - Risk
            - Best Use Case
    """
    comparison = pd.DataFrame({
        "Encoding": [
            "Label",
            "One-Hot",
            "Dummy",
            "Ordinal",
            "Frequency",
            "Target",
            "Binary"
        ],
        "Columns Created": [
            "1",
            "n",
            "n-1",
            "1",
            "1",
            "1",
            "log2(n)"
        ],
        "High Cardinality": [
            "Yes", "No", "No", "Yes", "Yes", "Yes", "Yes"
        ],
        "Preserves Order": [
            "No", "No", "No", "Yes", "No", "No", "No"
        ],
        "Risk": [
            "False order",
            "Dimensionality explosion",
            "Multicollinearity",
            "Manual mapping needed",
            "Collision risk",
            "Data leakage",
            "Less interpretable"
        ],
        "Best For": [
            "Tree models",
            "Low cardinality",
            "Linear/Regression",
            "Ordinal data",
            "High cardinality",
            "High cardinality",
            "Medium cardinality"
        ]
    })
    return comparison


def get_before_after_summary() -> pd.DataFrame:
    """
    Build a before vs after encoding summary showing
    how many columns each technique creates.

    Returns:
        DataFrame with original and encoded column counts
        per encoding technique applied in this project.
    """
    summary = pd.DataFrame({
        "Original Column": [
            "marital_status",
            "workclass + sex",
            "relationship",
            "education",
            "native_country",
            "occupation",
            "race"
        ],
        "Encoding Used": [
            "Label",
            "One-Hot",
            "Dummy",
            "Ordinal",
            "Frequency",
            "Target",
            "Binary"
        ],
        "Original Columns": [1, 2, 1, 1, 1, 1, 1],
        "After Encoding":   [1, 9, 5, 1, 1, 1, 3]
    })
    return summary


def print_encoding_summary() -> None:
    """
    Print a formatted summary of all encoding techniques
    with their key properties and recommendations.
    """
    summary_text = """
    ╔══════════════════════════════════════════════════════════════╗
    ║         CATEGORICAL ENCODING — QUICK REFERENCE              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Label      → Tree models, avoid for linear models          ║
    ║  One-Hot    → Low cardinality (< 10 unique values)          ║
    ║  Dummy      → Linear/Regression models                      ║
    ║  Ordinal    → When natural order exists (education, rating)  ║
    ║  Frequency  → High cardinality, no target variable          ║
    ║  Target     → High cardinality, supervised learning         ║
    ║  Binary     → Medium cardinality, memory efficient          ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(summary_text)


def get_cardinality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize cardinality of all categorical columns with
    recommended encoding based on unique value count.

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with cardinality and encoding recommendation
        per categorical column.
    """
    cat_cols = df.select_dtypes(include="str").columns.tolist()
    cardinality = df[cat_cols].nunique().sort_values(ascending=False)

    def recommend(n: int) -> str:
        if n == 2:
            return "One-Hot (binary)"
        elif n <= 5:
            return "One-Hot or Binary"
        elif n <= 10:
            return "One-Hot, Dummy, or Label"
        else:
            return "Target, Frequency, or Binary"

    result = pd.DataFrame({
        "Column": cardinality.index,
        "Unique Values": cardinality.values,
        "Cardinality Level": [
            "Low" if n <= 5
            else "Medium" if n <= 10
            else "High"
            for n in cardinality.values
        ],
        "Recommended Encoding": [
            recommend(n) for n in cardinality.values
        ]
    })
    return result