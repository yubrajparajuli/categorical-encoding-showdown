"""
visualize.py
------------
Contains all visualization functions for the categorical encoding project.
Each function saves the plot to the images/ directory and displays it.

Visualizations covered:
    1. Target variable distribution
    2. Cardinality per categorical column
    3. Label encoding mapping
    4. One-Hot encoding heatmap
    5. Dummy encoding heatmap
    6. Ordinal encoding mapping
    7. Frequency encoding bar chart
    8. Target encoding bar chart
    9. Binary encoding heatmap
    10. Comparison table
    11. Before vs After column count
    12. Decision flowchart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path


# Constants
IMAGES_DIR = Path("images")
DPI = 150


def _save_and_show(filename: str) -> None:
    """Save plot to images/ directory and display it."""
    IMAGES_DIR.mkdir(exist_ok=True)
    plt.savefig(IMAGES_DIR / filename, dpi=DPI, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"[INFO] Plot saved: images/{filename}")


def plot_target_distribution(df: pd.DataFrame, target_col: str) -> None:
    """
    Plot the distribution of the target variable.

    Args:
        df:         Input DataFrame.
        target_col: Name of the target column.
    """
    plt.figure(figsize=(6, 4))
    df[target_col].value_counts().plot(
        kind="bar",
        color=["steelblue", "salmon"]
    )
    plt.title(f"Target Variable Distribution ({target_col})")
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    _save_and_show("target_distribution.png")


def plot_cardinality(df: pd.DataFrame, categorical_cols: list[str]) -> None:
    """
    Plot the number of unique values per categorical column.
    Color-coded by cardinality level:
        - Red:   High cardinality (> 10)
        - Blue:  Medium cardinality (5-10)
        - Green: Low cardinality (< 5)

    Args:
        df:               Input DataFrame.
        categorical_cols: List of categorical column names.
    """
    cardinality = df[categorical_cols].nunique().sort_values(ascending=False)
    colors = [
        "tomato" if v > 10
        else "steelblue" if v > 5
        else "mediumseagreen"
        for v in cardinality.values
    ]

    plt.figure(figsize=(10, 5))
    cardinality.plot(kind="barh", color=colors)
    plt.axvline(x=10, color="red", linestyle="--", label="High cardinality (>10)")
    plt.axvline(x=5, color="blue", linestyle="--", label="Medium cardinality (>5)")
    plt.title("Cardinality per Categorical Column")
    plt.xlabel("Number of Unique Values")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    _save_and_show("cardinality.png")


def plot_label_encoding(classes: list, encoded_values: list) -> None:
    """
    Plot the label encoding mapping as a horizontal bar chart.

    Args:
        classes:        List of original category names.
        encoded_values: List of corresponding encoded integers.
    """
    plt.figure(figsize=(8, 4))
    plt.barh(classes, encoded_values, color="steelblue")
    plt.title("Label Encoding Mapping")
    plt.xlabel("Encoded Value")
    plt.tight_layout()
    _save_and_show("label_encoding.png")


def plot_heatmap(
    data: pd.DataFrame,
    title: str,
    filename: str,
    cmap: str = "Blues"
) -> None:
    """
    Plot a heatmap for binary encoding outputs (One-Hot, Dummy, Binary).

    Args:
        data:     DataFrame to visualize.
        title:    Plot title.
        filename: Output filename.
        cmap:     Colormap name.
    """
    plt.figure(figsize=(12, 5))
    sns.heatmap(
        data.head(10),
        annot=True,
        cbar=False,
        cmap=cmap,
        linewidths=0.5
    )
    plt.title(title)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0)
    plt.tight_layout()
    _save_and_show(filename)


def plot_ordinal_encoding(order: list[str]) -> None:
    """
    Plot the ordinal encoding mapping as a horizontal bar chart.

    Args:
        order: List of categories in ascending ordinal order.
    """
    plt.figure(figsize=(10, 5))
    plt.barh(order, range(len(order)), color="coral")
    plt.title("Ordinal Encoding — Education Level")
    plt.xlabel("Encoded Value (Higher = More Education)")
    plt.tight_layout()
    _save_and_show("ordinal_encoding.png")


def plot_frequency_encoding(freq_map: pd.Series, col: str, top_n: int = 15) -> None:
    """
    Plot the top N most frequent categories after frequency encoding.

    Args:
        freq_map: Series mapping categories to their frequencies.
        col:      Column name (used in title).
        top_n:    Number of top categories to display.
    """
    plt.figure(figsize=(10, 6))
    freq_map.head(top_n).plot(kind="barh", color="purple")
    plt.title(f"Frequency Encoding — {col} (Top {top_n})")
    plt.xlabel("Frequency Count")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    _save_and_show("frequency_encoding.png")


def plot_target_encoding(target_map: pd.Series, col: str) -> None:
    """
    Plot the target encoding mapping showing mean target value per category.

    Args:
        target_map: Series mapping categories to mean target values.
        col:        Column name (used in title).
    """
    plt.figure(figsize=(10, 6))
    target_map.sort_values().plot(kind="barh", color="tomato")
    plt.axvline(
        x=0.5,
        color="black",
        linestyle="--",
        label="50% threshold"
    )
    plt.title(f"Target Encoding — {col} (Mean Income >50K)")
    plt.xlabel("Mean Target Value (Probability of earning >50K)")
    plt.legend()
    plt.tight_layout()
    _save_and_show("target_encoding.png")


def plot_comparison_table(comparison: pd.DataFrame) -> None:
    """
    Plot the encoding comparison table with color-coded Yes/No cells.

    Args:
        comparison: DataFrame containing encoding comparison data.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    table = ax.table(
        cellText=comparison.values,
        colLabels=comparison.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)

    # Style header row
    for j in range(len(comparison.columns)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color Yes/No cells
    yes_no_cols = [
        i for i, col in enumerate(comparison.columns)
        if col in ["High Cardinality", "Preserves Order"]
    ]
    for i in range(1, len(comparison) + 1):
        for j in yes_no_cols:
            cell = table[i, j]
            text = cell.get_text().get_text()
            cell.set_facecolor("#d5f5e3" if text == "Yes" else "#fadbd8")

    plt.title(
        "Categorical Encoding — Complete Comparison",
        pad=20, fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    _save_and_show("comparison_table.png")


def plot_before_after(summary: pd.DataFrame) -> None:
    """
    Plot before vs after encoding column count comparison.

    Args:
        summary: DataFrame with original and encoded column counts.
    """
    plt.figure(figsize=(10, 5))
    x = range(len(summary))
    width = 0.35

    plt.barh(
        [i + width / 2 for i in x],
        summary["After Encoding"],
        width,
        label="After Encoding",
        color="tomato"
    )
    plt.barh(
        [i - width / 2 for i in x],
        summary["Original Columns"],
        width,
        label="Before Encoding",
        color="steelblue"
    )
    plt.yticks(x, summary["Original Column"])
    plt.xlabel("Number of Columns")
    plt.title("Before vs After Encoding — Column Count")
    plt.legend()
    plt.tight_layout()
    _save_and_show("before_after.png")


def plot_decision_flowchart() -> None:
    """
    Plot a decision flowchart to guide encoding technique selection.
    Covers all 7 encoding techniques with decision logic based on:
        - Natural order
        - Cardinality
        - Model type
        - Target variable availability
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def draw_box(x, y, text, color="#2c3e50", textcolor="white",
                 fontsize=9, width=2.2, height=0.6):
        box = mpatches.FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="white", linewidth=1.5
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color=textcolor,
                fontweight="bold", wrap=True)

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5)
        )

    def draw_label(x, y, text):
        ax.text(x, y, text, ha="center", va="center",
                fontsize=8, color="#555555", style="italic")

    # Nodes
    draw_box(5, 9.3, "Categorical Column", "#2c3e50", fontsize=10)
    draw_box(5, 8.0, "Is there a natural order?", "#8e44ad")
    draw_box(2, 6.8, "Ordinal Encoding", "#27ae60")
    draw_box(7, 6.8, "High Cardinality?\n(>10 unique)", "#8e44ad")
    draw_box(7, 5.3, "Target variable\navailable?", "#8e44ad")
    draw_box(9, 4.0, "Target Encoding", "#27ae60")
    draw_box(5.5, 4.0, "Frequency Encoding", "#27ae60")
    draw_box(3, 5.3, "Linear/Regression\nModel?", "#8e44ad")
    draw_box(1.5, 4.0, "Dummy Encoding", "#27ae60")
    draw_box(4, 4.0, "Tree-based\nModel?", "#8e44ad")
    draw_box(3, 2.7, "Label Encoding", "#27ae60")
    draw_box(5.5, 2.7, "One-Hot Encoding", "#27ae60")
    draw_box(7.5, 2.7, "Binary Encoding\n(medium cardinality)", "#e67e22")

    # Arrows
    draw_arrow(5, 9.0, 5, 8.3)
    draw_arrow(3.8, 8.0, 2.5, 7.1)
    draw_arrow(6.2, 8.0, 6.8, 7.1)
    draw_arrow(7, 6.5, 7, 5.7)
    draw_arrow(7.8, 5.3, 8.5, 4.2)
    draw_arrow(6.2, 5.3, 5.8, 4.3)
    draw_arrow(5.5, 6.8, 3.8, 5.6)
    draw_arrow(2.2, 5.3, 1.8, 4.3)
    draw_arrow(3.8, 5.3, 4.0, 4.3)
    draw_arrow(3.5, 4.0, 3.2, 3.0)
    draw_arrow(4.5, 4.0, 5.2, 3.0)
    draw_arrow(7, 4.0, 7.3, 3.0)

    # Labels
    draw_label(3.0, 7.6, "Yes")
    draw_label(6.4, 7.6, "No")
    draw_label(7.3, 6.1, "Yes")
    draw_label(8.5, 4.9, "Yes")
    draw_label(5.8, 4.9, "No")
    draw_label(4.5, 6.3, "No")
    draw_label(1.8, 4.9, "Yes")
    draw_label(4.2, 4.9, "No")
    draw_label(3.0, 3.5, "Yes")
    draw_label(5.2, 3.5, "No")
    draw_label(7.5, 3.5, "Alt")

    plt.title(
        "Which Encoding Should I Use? — Decision Guide",
        fontsize=13, fontweight="bold", pad=15
    )
    plt.tight_layout()
    _save_and_show("decision_flowchart.png")