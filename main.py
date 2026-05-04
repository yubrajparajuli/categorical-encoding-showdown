import matplotlib
matplotlib.use('Agg')
import pandas as pd
from src.data_loader import load_and_prepare, get_column_types
from src.encoders import (
    label_encode,
    onehot_encode,
    dummy_encode,
    ordinal_encode,
    frequency_encode,
    target_encode,
    binary_encode
)
from src.visualize import (
    plot_target_distribution,
    plot_cardinality,
    plot_label_encoding,
    plot_heatmap,
    plot_ordinal_encoding,
    plot_frequency_encoding,
    plot_target_encoding,
    plot_comparison_table,
    plot_before_after,
    plot_decision_flowchart
)
from src.compare import (
    get_encoding_comparison,
    get_before_after_summary,
    get_cardinality_summary,
    print_encoding_summary
)


#Constants
EDUCATION_ORDER = [
    "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th",
    "10th", "11th", "12th", "HS-grad", "Some-college",
    "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters",
    "Prof-school", "Doctorate"
]


def run_pipeline() -> None:
    """
    Run the full categorical encoding pipeline.
    """

    print("\n" + "="*60)
    print("   CATEGORICAL ENCODING SHOWDOWN")
    print("="*60 + "\n")

    #Step 1: Load & Prepare Data
    print("── Step 1: Loading and preparing data...\n")
    df = load_and_prepare()
    col_types = get_column_types(df)
    print(f"\nCategorical columns: {col_types['categorical']}")
    print(f"Numerical columns:   {col_types['numerical']}\n")

    #Step 2: EDA Visualizations
    print("── Step 2: Generating EDA visualizations...\n")
    plot_target_distribution(df, target_col="income")
    plot_cardinality(df, categorical_cols=col_types["categorical"])

    #Cardinality summary
    print("\nCardinality Summary:")
    print(get_cardinality_summary(df).to_string(index=False))

    #Step 3: Label Encoding
    print("\n── Step 3: Label Encoding — marital_status\n")
    df["marital_status_label"] = label_encode(df, "marital_status")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(df["marital_status"])
    plot_label_encoding(
        classes=le.classes_.tolist(),
        encoded_values=list(range(len(le.classes_)))
    )

    #Step 4: One-Hot Encoding
    print("\n── Step 4: One-Hot Encoding — workclass, sex\n")
    ohe_df = onehot_encode(df, cols=["workclass", "sex"])
    plot_heatmap(
        ohe_df,
        title="One-Hot Encoding — Workclass & Sex",
        filename="onehot_encoding.png",
        cmap="Blues"
    )

    #Step 5: Dummy Encoding
    print("\n── Step 5: Dummy Encoding — relationship\n")
    dummy_df = dummy_encode(df, col="relationship")
    plot_heatmap(
        dummy_df,
        title="Dummy Encoding — Relationship (Husband is reference)",
        filename="dummy_encoding.png",
        cmap="Greens"
    )

    #Step 6: Ordinal Encoding
    print("\n── Step 6: Ordinal Encoding — education\n")
    df["education_ordinal"] = ordinal_encode(
        df, col="education", order=EDUCATION_ORDER
    )
    plot_ordinal_encoding(order=EDUCATION_ORDER)

    #Step 7: Frequency Encoding
    print("\n── Step 7: Frequency Encoding — native_country\n")
    df["native_country_freq"] = frequency_encode(df, col="native_country")
    freq_map = df["native_country"].value_counts()
    plot_frequency_encoding(freq_map=freq_map, col="native_country")

    #Step 8: Target Encoding
    print("\n── Step 8: Target Encoding — occupation\n")
    df["income_binary"] = (df["income"] == ">50K").astype(int)
    df["occupation_target"] = target_encode(
        df, col="occupation", target_col="income_binary"
    )
    target_map = df.groupby("occupation")["income_binary"].mean()
    plot_target_encoding(target_map=target_map, col="occupation")

    #Step 9: Binary Encoding
    print("\n── Step 9: Binary Encoding — race\n")
    race_binary = binary_encode(df, col="race")

    #Step 10: Comparison & Summary
    print("\n── Step 10: Generating comparison table...\n")
    comparison = get_encoding_comparison()
    plot_comparison_table(comparison)

    summary = get_before_after_summary()
    plot_before_after(summary)

    plot_decision_flowchart()

    #Final Summary
    print_encoding_summary()

    print("\n" + "="*60)
    print("   PIPELINE COMPLETE — All plots saved to images/")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_pipeline()