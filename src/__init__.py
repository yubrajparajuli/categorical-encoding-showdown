"""
src/__init__.py
---------------
Exposes the main functions from each module for clean imports.

Usage:
    from src.data_loader import load_and_prepare
    from src.encoders import label_encode, onehot_encode
    from src.visualize import plot_cardinality
    from src.compare import get_encoding_comparison
"""

from .data_loader import (
    load_data,
    clean_column_names,
    handle_missing_values,
    get_column_types,
    load_and_prepare
)

from .encoders import (
    label_encode,
    onehot_encode,
    dummy_encode,
    ordinal_encode,
    frequency_encode,
    target_encode,
    binary_encode
)

from .visualize import (
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

from .compare import (
    get_encoding_comparison,
    get_before_after_summary,
    print_encoding_summary,
    get_cardinality_summary
)

__all__ = [
    # data_loader
    "load_data",
    "clean_column_names",
    "handle_missing_values",
    "get_column_types",
    "load_and_prepare",
    # encoders
    "label_encode",
    "onehot_encode",
    "dummy_encode",
    "ordinal_encode",
    "frequency_encode",
    "target_encode",
    "binary_encode",
    # visualize
    "plot_target_distribution",
    "plot_cardinality",
    "plot_label_encoding",
    "plot_heatmap",
    "plot_ordinal_encoding",
    "plot_frequency_encoding",
    "plot_target_encoding",
    "plot_comparison_table",
    "plot_before_after",
    "plot_decision_flowchart",
    # compare
    "get_encoding_comparison",
    "get_before_after_summary",
    "print_encoding_summary",
    "get_cardinality_summary"
]