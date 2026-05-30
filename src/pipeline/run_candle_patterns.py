"""Run default candle pattern detection against CPM-labeled data."""

from __future__ import annotations

import argparse
import json

from src.indicators.candle import CANDLE_SIGNAL_COLUMNS, add_candle_patterns
from src.utils.config import load_config
from src.utils.io import load_dataframe, save_dataframe
from src.utils.paths import (
    get_candle_pattern_figures_dir,
    get_candle_pattern_counts_figure_path,
    get_candle_patterns_data_path,
    get_candle_patterns_figure_path,
    get_candle_patterns_output_path,
    get_tp_data_path,
)
from src.visualization.plot_signals import (
    plot_individual_candle_patterns,
    plot_candle_pattern_counts,
    plot_candle_patterns,
)


def main() -> None:
    """Generate candle pattern columns and save signal outputs."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spy_1d.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ticker = config["ticker"]
    interval = config["interval"]
    tp_path = get_tp_data_path(ticker, interval)
    if not tp_path.exists():
        raise FileNotFoundError(f"TP data not found: {tp_path}. Run run_cpm first.")

    df = load_dataframe(tp_path)
    missing = [col for col in ["Open", "High", "Low", "Close", "turning_label"] if col not in df.columns]
    if missing:
        raise ValueError(f"TP data missing required columns: {missing}")

    candle_config = config.get("candle_default_params", {})
    params = [
        float(candle_config.get("a", 2.0)),
        float(candle_config.get("b", 0.5)),
        float(candle_config.get("c", 0.5)),
        float(candle_config.get("d", 0.0)),
        float(candle_config.get("e", 0.0)),
        float(candle_config.get("f", 0.0)),
        float(candle_config.get("g", 0.0)),
    ]
    signal_df = add_candle_patterns(
        df,
        open_col="Open",
        high_col="High",
        low_col="Low",
        close_col="Close",
        params=params,
    )

    processed_signal_path = get_candle_patterns_data_path(ticker, interval)
    output_signal_path = get_candle_patterns_output_path(ticker, interval)
    figure_path = get_candle_patterns_figure_path(ticker, interval)
    individual_figure_dir = get_candle_pattern_figures_dir(ticker, interval)
    counts_figure_path = get_candle_pattern_counts_figure_path(ticker, interval)

    save_dataframe(signal_df, processed_signal_path)
    save_dataframe(signal_df, output_signal_path)
    plot_candle_patterns(signal_df, figure_path)
    plot_individual_candle_patterns(signal_df, individual_figure_dir)
    plot_candle_pattern_counts(signal_df, counts_figure_path, CANDLE_SIGNAL_COLUMNS)

    pattern_counts = {col: int((signal_df[col] != 0).sum()) for col in CANDLE_SIGNAL_COLUMNS}
    print("candle pattern counts:")
    print(json.dumps(pattern_counts, indent=2))
    print(f"processed candle patterns: {processed_signal_path}")
    print(f"output candle patterns: {output_signal_path}")
    print(f"candle figure: {figure_path}")
    print(f"individual candle figures: {individual_figure_dir}")
    print(f"candle count figure: {counts_figure_path}")


if __name__ == "__main__":
    main()
