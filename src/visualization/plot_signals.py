"""Indicator signal plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.paths import ensure_parent_dir


def plot_ma_signals(
    df: pd.DataFrame,
    output_path: str | Path,
    date_col: str = "Date",
    close_col: str = "Close",
    buy_signal_col: str = "ma_buy_signal",
    sell_signal_col: str = "ma_sell_signal",
) -> None:
    """Plot Close price with MA buy and sell signals."""

    required = [date_col, close_col, buy_signal_col, sell_signal_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for MA signal plot: {missing}")

    dates = pd.to_datetime(df[date_col])
    close = pd.to_numeric(df[close_col], errors="coerce")
    buys = df[buy_signal_col] == 1
    sells = df[sell_signal_col] == 1

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, close, label="Close", linewidth=1.0)
    ax.scatter(dates[buys], close[buys], marker="^", color="green", label="MA buy", s=35)
    ax.scatter(dates[sells], close[sells], marker="v", color="red", label="MA sell", s=35)
    ax.set_title("MA Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = Path(output_path)
    ensure_parent_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rsi_signals(
    df: pd.DataFrame,
    output_path: str | Path,
    date_col: str = "Date",
    close_col: str = "Close",
    rsi_col: str = "rsi",
    buy_signal_col: str = "rsi_buy_signal",
    sell_signal_col: str = "rsi_sell_signal",
    lower_threshold: float = 30.0,
    upper_threshold: float = 70.0,
) -> None:
    """Plot Close price with RSI buy/sell signals and an RSI subplot."""

    required = [date_col, close_col, rsi_col, buy_signal_col, sell_signal_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for RSI signal plot: {missing}")

    dates = pd.to_datetime(df[date_col])
    close = pd.to_numeric(df[close_col], errors="coerce")
    rsi = pd.to_numeric(df[rsi_col], errors="coerce")
    buys = df[buy_signal_col] == 1
    sells = df[sell_signal_col] == 1

    fig, (price_ax, rsi_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    price_ax.plot(dates, close, label="Close", linewidth=1.0)
    price_ax.scatter(dates[buys], close[buys], marker="^", color="green", label="RSI buy", s=35)
    price_ax.scatter(dates[sells], close[sells], marker="v", color="red", label="RSI sell", s=35)
    price_ax.set_title("RSI Signals")
    price_ax.set_ylabel("Close")
    price_ax.legend()
    price_ax.grid(True, alpha=0.3)

    rsi_ax.plot(dates, rsi, label="RSI", color="purple", linewidth=1.0)
    rsi_ax.axhline(lower_threshold, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
    rsi_ax.axhline(upper_threshold, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    rsi_ax.set_xlabel("Date")
    rsi_ax.set_ylabel("RSI")
    rsi_ax.set_ylim(0, 100)
    rsi_ax.legend()
    rsi_ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = Path(output_path)
    ensure_parent_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_signals(
    df: pd.DataFrame,
    output_path: str | Path,
    date_col: str = "Date",
    close_col: str = "Close",
    roc_col: str = "roc",
    buy_signal_col: str = "roc_buy_signal",
    sell_signal_col: str = "roc_sell_signal",
    buy_threshold: float = 1.0,
    sell_threshold: float = -1.0,
) -> None:
    """Plot Close price with ROC buy/sell signals and a ROC subplot."""

    required = [date_col, close_col, roc_col, buy_signal_col, sell_signal_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for ROC signal plot: {missing}")

    dates = pd.to_datetime(df[date_col])
    close = pd.to_numeric(df[close_col], errors="coerce")
    roc = pd.to_numeric(df[roc_col], errors="coerce")
    buys = df[buy_signal_col] == 1
    sells = df[sell_signal_col] == 1

    fig, (price_ax, roc_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    price_ax.plot(dates, close, label="Close", linewidth=1.0)
    price_ax.scatter(dates[buys], close[buys], marker="^", color="green", label="ROC buy", s=35)
    price_ax.scatter(dates[sells], close[sells], marker="v", color="red", label="ROC sell", s=35)
    price_ax.set_title("ROC Signals")
    price_ax.set_ylabel("Close")
    price_ax.legend()
    price_ax.grid(True, alpha=0.3)

    roc_ax.plot(dates, roc, label="ROC", color="darkorange", linewidth=1.0)
    roc_ax.axhline(0, color="black", linestyle="-", linewidth=0.7, alpha=0.5)
    roc_ax.axhline(buy_threshold, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
    roc_ax.axhline(sell_threshold, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    roc_ax.set_xlabel("Date")
    roc_ax.set_ylabel("ROC")
    roc_ax.legend()
    roc_ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = Path(output_path)
    ensure_parent_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stochastic_signals(
    df: pd.DataFrame,
    output_path: str | Path,
    date_col: str = "Date",
    close_col: str = "Close",
    k_col: str = "stoch_k",
    d_col: str = "stoch_d",
    buy_signal_col: str = "stoch_buy_signal",
    sell_signal_col: str = "stoch_sell_signal",
    lower_threshold: float = 20.0,
    upper_threshold: float = 80.0,
) -> None:
    """Plot Close price with Stochastic signals and %K/%D subplot."""

    required = [date_col, close_col, k_col, d_col, buy_signal_col, sell_signal_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for Stochastic signal plot: {missing}")

    dates = pd.to_datetime(df[date_col])
    close = pd.to_numeric(df[close_col], errors="coerce")
    stoch_k = pd.to_numeric(df[k_col], errors="coerce")
    stoch_d = pd.to_numeric(df[d_col], errors="coerce")
    buys = df[buy_signal_col] == 1
    sells = df[sell_signal_col] == 1

    fig, (price_ax, stoch_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    price_ax.plot(dates, close, label="Close", linewidth=1.0)
    price_ax.scatter(
        dates[buys],
        close[buys],
        marker="^",
        color="green",
        label="Stochastic buy",
        s=35,
    )
    price_ax.scatter(
        dates[sells],
        close[sells],
        marker="v",
        color="red",
        label="Stochastic sell",
        s=35,
    )
    price_ax.set_title("Stochastic Signals")
    price_ax.set_ylabel("Close")
    price_ax.legend()
    price_ax.grid(True, alpha=0.3)

    stoch_ax.plot(dates, stoch_k, label="%K", color="tab:blue", linewidth=1.0)
    stoch_ax.plot(dates, stoch_d, label="%D", color="darkorange", linewidth=1.0)
    stoch_ax.axhline(lower_threshold, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
    stoch_ax.axhline(upper_threshold, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    stoch_ax.set_xlabel("Date")
    stoch_ax.set_ylabel("Stochastic")
    stoch_ax.set_ylim(0, 100)
    stoch_ax.legend()
    stoch_ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = Path(output_path)
    ensure_parent_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_candle_patterns(
    df: pd.DataFrame,
    output_path: str | Path,
    date_col: str = "Date",
    close_col: str = "Close",
    hammer_col: str = "candle_hammer_hanging_man_signal",
    dark_cloud_col: str = "candle_dark_cloud_cover_signal",
    piercing_col: str = "candle_piercing_line_signal",
    bullish_engulfing_col: str = "candle_bullish_engulfing_signal",
    bearish_engulfing_col: str = "candle_bearish_engulfing_signal",
) -> None:
    """Plot Close price with independent candle pattern signals."""

    required = [
        date_col,
        close_col,
        hammer_col,
        dark_cloud_col,
        piercing_col,
        bullish_engulfing_col,
        bearish_engulfing_col,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for candle pattern plot: {missing}")

    dates = pd.to_datetime(df[date_col])
    close = pd.to_numeric(df[close_col], errors="coerce")
    hammer_bull = df[hammer_col] == 1
    hammer_bear = df[hammer_col] == -1
    dark_cloud = df[dark_cloud_col] == -1
    piercing = df[piercing_col] == 1
    bullish_engulfing = df[bullish_engulfing_col] == 1
    bearish_engulfing = df[bearish_engulfing_col] == -1

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, close, label="Close", linewidth=1.0)
    ax.scatter(
        dates[hammer_bull],
        close[hammer_bull],
        marker="o",
        color="royalblue",
        label="Hammer",
        s=28,
        alpha=0.7,
    )
    ax.scatter(
        dates[hammer_bear],
        close[hammer_bear],
        marker="o",
        color="darkorange",
        label="Hanging Man",
        s=28,
        alpha=0.7,
    )
    ax.scatter(dates[piercing], close[piercing], marker="^", color="green", label="Piercing Line", s=35)
    ax.scatter(
        dates[bullish_engulfing],
        close[bullish_engulfing],
        marker="^",
        color="limegreen",
        label="Bullish Engulfing",
        s=35,
    )
    ax.scatter(
        dates[dark_cloud],
        close[dark_cloud],
        marker="v",
        color="red",
        label="Dark Cloud Cover",
        s=35,
    )
    ax.scatter(
        dates[bearish_engulfing],
        close[bearish_engulfing],
        marker="v",
        color="darkred",
        label="Bearish Engulfing",
        s=35,
    )
    ax.set_title("Candle Pattern Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = Path(output_path)
    ensure_parent_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_individual_candle_patterns(
    df: pd.DataFrame,
    output_dir: str | Path,
    date_col: str = "Date",
    close_col: str = "Close",
) -> None:
    """Plot each candle pattern signal to its own PNG file."""

    pattern_specs = [
        (
            "candle_hammer_hanging_man_signal",
            "hammer_hanging_man.png",
            "Hammer / Hanging Man",
            [("Hammer", 1, "^", "green"), ("Hanging Man", -1, "v", "red")],
        ),
        (
            "candle_dark_cloud_cover_signal",
            "dark_cloud_cover.png",
            "Dark Cloud Cover",
            [("Dark Cloud Cover", -1, "v", "red")],
        ),
        (
            "candle_piercing_line_signal",
            "piercing_line.png",
            "Piercing Line",
            [("Piercing Line", 1, "^", "green")],
        ),
        (
            "candle_bullish_engulfing_signal",
            "bullish_engulfing.png",
            "Bullish Engulfing",
            [("Bullish Engulfing", 1, "^", "green")],
        ),
        (
            "candle_bearish_engulfing_signal",
            "bearish_engulfing.png",
            "Bearish Engulfing",
            [("Bearish Engulfing", -1, "v", "red")],
        ),
    ]

    required = [date_col, close_col, *[spec[0] for spec in pattern_specs]]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for individual candle pattern plots: {missing}")

    dates = pd.to_datetime(df[date_col])
    close = pd.to_numeric(df[close_col], errors="coerce")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    for signal_col, filename, title, markers in pattern_specs:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, close, label="Close", linewidth=1.0)
        for label, signal_value, marker, color in markers:
            rows = df[signal_col] == signal_value
            ax.scatter(
                dates[rows],
                close[rows],
                marker=marker,
                color=color,
                label=label,
                s=35,
            )
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Close")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=150)
        plt.close(fig)


def plot_candle_pattern_counts(
    df: pd.DataFrame,
    output_path: str | Path,
    pattern_cols: list[str],
) -> None:
    """Plot occurrence counts for individual candle pattern columns."""

    missing = [col for col in pattern_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for candle pattern counts: {missing}")

    counts = (df[pattern_cols] != 0).sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Candle Pattern Counts")
    ax.set_xlabel("Pattern")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = Path(output_path)
    ensure_parent_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
