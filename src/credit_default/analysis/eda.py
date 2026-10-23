"""Стадия EDA: графики разведочного анализа обучающей выборки."""

from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from credit_default.config import FIGURES_DIR, PROJECT_ROOT, TARGET, TRAIN_PATH

NUMERIC_FEATURES = ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"]
CLASS_LABELS = {0: "нет дефолта", 1: "дефолт"}
CLASS_COLORS = {0: "tab:blue", 1: "tab:red"}
CLIP_QUANTILES = (0.01, 0.99)


def plot_class_balance(df: pd.DataFrame) -> Figure:
    """Строит столбчатую диаграмму числа клиентов в каждом классе с долей класса."""
    counts = df[TARGET].value_counts().sort_index()
    fig = Figure(figsize=(5, 4), layout="constrained")
    ax = fig.subplots()
    bars = ax.bar(
        [CLASS_LABELS[code] for code in counts.index],
        counts.to_numpy(),
        color=[CLASS_COLORS[code] for code in counts.index],
    )
    ax.bar_label(bars, labels=[f"{count} ({count / len(df):.1%})" for count in counts])
    ax.set_title("Баланс классов")
    ax.set_ylabel("Число клиентов")
    return fig


def plot_distributions(df: pd.DataFrame) -> Figure:
    """Строит гистограммы числовых признаков по классам в пределах 1–99-го перцентиля."""
    fig = Figure(figsize=(10, 7), layout="constrained")
    for ax, feature in zip(fig.subplots(2, 2).flat, NUMERIC_FEATURES, strict=True):
        low, high = df[feature].quantile(CLIP_QUANTILES)
        bins = min(40, int(high - low))
        for code, label in CLASS_LABELS.items():
            values = df.loc[df[TARGET] == code, feature]
            ax.hist(
                values, bins=bins, range=(low, high), density=True, alpha=0.5, color=CLASS_COLORS[code], label=label
            )
        ax.set_title(f"{feature}: медиана {df[feature].median():,.0f}, максимум {df[feature].max():,.0f}")
        ax.set_yticks([])
    fig.legend(*fig.axes[0].get_legend_handles_labels(), loc="outside upper right")
    fig.suptitle("Распределения признаков по классам (1–99-й перцентиль)")
    return fig


def plot_default_rate_by_pay1(df: pd.DataFrame) -> Figure:
    """Строит долю дефолтов по статусу платежа в последнем месяце с числом клиентов в каждом статусе."""
    stats = df.groupby("PAY_1")[TARGET].agg(["size", "mean"])
    fig = Figure(figsize=(9, 4.5), layout="constrained")
    ax = fig.subplots()
    bars = ax.bar(stats.index.astype(str), stats["mean"].to_numpy(), color="tab:red")
    ax.bar_label(bars, labels=[f"{rate:.0%}\nn={size}" for size, rate in stats.itertuples(index=False)], fontsize=8)
    ax.axhline(df[TARGET].mean(), color="gray", linestyle="--", label=f"в среднем {df[TARGET].mean():.1%}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("PAY_1: -2, -1, 0 — без просрочки, 1–8 — просрочка в месяцах")
    ax.set_ylabel("Доля дефолтов")
    ax.set_title("Доля дефолтов по статусу платежа в последнем месяце")
    ax.legend(loc="upper left")
    return fig


def save_figures(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Сохраняет все графики анализа в PNG и возвращает пути к файлам."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "class_balance.png": plot_class_balance(df),
        "distributions.png": plot_distributions(df),
        "default_rate_by_pay1.png": plot_default_rate_by_pay1(df),
    }
    paths = []
    for name, fig in figures.items():
        path = out_dir / name
        fig.savefig(path, dpi=100)
        paths.append(path)
    return paths


def main() -> None:
    """Выполняет стадию EDA: строит графики по обучающей выборке."""
    paths = save_figures(pd.read_csv(TRAIN_PATH), FIGURES_DIR)
    print(f"EDA: figures={len(paths)}, dir={FIGURES_DIR.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
