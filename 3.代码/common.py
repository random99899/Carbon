from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager


warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "数据"
RESULT_DIR = ROOT / "1.结果"
FIG_DIR = ROOT / "2.图片"
CODE_DIR = ROOT / "3.代码"

PROBLEM_RESULT_DIRS = {
    "问题一": RESULT_DIR / "问题一",
    "问题二": RESULT_DIR / "问题二",
    "问题三": RESULT_DIR / "问题三",
    "问题四": RESULT_DIR / "问题四",
    "汇总": RESULT_DIR / "汇总",
}

PROBLEM_FIG_DIRS = {
    "问题一": FIG_DIR / "问题一",
    "问题二": FIG_DIR / "问题二",
    "问题三": FIG_DIR / "问题三",
    "问题四": FIG_DIR / "问题四",
}

PROVINCE_FILE = DATA_DIR / "2022省级综合表.xlsx"
NATIONAL_FILE = DATA_DIR / "全国数据.xlsx"
CARBON_FILE = DATA_DIR / "附件1：2019—2025 年全国碳排放数据.csv"


def ensure_dirs() -> None:
    RESULT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    for path in [*PROBLEM_RESULT_DIRS.values(), *PROBLEM_FIG_DIRS.values()]:
        path.mkdir(parents=True, exist_ok=True)


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300


def save_fig(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def read_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    province = pd.read_excel(PROVINCE_FILE, sheet_name="主表")
    national = pd.read_excel(NATIONAL_FILE)
    carbon = pd.read_csv(CARBON_FILE, encoding="utf-8-sig")
    carbon["Date"] = pd.to_datetime(carbon["Date"])
    carbon["Year"] = carbon["Date"].dt.year
    return province, national, carbon


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")
    if df[columns].isna().any().any():
        missing_counts = df[columns].isna().sum()
        raise ValueError(f"关键字段存在缺失值:\n{missing_counts[missing_counts > 0]}")


def gini(values: pd.Series | np.ndarray) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return np.nan
    return float((2 * np.arange(1, n + 1) @ arr) / (n * arr.sum()) - (n + 1) / n)


def theil(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    mean = arr.mean()
    if mean == 0:
        return np.nan
    ratio = arr / mean
    ratio = ratio[ratio > 0]
    return float(np.mean(ratio * np.log(ratio)))


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    headers = [str(col) for col in df.columns]
    rows = []
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(format(float(value), floatfmt))
            else:
                values.append(str(value))
        rows.append(values)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(values) + " |" for values in rows)
    return "\n".join(lines)


def write_gitignore() -> None:
    gitignore = ROOT / ".gitignore"
    desired = [
        ".venv/",
        ".pycache/",
        "__pycache__/",
        "*.pyc",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ipynb_checkpoints/",
        "~$*.xlsx",
        "~$*.docx",
    ]
    existing = gitignore.read_text(encoding="utf-8").splitlines() if gitignore.exists() else []
    merged = existing[:]
    for line in desired:
        if line not in merged:
            merged.append(line)
    gitignore.write_text("\n".join(merged).rstrip() + "\n", encoding="utf-8")


def percent(value: float) -> str:
    return f"{value:.1%}"


def scenario_params() -> dict[str, dict[str, dict[str, float]]]:
    return {
        "基准情景": {
            "2025-2030": {"人口增长率": -0.001, "人均GDP增长率": 0.045, "能源强度变化率": -0.030, "能源碳强度变化率": -0.010},
            "2031-2045": {"人口增长率": -0.001, "人均GDP增长率": 0.030, "能源强度变化率": -0.028, "能源碳强度变化率": -0.012},
        },
        "低碳情景": {
            "2025-2030": {"人口增长率": -0.001, "人均GDP增长率": 0.042, "能源强度变化率": -0.038, "能源碳强度变化率": -0.018},
            "2031-2045": {"人口增长率": -0.001, "人均GDP增长率": 0.028, "能源强度变化率": -0.035, "能源碳强度变化率": -0.020},
        },
        "强化低碳情景": {
            "2025-2030": {"人口增长率": -0.001, "人均GDP增长率": 0.040, "能源强度变化率": -0.048, "能源碳强度变化率": -0.028},
            "2031-2045": {"人口增长率": -0.001, "人均GDP增长率": 0.026, "能源强度变化率": -0.042, "能源碳强度变化率": -0.030},
        },
    }
