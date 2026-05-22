from __future__ import annotations

"""
问题三脚本说明

算法与模型：
- 全国时间趋势统计：只使用附件1中Sector=Total的全国排放总量，避免Total与部门排放重复相加。
- Kaya-STIRPAT情景递推：C(t+1)=C(t)*(1+gP)*(1+gA)*(1+gEI)*(1+gCI)。
- 达峰研判：逐情景寻找预测序列最大值，输出达峰年份、峰值水平、2030/2035/2045排放和减排潜力。

使用数据：
- 输入：数据/附件1：2019—2025 年全国碳排放数据.csv。
- 输出：1.结果/问题三/ 下多个CSV结果文件。
- 输出图片：2.图片/问题三/07_三情景碳排放趋势图、08_三情景峰值与减排潜力对比图。
"""

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT.parent
sys.pycache_prefix = str(ROOT / ".pycache")
sys.path.append(str(CODE_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import PROBLEM_FIG_DIRS, PROBLEM_RESULT_DIRS, read_data, save_fig, scenario_params, setup_style, write_csv


PROBLEM = "问题三"
RESULT_DIR = PROBLEM_RESULT_DIRS[PROBLEM]
FIG_DIR = PROBLEM_FIG_DIRS[PROBLEM]


def run_problem_three(carbon: pd.DataFrame) -> dict[str, pd.DataFrame]:
    total = carbon[carbon["Sector"].eq("Total")].copy()
    annual = total.groupby("Year", as_index=False)["CO2 (Mt)"].sum()
    days = total.groupby("Year")["Date"].nunique().reset_index(name="记录天数")
    annual = annual.merge(days, on="Year", how="left")
    annual["是否完整年份"] = annual["记录天数"] >= 365
    if 2024 not in annual["Year"].values:
        raise ValueError("附件1缺少2024年Sector=Total数据，无法确定预测基准。")
    base_emission = float(annual.loc[annual["Year"].eq(2024), "CO2 (Mt)"].iloc[0])

    rows = []
    param_rows = []
    for scenario, periods in scenario_params().items():
        emission = base_emission
        rows.append({"情景": scenario, "年份": 2024, "预测CO2_Mt": emission})
        for period_name, values in periods.items():
            param_rows.append({"情景": scenario, "阶段": period_name, **values})
        for year in range(2025, 2046):
            values = periods["2025-2030"] if year <= 2030 else periods["2031-2045"]
            factor = 1.0
            for rate in values.values():
                factor *= 1 + rate
            emission *= factor
            rows.append({"情景": scenario, "年份": year, "预测CO2_Mt": emission})

    forecast_df = pd.DataFrame(rows)
    peak_rows = []
    for scenario, group in forecast_df.groupby("情景"):
        idx = group["预测CO2_Mt"].idxmax()
        peak_year = int(forecast_df.loc[idx, "年份"])
        peak_value = float(forecast_df.loc[idx, "预测CO2_Mt"])
        peak_rows.append(
            {
                "情景": scenario,
                "达峰年份": peak_year,
                "峰值_Mt": peak_value,
                "2030年_Mt": float(group.loc[group["年份"].eq(2030), "预测CO2_Mt"].iloc[0]),
                "2035年_Mt": float(group.loc[group["年份"].eq(2035), "预测CO2_Mt"].iloc[0]),
                "2045年_Mt": float(group.loc[group["年份"].eq(2045), "预测CO2_Mt"].iloc[0]),
                "2045较峰值下降比例": 1 - float(group.loc[group["年份"].eq(2045), "预测CO2_Mt"].iloc[0]) / peak_value,
            }
        )
    peak_df = pd.DataFrame(peak_rows)
    params_df = pd.DataFrame([{"情景": s, "阶段": p, **v} for s, periods in scenario_params().items() for p, v in periods.items()])
    sector_2024 = (
        carbon[(carbon["Year"].eq(2024)) & (~carbon["Sector"].eq("Total"))]
        .groupby("Sector", as_index=False)["CO2 (Mt)"]
        .sum()
        .sort_values("CO2 (Mt)", ascending=False)
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(annual, RESULT_DIR / "01_全国年度排放.csv")
    write_csv(sector_2024, RESULT_DIR / "02_2024部门排放.csv")
    write_csv(params_df, RESULT_DIR / "03_情景参数.csv")
    write_csv(forecast_df, RESULT_DIR / "04_2024-2045预测序列.csv")
    write_csv(peak_df, RESULT_DIR / "05_达峰与减排潜力.csv")

    return {"annual": annual, "sector_2024": sector_2024, "params": params_df, "forecast": forecast_df, "peak": peak_df}


def make_problem_three_figures(problem_three: dict[str, pd.DataFrame]) -> None:
    forecast = problem_three["forecast"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.lineplot(data=forecast, x="年份", y="预测CO2_Mt", hue="情景", linewidth=2.4, marker="o", markersize=3, ax=ax)
    ax.set_title("2026-2045年全国CO2排放三情景预测")
    ax.set_xlabel("年份")
    ax.set_ylabel("CO2排放量（Mt）")
    ax.axvline(2030, color="#666666", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(2030.2, forecast["预测CO2_Mt"].max() * 0.98, "2030", color="#666666")
    save_fig(fig, FIG_DIR, "07_三情景碳排放趋势图")

    peak = problem_three["peak"].copy()
    fig, ax1 = plt.subplots(figsize=(8.5, 5.5))
    sns.barplot(data=peak, x="情景", y="峰值_Mt", hue="情景", ax=ax1, palette="Set2", legend=False)
    ax1.set_title("三情景峰值水平与2045减排潜力")
    ax1.set_ylabel("峰值排放（Mt）")
    ax1.set_xlabel("")
    ax2 = ax1.twinx()
    ax2.plot(peak["情景"], peak["2045较峰值下降比例"] * 100, color="#C2410C", marker="o", linewidth=2.2)
    ax2.set_ylabel("2045较峰值下降比例（%）")
    for i, row in peak.reset_index().iterrows():
        ax1.text(i, row["峰值_Mt"], f'{row["达峰年份"]}', ha="center", va="bottom", fontsize=10)
        ax2.text(i, row["2045较峰值下降比例"] * 100, f'{row["2045较峰值下降比例"]:.1%}', ha="center", va="bottom", color="#C2410C")
    save_fig(fig, FIG_DIR, "08_三情景峰值与减排潜力对比图")


def main() -> None:
    setup_style()
    _, _, carbon = read_data()
    result = run_problem_three(carbon)
    make_problem_three_figures(result)
    print(f"问题三结果已生成: {RESULT_DIR}")
    print(f"问题三图片已生成: {FIG_DIR}")


if __name__ == "__main__":
    main()
