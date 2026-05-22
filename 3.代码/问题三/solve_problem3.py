from __future__ import annotations

"""
问题三脚本说明

算法与模型：
- 全国时间趋势统计：只使用附件1中Sector=Total的全国排放总量，避免Total与部门排放重复相加。
- STIRPAT系数递推预测：使用问题二OLS系数，按驱动变量变化推算全国排放相对变化。
- 达峰研判：逐情景寻找预测序列最大值，输出达峰年份、峰值水平、碳排放强度和减排潜力。

使用数据：
- 输入：数据/附件1：2019—2025 年全国碳排放数据.csv、数据/全国数据.xlsx、问题二STIRPAT系数。
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
import numpy as np
import pandas as pd
import seaborn as sns

from common import PROBLEM_FIG_DIRS, PROBLEM_RESULT_DIRS, read_data, save_fig, scenario_driver_assumptions, setup_style, write_csv
from 问题二.solve_problem2 import run_problem_two


PROBLEM = "问题三"
RESULT_DIR = PROBLEM_RESULT_DIRS[PROBLEM]
FIG_DIR = PROBLEM_FIG_DIRS[PROBLEM]


def _annual_rate(start: float, end: float, year: int, first_year: int = 2025, last_year: int = 2045) -> float:
    if last_year == first_year:
        return start
    ratio = (year - first_year) / (last_year - first_year)
    return start + (end - start) * ratio


def _prepare_national_base(national: pd.DataFrame, province: pd.DataFrame, base_year: int) -> tuple[dict[str, float], pd.DataFrame]:
    national_std = national.rename(
        columns={
            "年  份": "年份",
            "GDP（亿元）": "GDP_亿元",
            "人口（万人）": "人口_万人",
            "能源消费总量(万吨标准煤)": "能源消费总量_万吨标准煤",
            "煤炭占比": "煤炭占比_%",
        }
    ).copy()
    national_std["人均GDP_万元每人"] = national_std["GDP_亿元"] / national_std["人口_万人"]
    national_std["煤炭占比"] = national_std["煤炭占比_%"] / 100
    base = national_std[national_std["年份"].eq(base_year)]
    if base.empty:
        raise ValueError(f"全国数据缺少{base_year}年，无法确定问题三预测基准。")
    base_row = base.iloc[0]
    industry_base = province["第二产业_亿元"].sum() / province["GDP_亿元"].sum()
    urban_base = (province["城镇化率_%"] * province["人口_万人"]).sum() / province["人口_万人"].sum() / 100
    base_values = {
        "年份": base_year,
        "人口_万人": float(base_row["人口_万人"]),
        "GDP_亿元": float(base_row["GDP_亿元"]),
        "人均GDP_万元每人": float(base_row["人均GDP_万元每人"]),
        "煤炭相关排放占比": float(base_row["煤炭占比"]),
        "第二产业占比": float(industry_base),
        "城镇化率": float(urban_base),
    }
    base_table = pd.DataFrame(
        [
            {"变量": "人口_万人", "基准值": base_values["人口_万人"], "来源": "全国数据.xlsx 2024年人口"},
            {"变量": "GDP_亿元", "基准值": base_values["GDP_亿元"], "来源": "全国数据.xlsx 2024年GDP"},
            {"变量": "人均GDP_万元每人", "基准值": base_values["人均GDP_万元每人"], "来源": "GDP/人口"},
            {"变量": "煤炭相关排放占比", "基准值": base_values["煤炭相关排放占比"], "来源": "全国数据.xlsx 2024年煤炭占比"},
            {"变量": "第二产业占比", "基准值": base_values["第二产业占比"], "来源": "2022省级综合表按GDP聚合"},
            {"变量": "城镇化率", "基准值": base_values["城镇化率"], "来源": "2022省级综合表按人口加权"},
        ]
    )
    return base_values, base_table


def _simulate_forecast(
    base_emission: float,
    base_values: dict[str, float],
    coeff: dict[str, float],
    assumptions: dict[str, dict[str, float]],
    base_year: int = 2024,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    param_rows = []
    for scenario, params in assumptions.items():
        emission = base_emission
        population = base_values["人口_万人"]
        affluence = base_values["人均GDP_万元每人"]
        coal_share = base_values["煤炭相关排放占比"]
        industry_share = base_values["第二产业占比"]
        urban_rate = base_values["城镇化率"]
        gdp = population * affluence
        rows.append(
            {
                "情景": scenario,
                "年份": base_year,
                "预测CO2_Mt": emission,
                "预测GDP_亿元": gdp,
                "碳排放强度_吨每万元GDP": emission / gdp * 100,
                "人口_万人": population,
                "人均GDP_万元每人": affluence,
                "煤炭相关排放占比": coal_share,
                "第二产业占比": industry_share,
                "城镇化率": urban_rate,
                "STIRPAT对数增长项": 0.0,
            }
        )
        for year in range(2025, 2046):
            g_population = _annual_rate(params["人口增长率_起始"], params["人口增长率_末期"], year)
            g_affluence = _annual_rate(params["人均GDP增长率_起始"], params["人均GDP增长率_末期"], year)
            d_coal = _annual_rate(params["煤炭占比年变化_起始"], params["煤炭占比年变化_末期"], year)
            d_industry = _annual_rate(params["第二产业占比年变化_起始"], params["第二产业占比年变化_末期"], year)
            d_urban = _annual_rate(params["城镇化率年变化_起始"], params["城镇化率年变化_末期"], year)
            prev_population = population
            prev_affluence = affluence
            prev_coal = coal_share
            prev_industry = industry_share
            prev_urban = urban_rate
            population *= 1 + g_population
            affluence *= 1 + g_affluence
            coal_share = float(np.clip(coal_share + d_coal, 0.05, 0.95))
            industry_share = float(np.clip(industry_share + d_industry, 0.20, 0.60))
            urban_rate = float(np.clip(urban_rate + d_urban, 0.45, 0.90))
            d_ln_population = np.log(population / prev_population)
            d_ln_affluence = np.log(affluence / prev_affluence)
            d_coal_actual = coal_share - prev_coal
            d_industry_actual = industry_share - prev_industry
            d_urban_actual = urban_rate - prev_urban
            log_growth = (
                coeff["lnP"] * d_ln_population
                + coeff["lnA"] * d_ln_affluence
                + coeff["煤炭相关排放占比"] * d_coal_actual
                + coeff["第二产业占比"] * d_industry_actual
                + coeff["城镇化率"] * d_urban_actual
            )
            emission *= float(np.exp(log_growth))
            gdp = population * affluence
            param_rows.append(
                {
                    "情景": scenario,
                    "年份": year,
                    "人口增长率": g_population,
                    "人均GDP增长率": g_affluence,
                    "煤炭占比年变化": d_coal_actual,
                    "第二产业占比年变化": d_industry_actual,
                    "城镇化率年变化": d_urban_actual,
                    "ΔlnP": d_ln_population,
                    "ΔlnA": d_ln_affluence,
                    "STIRPAT对数增长项": log_growth,
                    "排放增长率": np.exp(log_growth) - 1,
                }
            )
            rows.append(
                {
                    "情景": scenario,
                    "年份": year,
                    "预测CO2_Mt": emission,
                    "预测GDP_亿元": gdp,
                    "碳排放强度_吨每万元GDP": emission / gdp * 100,
                    "人口_万人": population,
                    "人均GDP_万元每人": affluence,
                    "煤炭相关排放占比": coal_share,
                    "第二产业占比": industry_share,
                    "城镇化率": urban_rate,
                    "STIRPAT对数增长项": log_growth,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(param_rows)


def _summarize_peak(forecast_df: pd.DataFrame, scenario_order: list[str], base_year: int = 2024) -> pd.DataFrame:
    peak_rows = []
    for scenario, group in forecast_df.groupby("情景"):
        group = group.sort_values("年份")
        idx = group["预测CO2_Mt"].idxmax()
        peak_year = int(group.loc[idx, "年份"])
        peak_value = float(group.loc[idx, "预测CO2_Mt"])
        base_intensity = float(group.loc[group["年份"].eq(base_year), "碳排放强度_吨每万元GDP"].iloc[0])
        intensity_2045 = float(group.loc[group["年份"].eq(2045), "碳排放强度_吨每万元GDP"].iloc[0])
        peak_rows.append(
            {
                "情景": scenario,
                "达峰年份": peak_year,
                "峰值_Mt": peak_value,
                "2030年_Mt": float(group.loc[group["年份"].eq(2030), "预测CO2_Mt"].iloc[0]),
                "2035年_Mt": float(group.loc[group["年份"].eq(2035), "预测CO2_Mt"].iloc[0]),
                "2045年_Mt": float(group.loc[group["年份"].eq(2045), "预测CO2_Mt"].iloc[0]),
                "2045较峰值下降比例": 1 - float(group.loc[group["年份"].eq(2045), "预测CO2_Mt"].iloc[0]) / peak_value,
                "2030年强度_吨每万元GDP": float(group.loc[group["年份"].eq(2030), "碳排放强度_吨每万元GDP"].iloc[0]),
                "2045年强度_吨每万元GDP": intensity_2045,
                "2045较2024强度下降比例": 1 - intensity_2045 / base_intensity,
            }
        )
    peak_df = pd.DataFrame(peak_rows)
    peak_df["情景"] = pd.Categorical(peak_df["情景"], categories=scenario_order, ordered=True)
    peak_df = peak_df.sort_values("情景").reset_index(drop=True)
    peak_df["情景"] = peak_df["情景"].astype(str)
    return peak_df


def _copy_assumptions(assumptions: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {scenario: values.copy() for scenario, values in assumptions.items()}


def _single_factor_sensitivity(
    base_emission: float,
    base_values: dict[str, float],
    coeff: dict[str, float],
    assumptions: dict[str, dict[str, float]],
    baseline_peak: pd.DataFrame,
) -> pd.DataFrame:
    variable_keys = {
        "人口增长率": ["人口增长率_起始", "人口增长率_末期"],
        "人均GDP增长率": ["人均GDP增长率_起始", "人均GDP增长率_末期"],
        "煤炭占比下降速度": ["煤炭占比年变化_起始", "煤炭占比年变化_末期"],
        "第二产业占比下降速度": ["第二产业占比年变化_起始", "第二产业占比年变化_末期"],
        "城镇化率年变化": ["城镇化率年变化_起始", "城镇化率年变化_末期"],
    }
    perturbations = [-0.20, -0.10, 0.10, 0.20]
    baseline = baseline_peak.set_index("情景")
    rows = []
    scenario_order = list(assumptions.keys())
    for scenario in scenario_order:
        for variable, keys in variable_keys.items():
            for perturbation in perturbations:
                perturbed = _copy_assumptions(assumptions)
                for key in keys:
                    perturbed[scenario][key] *= 1 + perturbation
                forecast_df, _ = _simulate_forecast(base_emission, base_values, coeff, perturbed)
                peak = _summarize_peak(forecast_df, scenario_order)
                result = peak.set_index("情景").loc[scenario]
                base = baseline.loc[scenario]
                rows.append(
                    {
                        "情景": scenario,
                        "扰动变量": variable,
                        "扰动比例": perturbation,
                        "达峰年份": result["达峰年份"],
                        "达峰年份变化": result["达峰年份"] - base["达峰年份"],
                        "峰值_Mt": result["峰值_Mt"],
                        "峰值变化_Mt": result["峰值_Mt"] - base["峰值_Mt"],
                        "2045年_Mt": result["2045年_Mt"],
                        "2045变化_Mt": result["2045年_Mt"] - base["2045年_Mt"],
                        "2045较峰值下降比例": result["2045较峰值下降比例"],
                    }
                )
    return pd.DataFrame(rows)


def _coefficient_sensitivity(
    base_emission: float,
    base_values: dict[str, float],
    coeff: dict[str, float],
    assumptions: dict[str, dict[str, float]],
    baseline_peak: pd.DataFrame,
) -> pd.DataFrame:
    variable_labels = {
        "lnP": "人口规模系数",
        "lnA": "人均GDP系数",
        "煤炭相关排放占比": "煤炭占比系数",
        "第二产业占比": "第二产业占比系数",
        "城镇化率": "城镇化率系数",
    }
    perturbations = [-0.10, 0.10]
    baseline = baseline_peak.set_index("情景")
    rows = []
    scenario_order = list(assumptions.keys())
    for coeff_name, label in variable_labels.items():
        for perturbation in perturbations:
            perturbed_coeff = coeff.copy()
            perturbed_coeff[coeff_name] *= 1 + perturbation
            forecast_df, _ = _simulate_forecast(base_emission, base_values, perturbed_coeff, assumptions)
            peak = _summarize_peak(forecast_df, scenario_order)
            for scenario in scenario_order:
                result = peak.set_index("情景").loc[scenario]
                base = baseline.loc[scenario]
                rows.append(
                    {
                        "情景": scenario,
                        "扰动系数": coeff_name,
                        "系数说明": label,
                        "扰动比例": perturbation,
                        "达峰年份": result["达峰年份"],
                        "达峰年份变化": result["达峰年份"] - base["达峰年份"],
                        "峰值_Mt": result["峰值_Mt"],
                        "峰值变化_Mt": result["峰值_Mt"] - base["峰值_Mt"],
                        "2045年_Mt": result["2045年_Mt"],
                        "2045变化_Mt": result["2045年_Mt"] - base["2045年_Mt"],
                        "2045较峰值下降比例": result["2045较峰值下降比例"],
                    }
                )
    return pd.DataFrame(rows)


def run_problem_three(
    carbon: pd.DataFrame,
    national: pd.DataFrame,
    province: pd.DataFrame,
    problem_two: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    total = carbon[carbon["Sector"].eq("Total")].copy()
    annual = total.groupby("Year", as_index=False)["CO2 (Mt)"].sum()
    days = total.groupby("Year")["Date"].nunique().reset_index(name="记录天数")
    annual = annual.merge(days, on="Year", how="left")
    annual["是否完整年份"] = annual["记录天数"] >= 365
    base_year = 2024
    if base_year not in annual["Year"].values:
        raise ValueError("附件1缺少2024年Sector=Total数据，无法确定预测基准。")
    base_emission = float(annual.loc[annual["Year"].eq(base_year), "CO2 (Mt)"].iloc[0])
    base_values, base_table = _prepare_national_base(national, province, base_year)
    if problem_two is None:
        problem_two = run_problem_two(province)
    coeff = problem_two["ols_coef"].set_index("变量")["系数"].to_dict()
    required = ["lnP", "lnA", "煤炭相关排放占比", "第二产业占比", "城镇化率"]
    missing = [name for name in required if name not in coeff]
    if missing:
        raise ValueError(f"问题二OLS系数缺少变量: {missing}")

    coeff_rows = pd.DataFrame(
        [{"变量": key, "STIRPAT_OLS系数": coeff[key], "预测中含义": "驱动变量变化对ln(CO2)的边际影响"} for key in required]
    )
    assumptions = scenario_driver_assumptions()
    scenario_order = list(assumptions.keys())
    forecast_df, params_df = _simulate_forecast(base_emission, base_values, coeff, assumptions, base_year)
    peak_df = _summarize_peak(forecast_df, scenario_order, base_year)
    single_factor_sensitivity = _single_factor_sensitivity(base_emission, base_values, coeff, assumptions, peak_df)
    coefficient_sensitivity = _coefficient_sensitivity(base_emission, base_values, coeff, assumptions, peak_df)
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
    write_csv(coeff_rows, RESULT_DIR / "06_STIRPAT递推系数.csv")
    write_csv(base_table, RESULT_DIR / "07_驱动变量基准值.csv")
    write_csv(single_factor_sensitivity, RESULT_DIR / "08_单因素灵敏度分析.csv")
    write_csv(coefficient_sensitivity, RESULT_DIR / "09_STIRPAT系数灵敏度分析.csv")

    return {
        "annual": annual,
        "sector_2024": sector_2024,
        "params": params_df,
        "forecast": forecast_df,
        "peak": peak_df,
        "stirpat_coeff": coeff_rows,
        "driver_base": base_table,
        "single_factor_sensitivity": single_factor_sensitivity,
        "coefficient_sensitivity": coefficient_sensitivity,
    }


def make_problem_three_figures(problem_three: dict[str, pd.DataFrame]) -> None:
    forecast = problem_three["forecast"]
    peak = problem_three["peak"].copy()
    scenario_order = ["基准情景", "低碳情景", "强化低碳情景"]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    plot_data = forecast[forecast["年份"].between(2024, 2045)].rename(columns={"预测CO2_Mt": "预测CO²排放量（Mt）"})
    sns.lineplot(
        data=plot_data,
        x="年份",
        y="预测CO²排放量（Mt）",
        hue="情景",
        hue_order=scenario_order,
        linewidth=2.4,
        marker="o",
        markersize=3,
        ax=ax,
    )
    color_map = {text.get_text(): handle.get_color() for text, handle in zip(ax.legend_.texts, ax.legend_.legend_handles)}
    y_min = plot_data["预测CO²排放量（Mt）"].min() * 0.97
    ax.set_ylim(y_min, plot_data["预测CO²排放量（Mt）"].max() * 1.03)
    peak["情景"] = pd.Categorical(peak["情景"], categories=scenario_order, ordered=True)
    peak = peak.sort_values("情景").reset_index(drop=True)
    peak["情景"] = peak["情景"].astype(str)
    for _, row in peak.iterrows():
        color = color_map.get(row["情景"], "#555555")
        ax.vlines(row["达峰年份"], y_min, row["峰值_Mt"], color=color, linestyle="--", linewidth=1.2, alpha=0.75)
        ax.scatter(row["达峰年份"], row["峰值_Mt"], color=color, edgecolor="white", linewidth=0.8, s=58, zorder=5)
    peak_year_colors = {int(row["达峰年份"]): color_map.get(row["情景"], "#555555") for _, row in peak.iterrows()}
    ax.set_xticks([2024, 2025, 2026, 2030, 2032, 2035, 2040, 2045])
    ax.tick_params(axis="x", labelsize=10, pad=6)
    for label in ax.get_xticklabels():
        year = int(float(label.get_text()))
        if year in peak_year_colors:
            label.set_color(peak_year_colors[year])
    ax.legend(title="情景", loc="upper left", bbox_to_anchor=(1.01, 1), frameon=True)
    ax.set_title("2024-2045年全国CO²排放三情景STIRPAT预测")
    ax.set_xlabel("年份")
    ax.set_ylabel("CO²排放量（Mt）")
    save_fig(fig, FIG_DIR, "07_三情景碳排放趋势图")

    peak = problem_three["peak"].copy()
    peak["情景"] = pd.Categorical(peak["情景"], categories=scenario_order, ordered=True)
    peak = peak.sort_values("情景").reset_index(drop=True)
    peak["情景"] = peak["情景"].astype(str)
    fig, ax1 = plt.subplots(figsize=(8.5, 5.5))
    scenario_palette = {"基准情景": "#4C72B0", "低碳情景": "#DD8452", "强化低碳情景": "#55A868"}
    sns.barplot(data=peak, x="情景", y="峰值_Mt", hue="情景", order=scenario_order, ax=ax1, palette=scenario_palette, legend=False)
    for patch in ax1.patches:
        current_width = patch.get_width()
        new_width = current_width * 0.72
        patch.set_x(patch.get_x() + (current_width - new_width) / 2)
        patch.set_width(new_width)
    ax1.set_title("三情景峰值水平与2045减排潜力")
    ax1.set_ylabel("峰值排放（Mt）")
    ax1.set_xlabel("")
    left_ticks = [10000, 10600, 11200, 11800, 12400, 13000]
    right_ticks = [0, 9, 18, 27, 36, 45]
    ax1.set_ylim(10000, 13000)
    ax1.set_yticks(left_ticks)
    ax1.grid(True, axis="y")
    ax2 = ax1.twinx()
    line_color = "#4F46E5"
    ax2.set_ylim(0, 45)
    ax2.set_yticks(right_ticks)
    ax2.grid(False)
    ax2.plot(peak["情景"], peak["2045较峰值下降比例"] * 100, color=line_color, marker="o", linewidth=2.2)
    ax2.set_ylabel("2045较峰值下降比例（%）")
    for i, row in peak.reset_index().iterrows():
        ax1.text(i, row["峰值_Mt"], f'{row["达峰年份"]}', ha="center", va="bottom", fontsize=10)
        ratio = row["2045较峰值下降比例"] * 100
        offset = 1.8 if i != 2 else -2.2
        ax2.text(
            i,
            ratio + offset,
            f'{row["2045较峰值下降比例"]:.1%}',
            ha="center",
            va="bottom" if offset > 0 else "top",
            color="#312E81",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
    save_fig(fig, FIG_DIR, "08_三情景峰值与减排潜力对比图")

    single = problem_three["single_factor_sensitivity"].copy()
    single_plot = single[(single["情景"].eq("基准情景")) & (single["扰动比例"].isin([-0.20, 0.20]))].copy()
    single_plot["扰动方向"] = single_plot["扰动比例"].map({-0.20: "-20%", 0.20: "+20%"})
    order = (
        single_plot.groupby("扰动变量")["2045变化_Mt"]
        .apply(lambda s: s.abs().max())
        .sort_values(ascending=True)
        .index.tolist()
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    sns.barplot(
        data=single_plot,
        y="扰动变量",
        x="2045变化_Mt",
        hue="扰动方向",
        order=order,
        palette={"-20%": "#60A5FA", "+20%": "#F97316"},
        ax=ax,
    )
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title("单因素灵敏度：基准情景2045排放影响")
    ax.set_xlabel("2045年CO²排放变化量（Mt）")
    ax.set_ylabel("")
    ax.legend(title="扰动比例", frameon=True)
    save_fig(fig, FIG_DIR, "09_单因素灵敏度_2045排放影响图")

    coef = problem_three["coefficient_sensitivity"].copy()
    coef_plot = coef[(coef["情景"].eq("基准情景")) & (coef["扰动比例"].isin([-0.10, 0.10]))].copy()
    coef_plot["扰动方向"] = coef_plot["扰动比例"].map({-0.10: "-10%", 0.10: "+10%"})
    order = (
        coef_plot.groupby("系数说明")["2045变化_Mt"]
        .apply(lambda s: s.abs().max())
        .sort_values(ascending=True)
        .index.tolist()
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    sns.barplot(
        data=coef_plot,
        y="系数说明",
        x="2045变化_Mt",
        hue="扰动方向",
        order=order,
        palette={"-10%": "#93C5FD", "+10%": "#8B5CF6"},
        ax=ax,
    )
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title("STIRPAT系数灵敏度：基准情景2045排放影响")
    ax.set_xlabel("2045年CO²排放变化量（Mt）")
    ax.set_ylabel("")
    ax.legend(title="扰动比例", frameon=True)
    save_fig(fig, FIG_DIR, "10_STIRPAT系数灵敏度_2045排放影响图")


def main() -> None:
    setup_style()
    province, national, carbon = read_data()
    problem_two = run_problem_two(province)
    result = run_problem_three(carbon, national, province, problem_two)
    make_problem_three_figures(result)
    print(f"问题三结果已生成: {RESULT_DIR}")
    print(f"问题三图片已生成: {FIG_DIR}")


if __name__ == "__main__":
    main()
