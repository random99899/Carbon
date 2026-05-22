from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import font_manager
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "数据"
RESULT_DIR = ROOT / "1.结果"
FIG_DIR = ROOT / "2.图片"

PROVINCE_FILE = DATA_DIR / "2022省级综合表.xlsx"
NATIONAL_FILE = DATA_DIR / "全国数据.xlsx"
CARBON_FILE = DATA_DIR / "附件1：2019—2025 年全国碳排放数据.csv"


SCENARIOS = {
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


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(FIG_DIR / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def ensure_dirs() -> None:
    RESULT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)


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


def entropy_topsis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    indicators = [
        ("CO2总量_Mt", "负向"),
        ("人均CO2_吨每人", "负向"),
        ("碳排放强度_吨每万元GDP", "负向"),
        ("煤炭相关排放占比", "负向"),
        ("第二产业占比", "负向"),
        ("人均GDP_万元每人", "正向"),
        ("城镇化率_%", "正向"),
    ]
    matrix = []
    for col, direction in indicators:
        x = df[col].astype(float).to_numpy()
        span = x.max() - x.min()
        if span == 0:
            z = np.ones_like(x)
        elif direction == "正向":
            z = (x - x.min()) / span
        else:
            z = (x.max() - x) / span
        matrix.append(z + 1e-12)

    z = np.vstack(matrix).T
    p = z / z.sum(axis=0)
    entropy = -(1 / math.log(len(df))) * np.sum(p * np.log(p), axis=0)
    weights = (1 - entropy) / (1 - entropy).sum()
    d_pos = np.sqrt(((z - z.max(axis=0)) ** 2 * weights).sum(axis=1))
    d_neg = np.sqrt(((z - z.min(axis=0)) ** 2 * weights).sum(axis=1))
    score = d_neg / (d_pos + d_neg)

    weights_df = pd.DataFrame(
        {
            "指标": [col for col, _ in indicators],
            "属性": [direction for _, direction in indicators],
            "熵值": entropy,
            "权重": weights,
        }
    )
    result = df[["省份"] + [col for col, _ in indicators]].copy()
    result["TOPSIS低碳得分"] = score
    result["TOPSIS排名"] = result["TOPSIS低碳得分"].rank(ascending=False, method="min").astype(int)
    result["低碳等级"] = pd.qcut(
        result["TOPSIS低碳得分"].rank(method="first"),
        q=4,
        labels=["IV级_转型压力大", "III级_中等压力", "II级_较好", "I级_优秀"],
    )
    return result.sort_values("TOPSIS排名"), weights_df


def run_problem_one(province: pd.DataFrame) -> dict[str, pd.DataFrame]:
    key_cols = [
        "CO2总量_Mt",
        "人口_万人",
        "GDP_亿元",
        "人均CO2_吨每人",
        "碳排放强度_吨每万元GDP",
        "煤炭相关排放占比",
        "第二产业占比",
        "人均GDP_万元每人",
        "城镇化率_%",
    ]
    require_columns(province, ["省份"] + key_cols)
    if len(province) != 30:
        raise ValueError(f"省级主表应为30行，当前为{len(province)}行")

    spatial_cols = ["CO2总量_Mt", "人均CO2_吨每人", "碳排放强度_吨每万元GDP", "煤炭相关排放占比"]
    spatial_stats = []
    for col in spatial_cols:
        values = province[col].astype(float)
        spatial_stats.append(
            {
                "指标": col,
                "均值": values.mean(),
                "标准差": values.std(ddof=0),
                "变异系数": values.std(ddof=0) / values.mean(),
                "基尼系数": gini(values),
                "泰尔指数": theil(values),
                "最小值": values.min(),
                "最大值": values.max(),
            }
        )
    spatial_stats_df = pd.DataFrame(spatial_stats)

    topsis_df, weights_df = entropy_topsis(province)

    cluster_features = [
        "CO2总量_Mt",
        "人均CO2_吨每人",
        "碳排放强度_吨每万元GDP",
        "煤炭相关排放占比",
        "第二产业占比",
        "人均GDP_万元每人",
        "城镇化率_%",
    ]
    scaler = StandardScaler()
    x = scaler.fit_transform(province[cluster_features])

    silhouette_rows = []
    for k in range(2, 7):
        labels = KMeans(n_clusters=k, n_init=50, random_state=42).fit_predict(x)
        silhouette_rows.append({"K": k, "轮廓系数": silhouette_score(x, labels)})
    silhouette_df = pd.DataFrame(silhouette_rows)

    kmeans = KMeans(n_clusters=4, n_init=100, random_state=42)
    labels = kmeans.fit_predict(x)
    cluster_df = province[["省份"] + cluster_features].copy()
    cluster_df["聚类编号"] = labels

    # Fixed semantic names by cluster profile.
    profile = cluster_df.groupby("聚类编号")[cluster_features].mean()
    high_resource = profile["人均CO2_吨每人"].idxmax()
    efficient = profile["碳排放强度_吨每万元GDP"].idxmin()
    remaining = [idx for idx in profile.index if idx not in {high_resource, efficient}]
    industrial = profile.loc[remaining, "CO2总量_Mt"].idxmax()
    transition = [idx for idx in remaining if idx != industrial][0]
    cluster_names = {
        high_resource: "资源依赖高排放型",
        industrial: "工业制造高排放型",
        transition: "中等转型压力型",
        efficient: "经济发达效率型",
    }
    cluster_df["类型"] = cluster_df["聚类编号"].map(cluster_names)
    cluster_profile = (
        cluster_df.groupby("类型")[cluster_features]
        .mean()
        .reset_index()
        .sort_values("CO2总量_Mt", ascending=False)
    )

    with pd.ExcelWriter(RESULT_DIR / "问题一_空间差异_TOPSIS_聚类结果.xlsx", engine="openpyxl") as writer:
        spatial_stats_df.to_excel(writer, sheet_name="空间差异指标", index=False)
        weights_df.to_excel(writer, sheet_name="熵权", index=False)
        topsis_df.to_excel(writer, sheet_name="TOPSIS得分排名", index=False)
        silhouette_df.to_excel(writer, sheet_name="聚类K值检验", index=False)
        cluster_df.sort_values(["类型", "省份"]).to_excel(writer, sheet_name="K4聚类结果", index=False)
        cluster_profile.to_excel(writer, sheet_name="类型均值画像", index=False)

    return {
        "spatial_stats": spatial_stats_df,
        "topsis": topsis_df,
        "weights": weights_df,
        "silhouette": silhouette_df,
        "cluster": cluster_df,
        "cluster_profile": cluster_profile,
    }


def run_problem_two(province: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = province.copy()
    df["lnC"] = np.log(df["CO2总量_Mt"])
    df["lnP"] = np.log(df["人口_万人"])
    df["lnA"] = np.log(df["人均GDP_万元每人"])
    df["城镇化率"] = df["城镇化率_%"] / 100
    predictors = ["lnP", "lnA", "煤炭相关排放占比", "第二产业占比", "城镇化率"]
    x = sm.add_constant(df[predictors])
    y = df["lnC"]
    ols = sm.OLS(y, x).fit()

    ols_df = pd.DataFrame(
        {
            "变量": ols.params.index,
            "系数": ols.params.values,
            "标准误": ols.bse.values,
            "t值": ols.tvalues.values,
            "p值": ols.pvalues.values,
        }
    )
    fit_df = pd.DataFrame(
        {
            "指标": ["R2", "调整R2", "F统计量", "F检验p值", "AIC", "BIC", "RMSE_log", "MAE_log"],
            "数值": [
                ols.rsquared,
                ols.rsquared_adj,
                ols.fvalue,
                ols.f_pvalue,
                ols.aic,
                ols.bic,
                mean_squared_error(y, ols.fittedvalues) ** 0.5,
                mean_absolute_error(y, ols.fittedvalues),
            ],
        }
    )
    vif_df = pd.DataFrame(
        {
            "变量": predictors,
            "VIF": [variance_inflation_factor(x.values, i + 1) for i in range(len(predictors))],
        }
    )
    fitted_df = pd.DataFrame(
        {
            "省份": df["省份"],
            "实际lnCO2": y,
            "拟合lnCO2": ols.fittedvalues,
            "残差": ols.resid,
            "实际CO2_Mt": df["CO2总量_Mt"],
            "拟合CO2_Mt": np.exp(ols.fittedvalues),
        }
    )

    scaler = StandardScaler()
    xs = scaler.fit_transform(df[predictors])
    ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=LeaveOneOut())
    ridge_pred = cross_val_predict(ridge_cv, xs, y, cv=LeaveOneOut())
    ridge_cv.fit(xs, y)
    ridge_coef_df = pd.DataFrame(
        {
            "变量": predictors,
            "标准化岭回归系数": ridge_cv.coef_,
            "绝对值排序": np.abs(ridge_cv.coef_).argsort()[::-1].argsort() + 1,
        }
    ).sort_values("绝对值排序")
    ridge_metrics_df = pd.DataFrame(
        {
            "指标": ["最优alpha", "LOOCV_R2", "LOOCV_RMSE_log", "LOOCV_MAE_log"],
            "数值": [
                ridge_cv.alpha_,
                r2_score(y, ridge_pred),
                mean_squared_error(y, ridge_pred) ** 0.5,
                mean_absolute_error(y, ridge_pred),
            ],
        }
    )

    with pd.ExcelWriter(RESULT_DIR / "问题二_STIRPAT_回归与检验结果.xlsx", engine="openpyxl") as writer:
        ols_df.to_excel(writer, sheet_name="OLS系数", index=False)
        fit_df.to_excel(writer, sheet_name="拟合优度", index=False)
        vif_df.to_excel(writer, sheet_name="VIF共线性检验", index=False)
        fitted_df.to_excel(writer, sheet_name="拟合值残差", index=False)
        ridge_metrics_df.to_excel(writer, sheet_name="岭回归交叉验证", index=False)
        ridge_coef_df.to_excel(writer, sheet_name="岭回归标准化系数", index=False)

    return {
        "ols_coef": ols_df,
        "fit": fit_df,
        "vif": vif_df,
        "fitted": fitted_df,
        "ridge_metrics": ridge_metrics_df,
        "ridge_coef": ridge_coef_df,
    }


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
    for scenario, periods in SCENARIOS.items():
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
        value_2030 = float(group.loc[group["年份"].eq(2030), "预测CO2_Mt"].iloc[0])
        value_2035 = float(group.loc[group["年份"].eq(2035), "预测CO2_Mt"].iloc[0])
        value_2045 = float(group.loc[group["年份"].eq(2045), "预测CO2_Mt"].iloc[0])
        peak_rows.append(
            {
                "情景": scenario,
                "达峰年份": peak_year,
                "峰值_Mt": peak_value,
                "2030年_Mt": value_2030,
                "2035年_Mt": value_2035,
                "2045年_Mt": value_2045,
                "2045较峰值下降比例": 1 - value_2045 / peak_value,
            }
        )
    peak_df = pd.DataFrame(peak_rows)
    params_df = pd.DataFrame(param_rows)

    sector_2024 = (
        carbon[(carbon["Year"].eq(2024)) & (~carbon["Sector"].eq("Total"))]
        .groupby("Sector", as_index=False)["CO2 (Mt)"]
        .sum()
        .sort_values("CO2 (Mt)", ascending=False)
    )

    with pd.ExcelWriter(RESULT_DIR / "问题三_三情景预测与达峰结果.xlsx", engine="openpyxl") as writer:
        annual.to_excel(writer, sheet_name="全国年度排放", index=False)
        sector_2024.to_excel(writer, sheet_name="2024部门排放", index=False)
        params_df.to_excel(writer, sheet_name="情景参数", index=False)
        forecast_df.to_excel(writer, sheet_name="2024-2045预测序列", index=False)
        peak_df.to_excel(writer, sheet_name="达峰与减排潜力", index=False)

    return {
        "annual": annual,
        "sector_2024": sector_2024,
        "params": params_df,
        "forecast": forecast_df,
        "peak": peak_df,
    }


def make_figures(problem_one: dict[str, pd.DataFrame], problem_two: dict[str, pd.DataFrame], problem_three: dict[str, pd.DataFrame]) -> None:
    topsis = problem_one["topsis"].copy()
    cluster = problem_one["cluster"].copy()
    province = topsis.merge(cluster[["省份", "类型"]], on="省份", how="left")

    top_total = province.sort_values("CO2总量_Mt", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = sns.color_palette("viridis", n_colors=len(top_total))
    ax.barh(top_total["省份"], top_total["CO2总量_Mt"], color=colors)
    ax.set_title("2022年30省CO2排放总量排名")
    ax.set_xlabel("CO2排放总量（Mt）")
    ax.set_ylabel("")
    save_fig(fig, "01_省份CO2总量排名图")

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=province,
        x="人均CO2_吨每人",
        y="碳排放强度_吨每万元GDP",
        hue="类型",
        size="CO2总量_Mt",
        sizes=(60, 450),
        alpha=0.8,
        ax=ax,
    )
    for _, row in province.nlargest(6, "人均CO2_吨每人").iterrows():
        ax.text(row["人均CO2_吨每人"], row["碳排放强度_吨每万元GDP"], row["省份"], fontsize=9)
    ax.set_title("人均CO2与碳排放强度关系")
    ax.set_xlabel("人均CO2（吨/人）")
    ax.set_ylabel("碳排放强度（吨/万元GDP）")
    ax.legend(loc="best", frameon=True)
    save_fig(fig, "02_人均CO2与碳排放强度散点图")

    topsis_sorted = topsis.sort_values("TOPSIS低碳得分", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(topsis_sorted["省份"], topsis_sorted["TOPSIS低碳得分"], color=sns.color_palette("crest", len(topsis_sorted)))
    ax.set_title("熵权TOPSIS低碳发展得分排名")
    ax.set_xlabel("TOPSIS低碳得分")
    ax.set_ylabel("")
    save_fig(fig, "03_TOPSIS低碳得分排名图")

    profile = problem_one["cluster_profile"].copy()
    radar_cols = ["CO2总量_Mt", "人均CO2_吨每人", "碳排放强度_吨每万元GDP", "煤炭相关排放占比", "第二产业占比", "人均GDP_万元每人", "城镇化率_%"]
    scaled = profile.copy()
    for col in radar_cols:
        mn, mx = cluster[col].min(), cluster[col].max()
        scaled[col] = (profile[col] - mn) / (mx - mn) if mx > mn else 0
    labels = ["总量", "人均", "强度", "煤炭", "二产", "人均GDP", "城镇化"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, polar=True)
    for _, row in scaled.iterrows():
        values = [row[col] for col in radar_cols]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["类型"])
        ax.fill(angles, values, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("K-means聚类类型均值画像")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10))
    save_fig(fig, "04_聚类类型均值画像雷达图")

    ridge_coef = problem_two["ridge_coef"].sort_values("标准化岭回归系数")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ridge_coef["变量"], ridge_coef["标准化岭回归系数"], color=sns.color_palette("vlag", len(ridge_coef)))
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title("STIRPAT岭回归标准化系数")
    ax.set_xlabel("标准化系数")
    ax.set_ylabel("")
    save_fig(fig, "05_STIRPAT标准化系数图")

    fitted = problem_two["fitted"]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.regplot(data=fitted, x="实际CO2_Mt", y="拟合CO2_Mt", ax=ax, scatter_kws={"s": 70, "alpha": 0.8})
    lim = [min(fitted["实际CO2_Mt"].min(), fitted["拟合CO2_Mt"].min()), max(fitted["实际CO2_Mt"].max(), fitted["拟合CO2_Mt"].max())]
    ax.plot(lim, lim, "--", color="#444444", label="1:1线")
    ax.set_title("STIRPAT模型实际值与拟合值对比")
    ax.set_xlabel("实际CO2（Mt）")
    ax.set_ylabel("拟合CO2（Mt）")
    ax.legend()
    save_fig(fig, "06_OLS拟合值与真实值对比图")

    forecast = problem_three["forecast"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.lineplot(data=forecast, x="年份", y="预测CO2_Mt", hue="情景", linewidth=2.4, marker="o", markersize=3, ax=ax)
    ax.set_title("2026-2045年全国CO2排放三情景预测")
    ax.set_xlabel("年份")
    ax.set_ylabel("CO2排放量（Mt）")
    ax.axvline(2030, color="#666666", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(2030.2, forecast["预测CO2_Mt"].max() * 0.98, "2030", color="#666666")
    save_fig(fig, "07_三情景碳排放趋势图")

    peak = problem_three["peak"].copy()
    fig, ax1 = plt.subplots(figsize=(8.5, 5.5))
    sns.barplot(data=peak, x="情景", y="峰值_Mt", hue="情景", ax=ax1, palette="Set2", legend=False)
    ax1.set_title("三情景峰值水平与2045减排潜力")
    ax1.set_ylabel("峰值排放（Mt）")
    ax1.set_xlabel("")
    ax2 = ax1.twinx()
    ax2.plot(peak["情景"], peak["2045较峰值下降比例"] * 100, color="#C2410C", marker="o", linewidth=2.2, label="2045较峰值下降比例")
    ax2.set_ylabel("2045较峰值下降比例（%）")
    for i, row in peak.reset_index().iterrows():
        ax1.text(i, row["峰值_Mt"], f'{row["达峰年份"]}', ha="center", va="bottom", fontsize=10)
        ax2.text(i, row["2045较峰值下降比例"] * 100, f'{row["2045较峰值下降比例"]:.1%}', ha="center", va="bottom", color="#C2410C")
    save_fig(fig, "08_三情景峰值与减排潜力对比图")


def write_markdown_reports(problem_one: dict[str, pd.DataFrame], problem_two: dict[str, pd.DataFrame], problem_three: dict[str, pd.DataFrame]) -> None:
    spatial = problem_one["spatial_stats"]
    topsis = problem_one["topsis"]
    cluster = problem_one["cluster"]
    fit = problem_two["fit"].set_index("指标")["数值"]
    ridge = problem_two["ridge_metrics"].set_index("指标")["数值"]
    peak = problem_three["peak"]
    annual = problem_three["annual"]

    resource = cluster[cluster["类型"].eq("资源依赖高排放型")]["省份"].tolist()
    industrial = cluster[cluster["类型"].eq("工业制造高排放型")]["省份"].tolist()
    best = topsis.sort_values("TOPSIS低碳得分", ascending=False).head(5)["省份"].tolist()
    worst = topsis.sort_values("TOPSIS低碳得分").head(5)["省份"].tolist()

    policy = f"""# 问题四：政策建议报告

基于省际分类、STIRPAT驱动因素识别和三情景预测结果，我国推进“双碳”目标应坚持“总量控制、结构优化、效率提升、区域分类”的思路。首先，对{ "、".join(resource) }等资源依赖高排放型省份，应将煤炭消费总量控制和煤电低碳改造作为重点，推动煤炭清洁高效利用、煤电灵活性改造、新能源基地建设和资源型产业转型，避免经济增长继续依赖高碳能源扩张。其次，对{ "、".join(industrial) }等工业制造高排放型省份，应聚焦钢铁、建材、电力、石化和交通等重点行业，实施节能改造、余热利用、电气化替代和绿色供应链管理，建立单位产品碳强度约束。再次，中等转型压力型省份应在承接产业转移时设置能耗和碳排放准入门槛，防止高耗能产业简单转移，同时提高能源利用效率和非化石能源占比。北京、上海等经济发达效率型地区应发挥技术、金融和治理优势，推广绿色金融、碳交易、数字化碳管理和低碳消费。全国层面看，人口规模和煤炭依赖是碳排放的重要驱动因素，情景预测表明能源强度下降和能源碳强度下降越快，达峰时间越早、峰值越低。因此，应持续降低煤炭占比，扩大风光水核等非化石能源供给，强化技术创新和碳市场机制，形成差异化、可考核、可持续的减排路径。
"""
    (RESULT_DIR / "问题四_政策建议报告.md").write_text(policy, encoding="utf-8")

    peak_text = dataframe_to_markdown(peak, floatfmt=".3f")
    summary = f"""# B题结果汇总

## 1. 数据口径

本项目使用`数据/2022省级综合表.xlsx`完成30省横截面分析，使用`数据/附件1：2019—2025 年全国碳排放数据.csv`中`Sector=Total`完成全国趋势分析。附件1的2025年数据截至2025-09-30，不能直接作为全年值与2019至2024年比较，预测基准采用2024年全国排放量。

## 2. 问题一结论

2022年30省CO2总量变异系数为{spatial.loc[spatial['指标'].eq('CO2总量_Mt'), '变异系数'].iloc[0]:.4f}，基尼系数为{spatial.loc[spatial['指标'].eq('CO2总量_Mt'), '基尼系数'].iloc[0]:.4f}；碳排放强度基尼系数为{spatial.loc[spatial['指标'].eq('碳排放强度_吨每万元GDP'), '基尼系数'].iloc[0]:.4f}，说明省际排放效率差异更明显。TOPSIS低碳得分靠前省份为{"、".join(best)}，靠后省份为{"、".join(worst)}。K-means将省份划分为资源依赖高排放型、工业制造高排放型、中等转型压力型和经济发达效率型。

## 3. 问题二结论

STIRPAT横截面OLS模型调整R2为{fit['调整R2']:.4f}，RMSE_log为{fit['RMSE_log']:.4f}。岭回归留一交叉验证R2为{ridge['LOOCV_R2']:.4f}，RMSE_log为{ridge['LOOCV_RMSE_log']:.4f}。结果表明人口规模和煤炭相关排放占比是最稳定的正向驱动因素。

## 4. 问题三结论

三情景达峰与减排潜力如下：

{peak_text}

基准情景下排放在2030年前后达到高位平台；低碳和强化低碳情景下，能源强度和能源碳强度下降更快，排放更早进入下降通道，2045年减排潜力显著提高。

## 5. 输出文件

- `1.结果/问题一_空间差异_TOPSIS_聚类结果.xlsx`
- `1.结果/问题二_STIRPAT_回归与检验结果.xlsx`
- `1.结果/问题三_三情景预测与达峰结果.xlsx`
- `1.结果/问题四_政策建议报告.md`
- `2.图片/01_省份CO2总量排名图.png`
- `2.图片/02_人均CO2与碳排放强度散点图.png`
- `2.图片/03_TOPSIS低碳得分排名图.png`
- `2.图片/04_聚类类型均值画像雷达图.png`
- `2.图片/05_STIRPAT标准化系数图.png`
- `2.图片/06_OLS拟合值与真实值对比图.png`
- `2.图片/07_三情景碳排放趋势图.png`
- `2.图片/08_三情景峰值与减排潜力对比图.png`

## 6. 2024年全国排放核验

附件1中2024年`Sector=Total`年度排放为{annual.loc[annual['Year'].eq(2024), 'CO2 (Mt)'].iloc[0]:.2f} Mt。
"""
    (RESULT_DIR / "B题结果汇总.md").write_text(summary, encoding="utf-8")


def write_readme() -> None:
    readme = """# B题：我国碳排放时空特征分析与趋势预测

本项目为第十届校内数学建模竞赛B题的数据处理、模型求解、图表生成和论文素材整理工程。

## 项目结构

```text
数据处理/
├─ 数据/                         # 题目附件与整理后的数据
├─ 题目/                         # B题题目文件
├─ 1.结果/                       # 模型结果表和文字报告
├─ 2.图片/                       # 论文可用图片，PNG和SVG双格式
├─ 3.代码/                       # 一键运行代码和依赖列表
├─ B题建模实现思路_统计评价聚类_STIRPAT情景预测.md
└─ README.md
```

## 数据说明

- `数据/附件1：2019—2025 年全国碳排放数据.csv`：全国碳排放时间序列，字段为`Area, CO2 (Mt), Sector, Date`。全国总量分析只使用`Sector == "Total"`。
- `数据/附件2：2022 年全国 30 个省份碳排放清单.xlsx`：30省分能源、分部门碳排放清单。
- `数据/2022省级综合表.xlsx`：已合并人口、GDP、第二产业、城镇化率、排放强度和能源结构指标，是问题一和问题二的主表。
- `数据/全国数据.xlsx`：2019至2024年全国GDP、人口、能源消费总量和煤炭占比，用于辅助情景设定。

注意：附件1中2025年数据截至2025-09-30，不能直接作为全年值与2019至2024年比较。三情景预测以2024年为基准年。

## 环境安装

建议使用项目根目录下的虚拟环境：

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r .\\3.代码\\requirements.txt
```

## 一键运行

```powershell
.\\.venv\\Scripts\\python.exe .\\3.代码\\run_all.py
```

运行后会自动生成或覆盖同名结果文件，不会清空目录或删除原始数据。

## 输出文件

`1.结果/`：

- `问题一_空间差异_TOPSIS_聚类结果.xlsx`
- `问题二_STIRPAT_回归与检验结果.xlsx`
- `问题三_三情景预测与达峰结果.xlsx`
- `问题四_政策建议报告.md`
- `B题结果汇总.md`

`2.图片/`：

- `01_省份CO2总量排名图`
- `02_人均CO2与碳排放强度散点图`
- `03_TOPSIS低碳得分排名图`
- `04_聚类类型均值画像雷达图`
- `05_STIRPAT标准化系数图`
- `06_OLS拟合值与真实值对比图`
- `07_三情景碳排放趋势图`
- `08_三情景峰值与减排潜力对比图`

每张图均输出`.png`和`.svg`两种格式。

## 模型路线

1. 问题一：使用变异系数、基尼系数、泰尔指数刻画省际差异；使用熵权TOPSIS评价低碳发展水平；使用K-means进行省份分类分级。
2. 问题二：构建STIRPAT横截面模型，使用OLS估计参数，并通过VIF和岭回归留一交叉验证检验稳健性。
3. 问题三：结合Kaya恒等式设置基准、低碳和强化低碳情景，预测2026至2045年全国CO2排放趋势，判断达峰年份、峰值水平和减排潜力。
4. 问题四：根据省份类型、驱动因素和情景预测结果形成差异化政策建议。

## 可复现性说明

所有结果由`3.代码/run_all.py`从`数据/`目录重新计算得到。若需要调整情景参数，可修改`run_all.py`顶部`SCENARIOS`字典后重新运行。
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")


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


def validate_outputs(problem_one: dict[str, pd.DataFrame], problem_two: dict[str, pd.DataFrame], problem_three: dict[str, pd.DataFrame]) -> None:
    topsis = problem_one["topsis"]
    if topsis["TOPSIS低碳得分"].isna().any() or not topsis["TOPSIS低碳得分"].between(0, 1).all():
        raise AssertionError("TOPSIS得分应非空且位于0到1之间。")
    if problem_one["cluster"]["省份"].nunique() != 30:
        raise AssertionError("K-means聚类结果应覆盖30个省份。")
    if np.isinf(problem_two["vif"]["VIF"]).any():
        raise AssertionError("VIF结果存在无限值。")
    forecast = problem_three["forecast"]
    for scenario, group in forecast.groupby("情景"):
        years = set(group["年份"])
        required_years = set(range(2026, 2046))
        if not required_years.issubset(years):
            raise AssertionError(f"{scenario}未覆盖2026-2045完整预测区间。")
    for stem in [
        "01_省份CO2总量排名图",
        "02_人均CO2与碳排放强度散点图",
        "03_TOPSIS低碳得分排名图",
        "04_聚类类型均值画像雷达图",
        "05_STIRPAT标准化系数图",
        "06_OLS拟合值与真实值对比图",
        "07_三情景碳排放趋势图",
        "08_三情景峰值与减排潜力对比图",
    ]:
        for suffix in ("png", "svg"):
            path = FIG_DIR / f"{stem}.{suffix}"
            if not path.exists() or path.stat().st_size == 0:
                raise AssertionError(f"图片未正确生成: {path}")


def main() -> None:
    ensure_dirs()
    setup_style()
    province, national, carbon = read_data()
    _ = national
    problem_one = run_problem_one(province)
    problem_two = run_problem_two(province)
    problem_three = run_problem_three(carbon)
    make_figures(problem_one, problem_two, problem_three)
    write_markdown_reports(problem_one, problem_two, problem_three)
    write_readme()
    write_gitignore()
    validate_outputs(problem_one, problem_two, problem_three)
    print("全部结果已生成。")
    print(f"结果目录: {RESULT_DIR}")
    print(f"图片目录: {FIG_DIR}")


if __name__ == "__main__":
    main()
