from __future__ import annotations

"""
问题一脚本说明

算法与模型：
- 变异系数、基尼系数、泰尔指数：度量30省碳排放空间差异和离散程度。
- 熵权TOPSIS：构建低碳发展综合评价得分，指标包括排放规模、排放效率、能源结构、产业结构和经济发展水平。
- K-means聚类：按标准化后的7个指标将省份划分为4类，并计算K=2至6的轮廓系数作为检验。

使用数据：
- 输入：数据/2022省级综合表.xlsx，工作表“主表”。
- 输出：1.结果/问题一/ 下多个CSV结果文件。
- 输出图片：2.图片/问题一/01至04号图。
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from common import PROBLEM_FIG_DIRS, PROBLEM_RESULT_DIRS, gini, read_data, require_columns, save_fig, setup_style, theil, write_csv


PROBLEM = "问题一"
RESULT_DIR = PROBLEM_RESULT_DIRS[PROBLEM]
FIG_DIR = PROBLEM_FIG_DIRS[PROBLEM]


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
    entropy = -(1 / np.log(len(df))) * np.sum(p * np.log(p), axis=0)
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
    x = StandardScaler().fit_transform(province[cluster_features])
    silhouette_df = pd.DataFrame(
        [
            {
                "K": k,
                "轮廓系数": silhouette_score(x, KMeans(n_clusters=k, n_init=50, random_state=42).fit_predict(x)),
            }
            for k in range(2, 7)
        ]
    )

    labels = KMeans(n_clusters=4, n_init=100, random_state=42).fit_predict(x)
    cluster_df = province[["省份"] + cluster_features].copy()
    cluster_df["聚类编号"] = labels
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

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(spatial_stats_df, RESULT_DIR / "01_空间差异指标.csv")
    write_csv(weights_df, RESULT_DIR / "02_熵权指标权重.csv")
    write_csv(topsis_df, RESULT_DIR / "03_TOPSIS得分排名.csv")
    write_csv(silhouette_df, RESULT_DIR / "04_聚类K值检验.csv")
    write_csv(cluster_df.sort_values(["类型", "省份"]), RESULT_DIR / "05_K4聚类结果.csv")
    write_csv(cluster_profile, RESULT_DIR / "06_类型均值画像.csv")

    return {
        "spatial_stats": spatial_stats_df,
        "topsis": topsis_df,
        "weights": weights_df,
        "silhouette": silhouette_df,
        "cluster": cluster_df,
        "cluster_profile": cluster_profile,
    }


def make_problem_one_figures(problem_one: dict[str, pd.DataFrame]) -> None:
    topsis = problem_one["topsis"].copy()
    cluster = problem_one["cluster"].copy()
    province = topsis.merge(cluster[["省份", "类型"]], on="省份", how="left")

    top_total = province.sort_values("CO2总量_Mt", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top_total["省份"], top_total["CO2总量_Mt"], color=sns.color_palette("viridis", n_colors=len(top_total)))
    ax.set_title("2022年30省CO2排放总量排名")
    ax.set_xlabel("CO2排放总量（Mt）")
    ax.set_ylabel("")
    save_fig(fig, FIG_DIR, "01_省份CO2总量排名图")

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
    save_fig(fig, FIG_DIR, "02_人均CO2与碳排放强度散点图")

    topsis_sorted = topsis.sort_values("TOPSIS低碳得分", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(topsis_sorted["省份"], topsis_sorted["TOPSIS低碳得分"], color=sns.color_palette("crest", len(topsis_sorted)))
    ax.set_title("熵权TOPSIS低碳发展得分排名")
    ax.set_xlabel("TOPSIS低碳得分")
    ax.set_ylabel("")
    save_fig(fig, FIG_DIR, "03_TOPSIS低碳得分排名图")

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
        values = [row[col] for col in radar_cols] + [row[radar_cols[0]]]
        ax.plot(angles, values, linewidth=2, label=row["类型"])
        ax.fill(angles, values, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("K-means聚类类型均值画像")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10))
    save_fig(fig, FIG_DIR, "04_聚类类型均值画像雷达图")


def main() -> None:
    setup_style()
    province, _, _ = read_data()
    result = run_problem_one(province)
    make_problem_one_figures(result)
    print(f"问题一结果已生成: {RESULT_DIR}")
    print(f"问题一图片已生成: {FIG_DIR}")


if __name__ == "__main__":
    main()
