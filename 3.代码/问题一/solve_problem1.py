from __future__ import annotations

"""
问题一脚本说明

算法与模型：
- 变异系数、基尼系数、泰尔指数：度量30省碳排放空间差异和离散程度。
- 熵权TOPSIS：构建低碳发展综合评价得分，指标包括排放规模、排放效率、能源结构、产业结构和经济发展水平。
- K-means聚类：按标准化后的7个指标将省份划分为4类，并计算K=2至6的轮廓系数作为检验。

使用数据：
- 输入：数据/2022省级综合表.xlsx，工作表“主表”。
- 辅助边界：3.代码/assets/china_provinces.geojson，用于绘制省级空间分布热力图。
- 输出：1.结果/问题一/ 下多个CSV结果文件。
- 输出图片：2.图片/问题一/01至05号图。
"""

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT.parent
sys.pycache_prefix = str(ROOT / ".pycache")
sys.path.append(str(CODE_ROOT))

import json
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as MplPolygon
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
CHINA_GEOJSON = CODE_ROOT / "assets" / "china_provinces.geojson"


def normalize_province_name(name: str) -> str:
    result = str(name).strip()
    replacements = {
        "特别行政区": "",
        "维吾尔自治区": "",
        "壮族自治区": "",
        "回族自治区": "",
        "自治区": "",
        "省": "",
        "市": "",
    }
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def geometry_to_polygons(geometry: dict) -> list[np.ndarray]:
    polygons = []
    if geometry["type"] == "Polygon":
        coordinates = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        coordinates = geometry["coordinates"]
    else:
        return polygons
    for polygon in coordinates:
        if not polygon:
            continue
        exterior = np.asarray(polygon[0], dtype=float)
        if len(exterior) >= 3:
            polygons.append(exterior)
    return polygons


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
    scatter_data = province.rename(columns={"CO2总量_Mt": "CO²总量（Mt）"})
    sns.scatterplot(
        data=scatter_data,
        x="人均CO2_吨每人",
        y="碳排放强度_吨每万元GDP",
        hue="类型",
        size="CO²总量（Mt）",
        sizes=(120, 900),
        alpha=0.8,
        ax=ax,
    )
    label_offsets = {
        "内蒙古": (-54, -2),
        "宁夏": (18, 6),
        "山西": (26, -4),
        "新疆": (24, 8),
        "河北": (30, -8),
    }
    label_data = province[province["类型"].eq("资源依赖高排放型")]
    for _, row in label_data.iterrows():
        offset = label_offsets.get(row["省份"], (18, 6))
        ax.annotate(
            row["省份"],
            xy=(row["人均CO2_吨每人"], row["碳排放强度_吨每万元GDP"]),
            xytext=offset,
            textcoords="offset points",
            ha="right" if offset[0] < 0 else "left",
            va="center",
            fontsize=9,
            annotation_clip=False,
        )
    ax.set_title("人均CO2与碳排放强度关系")
    ax.set_xlabel("人均CO2（吨/人）")
    ax.set_ylabel("碳排放强度（吨/万元GDP）")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend_.remove()
    size_header = "CO²总量（Mt）"
    size_start = labels.index(size_header)
    type_handles = handles[1:size_start]
    type_labels = labels[1:size_start]
    size_handles = handles[size_start + 1 :]
    size_labels = labels[size_start + 1 :]
    type_legend = ax.legend(type_handles, type_labels, title="类型", loc="upper left", frameon=True)
    ax.add_artist(type_legend)
    ax.legend(size_handles, size_labels, title=size_header, loc="lower right", frameon=True)
    save_fig(fig, FIG_DIR, "02_人均CO2与碳排放强度散点图")

    if not CHINA_GEOJSON.exists():
        raise FileNotFoundError(f"缺少中国省级边界文件: {CHINA_GEOJSON}")
    geojson = json.loads(CHINA_GEOJSON.read_text(encoding="utf-8"))
    value_map = dict(zip(province["省份"].map(normalize_province_name), province["CO2总量_Mt"]))
    top_label_provinces = set(province.nlargest(6, "CO2总量_Mt")["省份"].map(normalize_province_name))
    values = np.array(list(value_map.values()), dtype=float)
    norm = plt.Normalize(values.min(), values.max())
    cmap = plt.get_cmap("YlOrRd")

    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    missing_patches = []
    data_patches = []
    data_colors = []
    label_points = []
    for feature in geojson["features"]:
        raw_name = feature.get("properties", {}).get("name", "")
        if not raw_name:
            continue
        province_name = normalize_province_name(raw_name)
        polygons = geometry_to_polygons(feature["geometry"])
        if province_name in value_map:
            color = cmap(norm(value_map[province_name]))
            for polygon in polygons:
                data_patches.append(MplPolygon(polygon, closed=True))
                data_colors.append(color)
            if province_name in top_label_provinces:
                center = feature.get("properties", {}).get("centroid") or feature.get("properties", {}).get("center")
                if center:
                    label_points.append((province_name, center[0], center[1]))
        else:
            for polygon in polygons:
                missing_patches.append(MplPolygon(polygon, closed=True))

    if missing_patches:
        ax.add_collection(PatchCollection(missing_patches, facecolor="#E5E7EB", edgecolor="white", linewidth=0.7, zorder=1))
    if data_patches:
        ax.add_collection(PatchCollection(data_patches, facecolor=data_colors, edgecolor="white", linewidth=0.8, zorder=2))
    for name, x_coord, y_coord in label_points:
        ax.text(x_coord, y_coord, name, fontsize=8, ha="center", va="center", color="#111827", zorder=3)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("CO²总量（Mt）")
    ax.legend(handles=[Patch(facecolor="#E5E7EB", edgecolor="white", label="无数据或未纳入30省样本")], loc="lower left", frameon=True)
    ax.set_title("2022年省级CO²排放总量空间分布")
    ax.set_xlim(72, 136)
    ax.set_ylim(17, 55)
    ax.set_aspect("equal")
    ax.axis("off")
    save_fig(fig, FIG_DIR, "05_省级CO2总量空间分布热力图")

    topsis_sorted = topsis.sort_values("TOPSIS低碳得分", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(topsis_sorted["省份"], topsis_sorted["TOPSIS低碳得分"], color=sns.color_palette("crest", len(topsis_sorted)))
    ax.set_title("熵权TOPSIS低碳发展得分排名")
    ax.set_xlabel("TOPSIS低碳得分")
    ax.set_ylabel("")
    save_fig(fig, FIG_DIR, "03_TOPSIS低碳得分排名图")

    old_radar = FIG_DIR / "04_聚类类型均值画像雷达图.png"
    if old_radar.exists():
        old_radar.unlink()

    profile = problem_one["cluster_profile"].copy()
    radar_cols = ["CO2总量_Mt", "人均CO2_吨每人", "碳排放强度_吨每万元GDP", "煤炭相关排放占比", "第二产业占比", "人均GDP_万元每人", "城镇化率_%"]
    scaled = profile.copy()
    for col in radar_cols:
        mn, mx = cluster[col].min(), cluster[col].max()
        scaled[col] = (profile[col] - mn) / (mx - mn) if mx > mn else 0
    labels = ["总量", "人均", "强度", "煤炭", "二产", "人均GDP", "城镇化"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    polygon_angles = np.r_[angles, angles[0]]
    palette = dict(zip(scaled["类型"], sns.color_palette("deep", n_colors=len(scaled))))

    for idx, (_, row) in enumerate(scaled.iterrows(), start=1):
        values = np.array([row[col] for col in radar_cols], dtype=float)
        closed_values = np.r_[values, values[0]]
        color = palette[row["类型"]]

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        ax.grid(False)
        ax.spines["polar"].set_visible(False)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Draw polygon gridlines and radial spokes manually, replacing circular grids.
        for radius in [0.2, 0.4, 0.6, 0.8, 1.0]:
            ax.plot(polygon_angles, [radius] * len(polygon_angles), color="#D1D5DB", linewidth=1)
        for angle in angles:
            ax.plot([angle, angle], [0, 1], color="#D1D5DB", linewidth=1)

        ax.plot(polygon_angles, closed_values, color=color, linewidth=2.6)
        ax.fill(polygon_angles, closed_values, color=color, alpha=0.20)
        ax.scatter(angles, values, color=color, s=42, zorder=3)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_title(row["类型"], fontsize=17, pad=22)
        save_fig(fig, FIG_DIR, f"04_{idx}_{row['类型']}_均值画像雷达图")


def main() -> None:
    setup_style()
    province, _, _ = read_data()
    result = run_problem_one(province)
    make_problem_one_figures(result)
    print(f"问题一结果已生成: {RESULT_DIR}")
    print(f"问题一图片已生成: {FIG_DIR}")


if __name__ == "__main__":
    main()
