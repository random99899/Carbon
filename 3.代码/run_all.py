from __future__ import annotations

"""
总入口脚本

作用：
- 依次调用问题一、问题二、问题三、问题四的独立脚本。
- 生成分类目录下的结果表、图片、政策建议和汇总报告。
- 执行必要的输出完整性检查。

运行方式：
    .\\.venv\\Scripts\\python.exe .\\3.代码\\run_all.py
"""

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
ROOT = CODE_ROOT.parent
sys.pycache_prefix = str(ROOT / ".pycache")
sys.path.append(str(CODE_ROOT))

import numpy as np
import pandas as pd

from common import (
    FIG_DIR,
    PROBLEM_FIG_DIRS,
    PROBLEM_RESULT_DIRS,
    RESULT_DIR,
    ROOT,
    dataframe_to_markdown,
    ensure_dirs,
    percent,
    read_data,
    setup_style,
    write_gitignore,
)
from 问题一.solve_problem1 import make_problem_one_figures, run_problem_one
from 问题二.solve_problem2 import make_problem_two_figures, run_problem_two
from 问题三.solve_problem3 import make_problem_three_figures, run_problem_three
from 问题四.solve_problem4 import write_policy_report


def write_summary_report(problem_one: dict[str, pd.DataFrame], problem_two: dict[str, pd.DataFrame], problem_three: dict[str, pd.DataFrame]) -> None:
    spatial = problem_one["spatial_stats"].set_index("指标")
    topsis = problem_one["topsis"].copy()
    weights = problem_one["weights"].sort_values("权重", ascending=False)
    silhouette = problem_one["silhouette"].copy()
    cluster = problem_one["cluster"].copy()
    cluster_profile = problem_one["cluster_profile"].copy()
    ols = problem_two["ols_coef"].copy()
    fit = problem_two["fit"].set_index("指标")["数值"]
    vif = problem_two["vif"].copy()
    ridge = problem_two["ridge_metrics"].set_index("指标")["数值"]
    ridge_coef = problem_two["ridge_coef"].sort_values("绝对值排序")
    annual = problem_three["annual"].copy()
    sector_2024 = problem_three["sector_2024"].copy()
    peak = problem_three["peak"].copy()

    top_total = topsis.sort_values("CO2总量_Mt", ascending=False).head(8)[["省份", "CO2总量_Mt", "人均CO2_吨每人", "碳排放强度_吨每万元GDP"]]
    top_pc = topsis.sort_values("人均CO2_吨每人", ascending=False).head(5)[["省份", "人均CO2_吨每人", "碳排放强度_吨每万元GDP"]]
    topsis_top = topsis.sort_values("TOPSIS低碳得分", ascending=False).head(8)[["省份", "TOPSIS低碳得分", "低碳等级"]]
    topsis_bottom = topsis.sort_values("TOPSIS低碳得分").head(8)[["省份", "TOPSIS低碳得分", "低碳等级"]]
    cluster_table = (
        cluster.groupby("类型")["省份"]
        .apply(lambda s: "、".join(s.tolist()))
        .reset_index()
        .merge(cluster_profile, on="类型", how="left")
    )
    ols_focus = ols[ols["变量"].isin(["lnP", "lnA", "煤炭相关排放占比", "第二产业占比", "城镇化率"])].copy()
    annual_visible = annual[["Year", "CO2 (Mt)", "记录天数", "是否完整年份"]]
    peak_visible = peak.copy()
    peak_visible["2045较峰值下降比例"] = peak_visible["2045较峰值下降比例"].map(percent)

    total_stats = spatial.loc["CO2总量_Mt"]
    pc_stats = spatial.loc["人均CO2_吨每人"]
    intensity_stats = spatial.loc["碳排放强度_吨每万元GDP"]
    coal_stats = spatial.loc["煤炭相关排放占比"]
    best_k = int(silhouette.sort_values("轮廓系数", ascending=False).iloc[0]["K"])
    best_k_score = float(silhouette.sort_values("轮廓系数", ascending=False).iloc[0]["轮廓系数"])
    base_peak = peak.set_index("情景").loc["基准情景"]
    low_peak = peak.set_index("情景").loc["低碳情景"]
    strong_peak = peak.set_index("情景").loc["强化低碳情景"]

    summary = f"""# B题结果汇总

## 1. 数据口径与总说明

本项目使用`数据/2022省级综合表.xlsx`完成30省横截面分析，使用`数据/附件1：2019—2025 年全国碳排放数据.csv`中`Sector=Total`完成全国趋势分析。附件1中2025年只有截至2025-09-30的数据，不能直接作为全年值与2019至2024年比较；因此三情景预测以2024年全国排放量{annual.loc[annual['Year'].eq(2024), 'CO2 (Mt)'].iloc[0]:.2f} Mt作为基准。

## 2. 问题一：碳排放空间差异分析与省份分类分级

> 题目要求：“选取**碳排放总量、人均碳排放、碳排放强度等**核心指标，分析各省碳排放空间分布特征与差异显著性。”

解答：省际碳排放差异显著。2022年30省CO2总量均值为{total_stats['均值']:.2f} Mt，变异系数为{total_stats['变异系数']:.4f}，基尼系数为{total_stats['基尼系数']:.4f}，泰尔指数为{total_stats['泰尔指数']:.4f}，说明排放总量具有明显空间集聚性。人均CO2变异系数为{pc_stats['变异系数']:.4f}，碳排放强度变异系数为{intensity_stats['变异系数']:.4f}、基尼系数为{intensity_stats['基尼系数']:.4f}，说明不同省份不仅总量差异大，单位人口和单位经济产出的排放压力差异也很突出。煤炭相关排放占比均值为{coal_stats['均值']:.4f}，说明煤炭依赖是普遍问题，但其基尼系数仅为{coal_stats['基尼系数']:.4f}，地区间差异小于排放强度差异。

高排放总量省份主要为：

{dataframe_to_markdown(top_total, ".3f")}

人均排放压力最高的省份主要为：

{dataframe_to_markdown(top_pc, ".3f")}

> 题目要求：“综合排放规模、排放效率、经济关联度构建多维度评价体系，建立分类分级模型，对全国各省碳排放水平进行科学分类与等级划分，并说明分类依据与政策适配性。”

解答：本文以CO2总量、人均CO2、碳排放强度、煤炭相关排放占比、第二产业占比、人均GDP、城镇化率7个指标构建评价体系，采用熵权TOPSIS计算低碳发展综合得分。权重最高的三个指标是{weights.iloc[0]['指标']}（{weights.iloc[0]['权重']:.4f}）、{weights.iloc[1]['指标']}（{weights.iloc[1]['权重']:.4f}）、{weights.iloc[2]['指标']}（{weights.iloc[2]['权重']:.4f}），说明经济发展质量、煤炭依赖程度和城镇化治理能力对综合评价影响较大。

TOPSIS排名靠前省份为：

{dataframe_to_markdown(topsis_top, ".3f")}

TOPSIS排名靠后省份为：

{dataframe_to_markdown(topsis_bottom, ".3f")}

聚类模型方面，K=2至6的轮廓系数显示K={best_k}时统计分离度最高（轮廓系数{best_k_score:.4f}）。但为增强政策解释性，主模型采用K=4，将省份划分为资源依赖高排放型、工业制造高排放型、中等转型压力型和经济发达效率型。分类结果如下：

{dataframe_to_markdown(cluster_table, ".3f")}

政策适配性：资源依赖高排放型应重点控煤和推动资源型产业转型；工业制造高排放型应聚焦钢铁、建材、电力等行业节能改造；中等转型压力型应避免高耗能产业简单转移；经济发达效率型应发挥技术、金融和治理优势。

## 3. 问题二：碳排放影响因素识别与预测模型构建

> 题目要求：“选取**能源消费结构、产业结构、人均GDP、城镇化率、技术水平等**关键指标，分析影响碳排放的核心驱动因素。”

解答：基于现有横截面数据，构建STIRPAT模型：因变量为ln(CO2总量)，解释变量为ln(人口)、ln(人均GDP)、煤炭相关排放占比、第二产业占比、城镇化率。OLS估计结果显示，ln(人口)系数为{ols_focus.loc[ols_focus['变量'].eq('lnP'), '系数'].iloc[0]:.4f}，t值为{ols_focus.loc[ols_focus['变量'].eq('lnP'), 't值'].iloc[0]:.2f}；煤炭相关排放占比系数为{ols_focus.loc[ols_focus['变量'].eq('煤炭相关排放占比'), '系数'].iloc[0]:.4f}，t值为{ols_focus.loc[ols_focus['变量'].eq('煤炭相关排放占比'), 't值'].iloc[0]:.2f}。二者是当前样本中最显著、最稳定的正向驱动因素。人均GDP、第二产业占比和城镇化率系数为正，但显著性较弱，说明其影响可能通过产业结构、能源结构和区域发展阶段间接体现。

OLS主要系数如下：

{dataframe_to_markdown(ols_focus, ".4f")}

> 题目要求：“构建碳排放定量预测模型，完成模型参数估计、显著性检验与优化，确保模型具有良好拟合效果与预测精度，并说明模型的适用范围与局限性。”

解答：OLS模型R2为{fit['R2']:.4f}，调整R2为{fit['调整R2']:.4f}，RMSE_log为{fit['RMSE_log']:.4f}，F检验p值为{fit['F检验p值']:.4g}，说明模型对2022年省际排放差异具有较强解释力。VIF检验结果如下，所有变量VIF均低于10，不存在严重多重共线性：

{dataframe_to_markdown(vif, ".3f")}

为提升稳健性，进一步采用岭回归和留一交叉验证。岭回归最优alpha为{ridge['最优alpha']:.4f}，LOOCV_R2为{ridge['LOOCV_R2']:.4f}，LOOCV_RMSE_log为{ridge['LOOCV_RMSE_log']:.4f}。标准化系数排序显示，{ridge_coef.iloc[0]['变量']}和{ridge_coef.iloc[1]['变量']}仍是最重要变量，说明“人口规模+煤炭依赖”的结论稳定。模型适用于解释2022年省际横截面差异；局限是当前缺少多年省级面板数据，难以直接识别长期动态因果关系。

## 4. 问题三：多情景碳排放趋势预测与碳达峰研判

> 题目要求：“基于最优预测模型，设定**基准、低碳、强化低碳**三种发展情景，预测2026—2045年全国碳排放总量与强度变化趋势。”

解答：由于全国年度辅助变量样本较短，本文将STIRPAT驱动因素识别结果与Kaya恒等式结合，采用人口、人均GDP、能源强度、能源碳强度四类增长率进行情景递推。附件1年度排放核验如下，其中2025年为不完整年份：

{dataframe_to_markdown(annual_visible, ".2f")}

2024年分部门排放显示，工业、电力和地面交通是主要排放来源：

{dataframe_to_markdown(sector_2024.head(6), ".2f")}

> 题目要求：“研判碳达峰时间节点、峰值水平及减排潜力，为‘双碳’目标实施提供量化支撑。”

解答：三情景预测结果如下：

{dataframe_to_markdown(peak_visible, ".3f")}

基准情景下，全国排放在{int(base_peak['达峰年份'])}年达到峰值{base_peak['峰值_Mt']:.0f} Mt，之后缓慢下降，2045年为{base_peak['2045年_Mt']:.0f} Mt，较峰值下降{base_peak['2045较峰值下降比例']:.1%}。低碳情景下，排放自2024年后进入下降通道，2030年降至{low_peak['2030年_Mt']:.0f} Mt，2045年较峰值下降{low_peak['2045较峰值下降比例']:.1%}。强化低碳情景下降幅最大，2045年降至{strong_peak['2045年_Mt']:.0f} Mt，较峰值下降{strong_peak['2045较峰值下降比例']:.1%}。因此，能源强度下降和能源碳强度下降速度是决定能否低峰值达峰和中长期深度减排的关键。

## 5. 问题四：政策建议

> 题目要求：“结合问题一至三的研究结论，针对‘双碳’目标实现路径，从优化能源结构、推动产业升级、实施区域差异化减排、强化技术创新、完善政策机制等方面，提出科学可行、针对性强、可操作性高的政策建议。”

解答：政策建议已单独写入`1.结果/问题四/问题四_政策建议报告.md`。核心建议是按省份类型实施差异化减排：资源依赖高排放型控煤并推动资源型产业转型，工业制造高排放型聚焦重点行业节能降碳，中等转型压力型强化产业准入和效率提升，经济发达效率型发挥绿色金融、技术创新和碳管理优势。全国层面应持续降低煤炭占比、扩大非化石能源供给、强化碳市场和技术创新，因为情景预测表明能源强度和能源碳强度下降越快，峰值越低、达峰后下降越明显。

## 6. 文件索引

- 问题一结果：`1.结果/问题一/01_空间差异指标.csv`等6个CSV文件
- 问题二结果：`1.结果/问题二/01_OLS系数.csv`等6个CSV文件
- 问题三结果：`1.结果/问题三/01_全国年度排放.csv`等5个CSV文件
- 问题四报告：`1.结果/问题四/问题四_政策建议报告.md`
- 图片目录：`2.图片/问题一`、`2.图片/问题二`、`2.图片/问题三`
- 代码目录：`3.代码/问题一`、`3.代码/问题二`、`3.代码/问题三`、`3.代码/问题四`
"""
    out_dir = PROBLEM_RESULT_DIRS["汇总"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "B题结果汇总.md").write_text(summary, encoding="utf-8")


def write_readme() -> None:
    readme = """# B题：我国碳排放时空特征分析与趋势预测

本项目为第十届校内数学建模竞赛B题的数据处理、模型求解、图表生成和论文素材整理工程。

## 项目结构

```text
数据处理/
├─ 数据/                         # 题目附件与整理后的数据
├─ 题目/                         # B题题目文件
├─ 1.结果/
│  ├─ 问题一/                    # 空间差异、TOPSIS、聚类结果
│  ├─ 问题二/                    # STIRPAT回归、检验、岭回归结果
│  ├─ 问题三/                    # 三情景预测和达峰结果
│  ├─ 问题四/                    # 政策建议报告
│  └─ 汇总/                      # B题结果汇总
├─ 2.图片/
│  ├─ 问题一/                    # 空间分布、TOPSIS、聚类图
│  ├─ 问题二/                    # STIRPAT系数和OLS拟合图
│  └─ 问题三/                    # 情景预测和达峰图
├─ 3.代码/
│  ├─ 问题一/solve_problem1.py
│  ├─ 问题二/solve_problem2.py
│  ├─ 问题三/solve_problem3.py
│  ├─ 问题四/solve_problem4.py
│  ├─ common.py
│  ├─ run_all.py
│  └─ requirements.txt
└─ README.md
```

## 环境安装

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r .\\3.代码\\requirements.txt
```

## 一键运行

```powershell
.\\.venv\\Scripts\\python.exe .\\3.代码\\run_all.py
```

也可以单独运行某个问题：

```powershell
.\\.venv\\Scripts\\python.exe .\\3.代码\\问题一\\solve_problem1.py
.\\.venv\\Scripts\\python.exe .\\3.代码\\问题二\\solve_problem2.py
.\\.venv\\Scripts\\python.exe .\\3.代码\\问题三\\solve_problem3.py
.\\.venv\\Scripts\\python.exe .\\3.代码\\问题四\\solve_problem4.py
```

运行后会自动生成或覆盖同名结果文件，不会清空目录或删除原始数据。

## 数据说明

- `数据/附件1：2019—2025 年全国碳排放数据.csv`：全国碳排放时间序列，字段为`Area, CO2 (Mt), Sector, Date`。全国总量分析只使用`Sector == "Total"`。
- `数据/附件2：2022 年全国 30 个省份碳排放清单.xlsx`：30省分能源、分部门碳排放清单。
- `数据/2022省级综合表.xlsx`：已合并人口、GDP、第二产业、城镇化率、排放强度和能源结构指标，是问题一和问题二主表。
- `数据/全国数据.xlsx`：2019至2024年全国GDP、人口、能源消费总量和煤炭占比，用于辅助情景设定。

注意：附件1中2025年数据截至2025-09-30，不能直接作为全年值与2019至2024年比较。三情景预测以2024年为基准年。

## 模型路线

1. 问题一：变异系数、基尼系数、泰尔指数、熵权TOPSIS、K-means聚类。
2. 问题二：STIRPAT横截面OLS、VIF共线性检验、岭回归留一交叉验证。
3. 问题三：Kaya-STIRPAT三情景递推预测、达峰年份和减排潜力判断。
4. 问题四：根据分类、驱动因素和情景预测生成差异化政策建议。

## 输出说明

- `1.结果/汇总/B题结果汇总.md`：按题目原问题逐条引用并给出详细解答。
- `1.结果/问题一/`：空间差异、TOPSIS和聚类CSV结果。
- `1.结果/问题二/`：OLS、VIF、岭回归和拟合残差CSV结果。
- `1.结果/问题三/`：年度排放、情景参数、预测序列和达峰结果。
- `1.结果/问题四/`：约500字政策建议报告。
- `2.图片/`：按问题分文件夹保存PNG图片。

## 可复现性说明

所有结果由`3.代码/run_all.py`从`数据/`目录重新计算得到。若需要调整情景参数，可修改`3.代码/common.py`中的`scenario_params()`后重新运行。
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")


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
        if not set(range(2026, 2046)).issubset(set(group["年份"])):
            raise AssertionError(f"{scenario}未覆盖2026-2045完整预测区间。")
    expected_csvs = {
        "问题一": ["01_空间差异指标", "02_熵权指标权重", "03_TOPSIS得分排名", "04_聚类K值检验", "05_K4聚类结果", "06_类型均值画像"],
        "问题二": ["01_OLS系数", "02_拟合优度", "03_VIF共线性检验", "04_拟合值残差", "05_岭回归交叉验证", "06_岭回归标准化系数"],
        "问题三": ["01_全国年度排放", "02_2024部门排放", "03_情景参数", "04_2024-2045预测序列", "05_达峰与减排潜力"],
    }
    for problem, stems in expected_csvs.items():
        for stem in stems:
            path = PROBLEM_RESULT_DIRS[problem] / f"{stem}.csv"
            if not path.exists() or path.stat().st_size == 0:
                raise AssertionError(f"CSV结果未正确生成: {path}")
    expected_figs = {
        "问题一": ["01_省份CO2总量排名图", "02_人均CO2与碳排放强度散点图", "03_TOPSIS低碳得分排名图", "04_聚类类型均值画像雷达图"],
        "问题二": ["05_STIRPAT标准化系数图", "06_OLS拟合值与真实值对比图"],
        "问题三": ["07_三情景碳排放趋势图", "08_三情景峰值与减排潜力对比图"],
    }
    for problem, stems in expected_figs.items():
        for stem in stems:
            path = PROBLEM_FIG_DIRS[problem] / f"{stem}.png"
            if not path.exists() or path.stat().st_size == 0:
                raise AssertionError(f"图片未正确生成: {path}")


def main() -> None:
    ensure_dirs()
    setup_style()
    province, _, carbon = read_data()
    problem_one = run_problem_one(province)
    problem_two = run_problem_two(province)
    problem_three = run_problem_three(carbon)
    make_problem_one_figures(problem_one)
    make_problem_two_figures(problem_two)
    make_problem_three_figures(problem_three)
    write_policy_report(problem_one, problem_two, problem_three)
    write_summary_report(problem_one, problem_two, problem_three)
    write_readme()
    write_gitignore()
    validate_outputs(problem_one, problem_two, problem_three)
    print("全部结果已按问题分类生成。")
    print(f"结果目录: {RESULT_DIR}")
    print(f"图片目录: {FIG_DIR}")


if __name__ == "__main__":
    main()
