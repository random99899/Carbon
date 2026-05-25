from __future__ import annotations

"""
总入口脚本

作用：
- 依次调用问题一、问题二、问题三的独立脚本。
- 生成分类目录下的结果表和图片。
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
    ensure_dirs,
    percent,
    read_data,
    scenario_driver_assumptions,
    setup_style,
)
from 问题一.solve_problem1 import make_problem_one_figures, run_problem_one
from 问题二.solve_problem2 import make_problem_two_figures, run_problem_two
from 问题三.solve_problem3 import make_problem_three_figures, run_problem_three




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
        "问题三": [
            "01_全国年度排放",
            "02_2024部门排放",
            "03_情景参数",
            "04_2024-2045预测序列",
            "05_达峰与减排潜力",
            "06_STIRPAT递推系数",
            "07_驱动变量基准值",
            "08_单因素灵敏度分析",
            "09_STIRPAT系数灵敏度分析",
        ],
    }
    for problem, stems in expected_csvs.items():
        for stem in stems:
            path = PROBLEM_RESULT_DIRS[problem] / f"{stem}.csv"
            if not path.exists() or path.stat().st_size == 0:
                raise AssertionError(f"CSV结果未正确生成: {path}")
    expected_figs = {
        "问题一": [
            "01_省份CO2总量排名图",
            "02_人均CO2与碳排放强度散点图",
            "03_TOPSIS低碳得分排名图",
            "04_1_资源依赖高排放型_均值画像雷达图",
            "04_2_工业制造高排放型_均值画像雷达图",
            "04_3_中等转型压力型_均值画像雷达图",
            "04_4_经济发达效率型_均值画像雷达图",
            "05_省级CO2总量空间分布热力图",
        ],
        "问题二": ["05_STIRPAT标准化系数图", "06_OLS拟合值与真实值对比图"],
        "问题三": [
            "07_三情景碳排放趋势图",
            "08_三情景峰值与减排潜力对比图",
            "09_单因素灵敏度_2045排放影响图",
            "10_STIRPAT系数灵敏度_2045排放影响图",
        ],
    }
    for problem, stems in expected_figs.items():
        for stem in stems:
            path = PROBLEM_FIG_DIRS[problem] / f"{stem}.png"
            if not path.exists() or path.stat().st_size == 0:
                raise AssertionError(f"图片未正确生成: {path}")


def main() -> None:
    ensure_dirs()
    setup_style()
    province, national, carbon = read_data()
    problem_one = run_problem_one(province)
    problem_two = run_problem_two(province)
    problem_three = run_problem_three(carbon, national, province, problem_two)
    make_problem_one_figures(problem_one)
    make_problem_two_figures(problem_two)
    make_problem_three_figures(problem_three)
    validate_outputs(problem_one, problem_two, problem_three)
    print("全部结果已按问题分类生成。")
    print(f"结果目录: {RESULT_DIR}")
    print(f"图片目录: {FIG_DIR}")


if __name__ == "__main__":
    main()
