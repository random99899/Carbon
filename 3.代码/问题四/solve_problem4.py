from __future__ import annotations

"""
问题四脚本说明

算法与模型：
- 不再重新建模，而是读取问题一的省份分类、问题二的驱动因素检验、问题三的情景预测结果。
- 按“区域分类-关键驱动-达峰预测”逻辑生成政策建议报告。

使用数据：
- 输入：问题一、问题二、问题三脚本返回的结果字典，或由run_all.py统一传入。
- 输出：1.结果/问题四/问题四_政策建议报告.md。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import PROBLEM_RESULT_DIRS, read_data, setup_style
from 问题一.solve_problem1 import run_problem_one
from 问题二.solve_problem2 import run_problem_two
from 问题三.solve_problem3 import run_problem_three


PROBLEM = "问题四"
RESULT_DIR = PROBLEM_RESULT_DIRS[PROBLEM]


def write_policy_report(problem_one: dict[str, pd.DataFrame], problem_two: dict[str, pd.DataFrame], problem_three: dict[str, pd.DataFrame]) -> str:
    cluster = problem_one["cluster"]
    ridge = problem_two["ridge_coef"].sort_values("绝对值排序")
    peak = problem_three["peak"].set_index("情景")
    resource = cluster[cluster["类型"].eq("资源依赖高排放型")]["省份"].tolist()
    industrial = cluster[cluster["类型"].eq("工业制造高排放型")]["省份"].tolist()
    main_drivers = "、".join(ridge.head(2)["变量"].tolist())

    policy = f"""# 问题四：政策建议报告

基于省际分类、STIRPAT驱动因素识别和三情景预测结果，我国推进“双碳”目标应坚持“总量控制、结构优化、效率提升、区域分类”的思路。首先，对{"、".join(resource)}等资源依赖高排放型省份，应将煤炭消费总量控制和煤电低碳改造作为重点，推动煤炭清洁高效利用、煤电灵活性改造、新能源基地建设和资源型产业转型，避免经济增长继续依赖高碳能源扩张。其次，对{"、".join(industrial)}等工业制造高排放型省份，应聚焦钢铁、建材、电力、石化和交通等重点行业，实施节能改造、余热利用、电气化替代和绿色供应链管理，建立单位产品碳强度约束。再次，中等转型压力型省份应在承接产业转移时设置能耗和碳排放准入门槛，防止高耗能产业简单转移，同时提高能源利用效率和非化石能源占比。北京、上海等经济发达效率型地区应发挥技术、金融和治理优势，推广绿色金融、碳交易、数字化碳管理和低碳消费。模型结果表明，{main_drivers}是最稳定的关键驱动因素；情景预测显示，基准情景约在{int(peak.loc["基准情景", "达峰年份"])}年达峰，强化低碳情景下2045年较峰值下降{peak.loc["强化低碳情景", "2045较峰值下降比例"]:.1%}。因此，应持续降低煤炭占比，扩大风光水核等非化石能源供给，强化技术创新和碳市场机制，形成差异化、可考核、可持续的减排路径。
"""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "问题四_政策建议报告.md").write_text(policy, encoding="utf-8")
    return policy


def main() -> None:
    setup_style()
    province, _, carbon = read_data()
    problem_one = run_problem_one(province)
    problem_two = run_problem_two(province)
    problem_three = run_problem_three(carbon)
    write_policy_report(problem_one, problem_two, problem_three)
    print(f"问题四报告已生成: {RESULT_DIR}")


if __name__ == "__main__":
    main()
