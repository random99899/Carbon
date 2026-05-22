# B题：我国碳排放时空特征分析与趋势预测

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
│  ├─ assets/china_provinces.geojson
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
.\.venv\Scripts\python.exe -m pip install -r .\3.代码\requirements.txt
```

## 一键运行

```powershell
.\.venv\Scripts\python.exe .\3.代码\run_all.py
```

也可以单独运行某个问题：

```powershell
.\.venv\Scripts\python.exe .\3.代码\问题一\solve_problem1.py
.\.venv\Scripts\python.exe .\3.代码\问题二\solve_problem2.py
.\.venv\Scripts\python.exe .\3.代码\问题三\solve_problem3.py
.\.venv\Scripts\python.exe .\3.代码\问题四\solve_problem4.py
```

运行后会自动生成或覆盖同名结果文件，不会清空目录或删除原始数据。

## 数据说明

- `数据/附件1：2019—2025 年全国碳排放数据.csv`：全国碳排放时间序列，字段为`Area, CO2 (Mt), Sector, Date`。全国总量分析只使用`Sector == "Total"`。
- `数据/附件2：2022 年全国 30 个省份碳排放清单.xlsx`：30省分能源、分部门碳排放清单。
- `数据/2022省级综合表.xlsx`：已合并人口、GDP、第二产业、城镇化率、排放强度和能源结构指标，是问题一和问题二主表。
- `数据/全国数据.xlsx`：2019至2024年全国GDP、人口、能源消费总量和煤炭占比，用于辅助情景设定。
- `3.代码/assets/china_provinces.geojson`：省级行政边界文件，用于绘制问题一省级CO2总量空间分布热力图。

注意：附件1中2025年数据截至2025-09-30，不能直接作为全年值与2019至2024年比较。三情景预测以2024年为基准年。

## 模型路线

1. 问题一：变异系数、基尼系数、泰尔指数、熵权TOPSIS、K-means聚类，并结合省级边界绘制CO2总量空间分布热力图。
2. 问题二：STIRPAT横截面OLS、VIF共线性检验、岭回归留一交叉验证。
3. 问题三：STIRPAT系数驱动的三情景递推预测、达峰年份和减排潜力判断。
4. 问题四：根据分类、驱动因素和情景预测生成差异化政策建议。

## 输出说明

- `1.结果/汇总/B题结果汇总.md`：按题目原问题逐条引用并给出详细解答。
- `1.结果/问题一/`：空间差异、TOPSIS和聚类CSV结果。
- `1.结果/问题二/`：OLS、VIF、岭回归和拟合残差CSV结果。
- `1.结果/问题三/`：年度排放、情景参数、STIRPAT递推系数、预测序列、达峰结果和灵敏度分析。
- `1.结果/问题四/`：约500字政策建议报告。
- `2.图片/`：按问题分文件夹保存PNG图片。

## 可复现性说明

所有结果由`3.代码/run_all.py`从`数据/`目录重新计算得到。若需要调整情景参数，可修改`3.代码/common.py`中的`scenario_driver_assumptions()`后重新运行。
