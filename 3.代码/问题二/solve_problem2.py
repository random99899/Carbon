from __future__ import annotations

"""
问题二脚本说明

算法与模型：
- STIRPAT横截面回归：ln(CO2)=a+b1 ln(P)+b2 ln(A)+b3 Coal+b4 IS+b5 U+e。
- OLS显著性检验：输出系数、标准误、t值、p值、R2、调整R2。
- VIF共线性检验：判断解释变量是否存在严重多重共线性。
- 岭回归留一交叉验证：用正则化模型检验驱动因素排序和预测稳健性。

使用数据：
- 输入：数据/2022省级综合表.xlsx，工作表“主表”。
- 输出：1.结果/问题二/ 下多个CSV结果文件。
- 输出图片：2.图片/问题二/05_STIRPAT标准化系数图、06_OLS拟合值与真实值对比图。
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
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from common import PROBLEM_FIG_DIRS, PROBLEM_RESULT_DIRS, read_data, save_fig, setup_style, write_csv


PROBLEM = "问题二"
RESULT_DIR = PROBLEM_RESULT_DIRS[PROBLEM]
FIG_DIR = PROBLEM_FIG_DIRS[PROBLEM]


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
    fitted_df["绝对误差_Mt"] = (fitted_df["拟合CO2_Mt"] - fitted_df["实际CO2_Mt"]).abs()
    fitted_df["误差方向"] = np.where(fitted_df["拟合CO2_Mt"] > fitted_df["实际CO2_Mt"], "模型高估", "模型低估")

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

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(ols_df, RESULT_DIR / "01_OLS系数.csv")
    write_csv(fit_df, RESULT_DIR / "02_拟合优度.csv")
    write_csv(vif_df, RESULT_DIR / "03_VIF共线性检验.csv")
    write_csv(fitted_df, RESULT_DIR / "04_拟合值残差.csv")
    write_csv(ridge_metrics_df, RESULT_DIR / "05_岭回归交叉验证.csv")
    write_csv(ridge_coef_df, RESULT_DIR / "06_岭回归标准化系数.csv")

    return {
        "ols_coef": ols_df,
        "fit": fit_df,
        "vif": vif_df,
        "fitted": fitted_df,
        "ridge_metrics": ridge_metrics_df,
        "ridge_coef": ridge_coef_df,
    }


def make_problem_two_figures(problem_two: dict[str, pd.DataFrame]) -> None:
    ridge_coef = problem_two["ridge_coef"].sort_values("标准化岭回归系数")
    variable_labels = {
        "lnP": "人口规模（lnP）",
        "lnA": "经济发展水平（lnA）",
        "煤炭相关排放占比": "煤炭相关排放占比",
        "第二产业占比": "第二产业占比",
        "城镇化率": "城镇化率",
    }
    variable_colors = {
        "lnP": "#2563EB",
        "lnA": "#059669",
        "煤炭相关排放占比": "#DC2626",
        "第二产业占比": "#D97706",
        "城镇化率": "#7C3AED",
    }
    ridge_coef["变量说明"] = ridge_coef["变量"].map(variable_labels).fillna(ridge_coef["变量"])
    bar_colors = ridge_coef["变量"].map(variable_colors).fillna("#4B5563")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ridge_coef["变量说明"], ridge_coef["标准化岭回归系数"], color=bar_colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title("STIRPAT岭回归标准化系数")
    ax.set_xlabel("标准化系数")
    ax.set_ylabel("")
    save_fig(fig, FIG_DIR, "05_STIRPAT标准化系数图")

    fitted = problem_two["fitted"].copy()
    fitted["误差_Mt"] = fitted["拟合CO2_Mt"] - fitted["实际CO2_Mt"]
    top_labels = fitted.nlargest(8, "绝对误差_Mt")
    lim_min = min(fitted["实际CO2_Mt"].min(), fitted["拟合CO2_Mt"].min()) * 0.88
    lim_max = max(fitted["实际CO2_Mt"].max(), fitted["拟合CO2_Mt"].max()) * 1.08

    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.scatterplot(
        data=fitted,
        x="实际CO2_Mt",
        y="拟合CO2_Mt",
        hue="误差方向",
        size="绝对误差_Mt",
        sizes=(50, 280),
        palette={"模型高估": "#C2410C", "模型低估": "#2563EB"},
        alpha=0.82,
        ax=ax,
    )
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "--", color="#333333", linewidth=1.5, label="完全拟合线：拟合值=实际值")
    for _, row in top_labels.iterrows():
        ax.annotate(
            row["省份"],
            xy=(row["实际CO2_Mt"], row["拟合CO2_Mt"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )
        ax.vlines(row["实际CO2_Mt"], row["实际CO2_Mt"], row["拟合CO2_Mt"], color="#777777", alpha=0.35, linewidth=1)
    ax.text(
        0.03,
        0.97,
        "点：省份；横轴：实际CO2；纵轴：STIRPAT拟合CO2\n黑色虚线：完全拟合线；点在虚线上方表示模型高估\n灰色竖线：该省拟合误差大小；点越大误差越大",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88},
    )
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_title("STIRPAT模型实际值与拟合值对比及误差说明")
    ax.set_xlabel("实际CO2排放量（Mt）")
    ax.set_ylabel("模型拟合CO2排放量（Mt）")
    ax.legend(loc="lower right", frameon=True)
    save_fig(fig, FIG_DIR, "06_OLS拟合值与真实值对比图")


def main() -> None:
    setup_style()
    province, _, _ = read_data()
    result = run_problem_two(province)
    make_problem_two_figures(result)
    print(f"问题二结果已生成: {RESULT_DIR}")
    print(f"问题二图片已生成: {FIG_DIR}")


if __name__ == "__main__":
    main()
