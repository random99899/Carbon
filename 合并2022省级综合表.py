from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "数据"

INPUT_PROVINCE_DATA = DATA_DIR / "2022数据.xlsx"
INPUT_EMISSIONS = DATA_DIR / "附件2：2022 年全国 30 个省份碳排放清单.xlsx"
OUTPUT_FILE = DATA_DIR / "2022省级综合表.xlsx"


SHEET_TO_PROVINCE = {
    "Beijing2022": "北京",
    "Tianjin2022": "天津",
    "Hebei2022": "河北",
    "Shanxi2022": "山西",
    "InnerMongolia2022": "内蒙古",
    "Liaoning2022": "辽宁",
    "Jilin2022": "吉林",
    "Heilongjiang2022": "黑龙江",
    "Shanghai2022": "上海",
    "Jiangsu2022": "江苏",
    "Zhejiang2022": "浙江",
    "Anhui2022": "安徽",
    "Fujian2022": "福建",
    "Jiangxi2022": "江西",
    "Shandong2022": "山东",
    "Henan2022": "河南",
    "Hubei2022": "湖北",
    "Hunan2022": "湖南",
    "Guangdong2022": "广东",
    "Guangxi2022": "广西",
    "Hainan2022": "海南",
    "Chongqing2022": "重庆",
    "Sichuan2022": "四川",
    "Guizhou2022": "贵州",
    "Yunnan2022": "云南",
    "Shaanxi2022": "陕西",
    "Gansu2022": "甘肃",
    "Qinghai2022": "青海",
    "Ningxia2022": "宁夏",
    "Xinjiang2022": "新疆",
}


EMISSION_FIELDS = [
    "Scope_1_Total",
    "Raw_Coal",
    "CleanedCoal",
    "Other_Washed_Coal",
    "Briquettes",
    "Coke",
    "Coke_Oven_Gas",
    "Other_Gas",
    "Other_Coking_Products",
    "Crude_Oil",
    "Gasoline",
    "Kerosene",
    "Diesel_Oil",
    "Fuel_Oil",
    "LPG",
    "Refinery_Gas",
    "Other_Petroleum_Products",
    "Natural_Gas",
    "Process",
]

COAL_FIELDS = [
    "Raw_Coal",
    "CleanedCoal",
    "Other_Washed_Coal",
    "Briquettes",
    "Coke",
    "Coke_Oven_Gas",
    "Other_Gas",
    "Other_Coking_Products",
]

OIL_FIELDS = [
    "Crude_Oil",
    "Gasoline",
    "Kerosene",
    "Diesel_Oil",
    "Fuel_Oil",
    "LPG",
    "Refinery_Gas",
    "Other_Petroleum_Products",
]

FINAL_COLUMNS = [
    "省份",
    "CO2总量_Mt",
    "人口_万人",
    "GDP_亿元",
    "第二产业_亿元",
    "城镇化率_%",
    "人均CO2_吨每人",
    "碳排放强度_吨每万元GDP",
    "人均GDP_万元每人",
    "第二产业占比",
    "技术效率_万元GDP每吨CO2",
    "煤炭相关排放_Mt",
    "煤炭相关排放占比",
    "石油相关排放_Mt",
    "石油相关排放占比",
    "天然气排放_Mt",
    "天然气排放占比",
    "过程排放_Mt",
    "过程排放占比",
]

DESCRIBE_COLUMNS = [
    "人均CO2_吨每人",
    "碳排放强度_吨每万元GDP",
    "第二产业占比",
    "煤炭相关排放占比",
    "石油相关排放占比",
    "天然气排放占比",
    "过程排放占比",
]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def read_province_data() -> pd.DataFrame:
    df = pd.read_excel(INPUT_PROVINCE_DATA)
    df = df.dropna(how="all").copy()
    df.columns = [str(col).strip() for col in df.columns]
    df = df.rename(
        columns={
            "地  区": "省份",
            "地区": "省份",
            "人口（万人）": "人口_万人",
            "第二产业（亿元）": "第二产业_亿元",
            "城镇化": "城镇化率_%",
            "GDP（亿元）": "GDP_亿元",
        }
    )
    df["省份"] = df["省份"].astype(str).str.replace(r"\s+", "", regex=True)
    numeric_cols = ["人口_万人", "第二产业_亿元", "城镇化率_%", "GDP_亿元"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["省份", *numeric_cols]]


def read_emissions_data() -> pd.DataFrame:
    excel = pd.ExcelFile(INPUT_EMISSIONS)
    rows = []

    for sheet_name in excel.sheet_names:
        if sheet_name == "NOTE":
            continue
        if sheet_name not in SHEET_TO_PROVINCE:
            raise ValueError(f"未配置的附件2 sheet 名称: {sheet_name}")

        sheet_df = pd.read_excel(INPUT_EMISSIONS, sheet_name=sheet_name)
        total = sheet_df.loc[sheet_df["Emission_Inventory"].eq("TotalEmissions")]
        if total.empty:
            raise ValueError(f"{sheet_name} 中未找到 TotalEmissions 行")

        row = total.iloc[0][EMISSION_FIELDS].copy()
        row["省份"] = SHEET_TO_PROVINCE[sheet_name]
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.rename(columns={"Scope_1_Total": "CO2总量_Mt"})

    emission_numeric_cols = [col for col in df.columns if col != "省份"]
    for col in emission_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["煤炭相关排放_Mt"] = df[COAL_FIELDS].sum(axis=1)
    df["石油相关排放_Mt"] = df[OIL_FIELDS].sum(axis=1)
    df["天然气排放_Mt"] = df["Natural_Gas"]
    df["过程排放_Mt"] = df["Process"]

    df["煤炭相关排放占比"] = safe_divide(df["煤炭相关排放_Mt"], df["CO2总量_Mt"])
    df["石油相关排放占比"] = safe_divide(df["石油相关排放_Mt"], df["CO2总量_Mt"])
    df["天然气排放占比"] = safe_divide(df["天然气排放_Mt"], df["CO2总量_Mt"])
    df["过程排放占比"] = safe_divide(df["过程排放_Mt"], df["CO2总量_Mt"])
    return df


def build_main_table(province_df: pd.DataFrame, emissions_df: pd.DataFrame) -> pd.DataFrame:
    merged = province_df.merge(emissions_df, on="省份", how="inner")

    merged["人均CO2_吨每人"] = merged["CO2总量_Mt"] * 100 / merged["人口_万人"]
    merged["碳排放强度_吨每万元GDP"] = merged["CO2总量_Mt"] * 100 / merged["GDP_亿元"]
    merged["人均GDP_万元每人"] = merged["GDP_亿元"] / merged["人口_万人"]
    merged["第二产业占比"] = merged["第二产业_亿元"] / merged["GDP_亿元"]
    merged["技术效率_万元GDP每吨CO2"] = safe_divide(
        pd.Series(1, index=merged.index), merged["碳排放强度_吨每万元GDP"]
    )

    return merged[FINAL_COLUMNS].reset_index(drop=True)


def make_validation_tables(
    province_df: pd.DataFrame, emissions_df: pd.DataFrame, main_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    province_set = set(province_df["省份"])
    emissions_set = set(emissions_df["省份"])

    province_diff = sorted(province_set - emissions_set)
    emissions_diff = sorted(emissions_set - province_set)

    match_check = pd.DataFrame(
        {
            "项目": [
                "最终主表行数",
                "2022数据中有但附件2没有的省份",
                "附件2中有但2022数据没有的省份",
            ],
            "值": [
                len(main_df),
                "、".join(province_diff) if province_diff else "无",
                "、".join(emissions_diff) if emissions_diff else "无",
            ],
        }
    )

    missing_counts = (
        main_df.isna()
        .sum()
        .rename("缺失值数量")
        .reset_index()
        .rename(columns={"index": "列名"})
    )

    descriptive_stats = main_df[DESCRIBE_COLUMNS].describe().T.reset_index()
    descriptive_stats = descriptive_stats.rename(columns={"index": "指标"})
    return match_check, missing_counts, descriptive_stats


def autofit_columns(writer: pd.ExcelWriter, sheet_names: list[str]) -> None:
    for sheet_name in sheet_names:
        worksheet = writer.sheets[sheet_name]
        for column_cells in worksheet.columns:
            max_length = 0
            column_letter = column_cells[0].column_letter
            for cell in column_cells:
                value = "" if cell.value is None else str(cell.value)
                max_length = max(max_length, len(value))
            worksheet.column_dimensions[column_letter].width = min(max(max_length + 2, 10), 32)
        worksheet.freeze_panes = "A2"


def main() -> None:
    province_df = read_province_data()
    emissions_df = read_emissions_data()
    main_df = build_main_table(province_df, emissions_df)
    match_check, missing_counts, descriptive_stats = make_validation_tables(
        province_df, emissions_df, main_df
    )

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        main_df.to_excel(writer, sheet_name="主表", index=False)
        match_check.to_excel(writer, sheet_name="省份匹配校验", index=False)
        missing_counts.to_excel(writer, sheet_name="列缺失值数量", index=False)
        descriptive_stats.to_excel(writer, sheet_name="描述性统计", index=False)
        autofit_columns(writer, ["主表", "省份匹配校验", "列缺失值数量", "描述性统计"])

    print(f"已生成: {OUTPUT_FILE}")
    print("\n省份匹配校验:")
    print(match_check.to_string(index=False))
    print("\n每列缺失值数量:")
    print(missing_counts.to_string(index=False))
    print("\n描述性统计:")
    print(descriptive_stats.to_string(index=False))


if __name__ == "__main__":
    main()
