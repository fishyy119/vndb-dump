"""
extract_kana_mappings.py

功能：
    该脚本加载 VNDB 导出的 producers.csv 文件，筛选语言为日语（lang=ja）的制作商名称，
    并根据名称中的假名类型（纯片假名、纯平假名、混合假名）进行分类。
    支持忽略名称中的“株式会社”四个汉字。
    最终导出4份 JSON 文件，键为日文名称，值为对应的拉丁字母名称。

输出：
    - 处理后按假名类型分类的 JSON 文件，存储于 data/processed 目录，顺序为源数据顺序
    - 上述三个文件的合并版，按拉丁字母名称排序
"""

import pandas as pd
import re
import json
from pathlib import Path
from paths import DATA_RAW_DIR, DATA_PROCESSED_DIR
from enum import auto, Enum
from typing import Dict


katakana_re = re.compile(r"^[\u30A0-\u30FF・ー]+$")
hiragana_re = re.compile(r"^[\u3040-\u309F・ー]+$")
kana_mixed_re = re.compile(r"^[\u3040-\u309F\u30A0-\u30FF・ー]+$")


class NamePattern(Enum):
    KATAKANA = auto()
    HIRAGANA = auto()
    KANA_MIX = auto()
    NORMAL = auto()


def classify_kana(name: str) -> NamePattern:
    # 去除所有“株式会社”字样
    name_filtered = name.replace("株式会社", "")

    if katakana_re.fullmatch(name_filtered):
        return NamePattern.KATAKANA
    if hiragana_re.fullmatch(name_filtered):
        return NamePattern.HIRAGANA
    if kana_mixed_re.fullmatch(name_filtered):
        return NamePattern.KANA_MIX

    return NamePattern.NORMAL


# 导出为json，键是name，值是latin
def to_json(df: pd.DataFrame, file: Path):
    # 只导出name和latin列，转成字典
    d: Dict[str, str] = df.set_index("name")["latin"].dropna().to_dict()
    with open(file, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(d)} entries to {file}")


def process_producers(input_csv: Path, output_dir: Path):
    df = pd.read_csv(input_csv)
    df_ja = df[df["lang"] == "ja"].copy()
    df_ja["kana_type"] = df_ja["name"].apply(lambda x: classify_kana(x).name)

    dfs = []
    for pattern in [NamePattern.KATAKANA, NamePattern.HIRAGANA, NamePattern.KANA_MIX]:
        filtered = df_ja[df_ja["kana_type"] == pattern.name]
        to_json(filtered, output_dir / f"producers_{pattern.name.lower()}.json")
        dfs.append(filtered)

    # 合并三种假名数据，去重并按name排序
    df_combined = pd.concat(dfs).drop_duplicates(subset="name")
    df_combined_sorted = df_combined.sort_values(by="latin")

    to_json(df_combined_sorted, output_dir / "producers_kana_combined_sorted.json")


if __name__ == "__main__":
    process_producers(DATA_RAW_DIR / "producers_20250521.csv", DATA_PROCESSED_DIR)
