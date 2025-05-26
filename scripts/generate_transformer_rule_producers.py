"""
generate_transformer_rule_producers.py

功能：
    将简单的映射json转换为 Vnite 中的 TransformerRule 结构。
"""

import json
import re
from paths import DATA_PROCESSED_DIR
from typing import Dict

# 输入和输出路径
INPUT_PATH = DATA_PROCESSED_DIR / "producers_kana_combined_sorted.json"
OUTPUT_PATH = DATA_PROCESSED_DIR / "producers_rule_vnite.json"


if __name__ == "__main__":
    # 加载映射数据
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        mapping: Dict[str, str] = json.load(f)

    # 转换为 RuleSet 结构
    rules = [{"match": [f"^{re.escape(k)}$"], "replace": v} for k, v in mapping.items()]

    # 构造 TransformerRule 结构
    transformer_rule = {"developers": rules}

    # 导出为 JSON 文件
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(transformer_rule, f, ensure_ascii=False, separators=(",", ":"))
