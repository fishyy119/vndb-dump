# vndb-dump

## 数据来源

本仓库存储的数据来源于[Database Dumps | vndb](https://vndb.org/d14)，快照日期为 2025 年 5 月 21 日。

## 文件结构

### 数据文件夹 `data/`

* `raw/`：原始数据文件，直接从 VNDB 导出，未经处理。
  * `producers_20250521.csv`：包含所有制作商的原始信息。
  
* `processed/`：数据清洗与提取后的结果：
  * `producers_*.json`：将名称为纯假名（允许包含“株式会社”四个汉字）的制作商名称映射为对应的拉丁转写。
    * `producers_katakana.json`：名称为纯片假名，保持原始顺序。
    * `producers_hiragana.json`：名称为纯平假名，保持原始顺序。
    * `producers_kana_mix.json`：名称为片假名和平假名混合，保持原始顺序。
    * `producers_kana_combined_sorted.json`：上述三者合并后的结果，按拉丁字母名称排序。
  

### 脚本文件夹 `scripts/`

* `extract_kana_mappings.py`：用于筛选日文名称并根据假名类型分类，导出对应的映射 JSON 文件。详细说明见脚本内文注释。

