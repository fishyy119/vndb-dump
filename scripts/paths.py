from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parents[1]

# 数据路径
DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
