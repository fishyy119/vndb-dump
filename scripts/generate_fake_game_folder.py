"""
generate_fake_game_folder.py

功能：
    该脚本加载 VNDB 导出的 vn_titles.csv 文件，创建虚假的游戏库文件结构

输出：
    生成一个游戏库，第一级目录为标题，内部含有`fake.exe`文件
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from paths import DATA_RAW_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fake game library from vn_titles.csv")

    parser.add_argument("--csv", type=str, default="vn_titles_20260218.csv", help="Name of vn_titles.csv")
    parser.add_argument("--seed", type=int, default=42, help="Random sampling seed (default: 42)")
    parser.add_argument("--num", type=int, required=True, help="Number of games to sample")
    parser.add_argument("--lang", type=str, required=True, help="Target language (e.g. en, ja, zh-Hans)")
    parser.add_argument("--output", type=str, default="fake_game", help="Output directory (default: fake_game)")

    return parser.parse_args()


def sanitize_filename(name: str) -> str:
    """
    Remove characters invalid for most file systems.
    """
    invalid_chars = '<>:"/\\|?*'
    for ch in invalid_chars:
        name = name.replace(ch, "_")
    return name.strip()


def main():
    args = parse_args()

    csv_path = DATA_RAW_DIR / args.csv
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # 检查语言
    available_langs = sorted(df["lang"].unique())

    if args.lang not in available_langs:
        print("Error: target language not found in CSV.")
        print("Available languages:")
        for lang in available_langs:
            print(f"  - {lang}")
        sys.exit(1)

    # 筛选目标语言
    df_lang = df[df["lang"] == args.lang]

    # 按 id 去重（同一 id 可能有多语言）
    df_lang = df_lang.drop_duplicates(subset=["id"])

    unique_ids = df_lang["id"].unique()

    if args.num > len(unique_ids):
        print(
            f"Warning: requested sample size ({args.num}) "
            f"exceeds available games ({len(unique_ids)}). "
            f"Using all available."
        )
        sample_size = len(unique_ids)
    else:
        sample_size = args.num

    sampled_ids = pd.Series(unique_ids).sample(n=sample_size, random_state=args.seed).tolist()

    df_sampled = df_lang[df_lang["id"].isin(sampled_ids)]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df_sampled.iterrows():
        title = row["title"]

        if pd.isna(title) or not title:
            continue

        folder_name = sanitize_filename(title)
        game_dir = output_dir / folder_name
        game_dir.mkdir(parents=True, exist_ok=True)

        fake_exe_path = game_dir / "fake.exe"
        fake_exe_path.write_text("This is a fake executable file.\n")

    print(f"Generated {len(df_sampled)} fake game folders in: {output_dir}")


if __name__ == "__main__":
    main()
