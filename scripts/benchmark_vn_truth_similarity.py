"""
benchmark_vn_truth_similarity.py

功能：
    按语言统计 VNDB 真值关系对的标题 Jaro-Winkler 相似度与开发商 Jaccard 相似度。
    关系按无向 VN 对去重，与推荐 benchmark 将关系升级为双向边的口径一致。

输出：
    仅输出一张 PNG 柱状图。每种语言占一行，对比标题 Jaro-Winkler、全量开发商
    Jaccard 和标注完整子集 Jaccard 的非零极值、分位值、平均值等统计量。
    Zero rate 和 One rate 仍使用未过滤的全量真值对计算。
"""

import argparse
import json
import math
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import matplotlib
import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_DIR / "data" / "raw"
SUPPORTED_LANGUAGES = ("zh-Hans", "ja", "en")
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
PLOT_STATISTICS = (
    ("min", "Min"),
    ("p05", "P05"),
    ("p10", "P10"),
    ("p25", "P25"),
    ("p50", "Median"),
    ("mean", "Mean"),
    ("std", "Std"),
    ("p75", "P75"),
    ("p90", "P90"),
    ("p95", "P95"),
    ("max", "Max"),
    ("zero_rate", "Zero rate (all)"),
    ("one_rate", "One rate (all)"),
)
PLOT_METRICS = (
    ("title_jaro_winkler", "Title Jaro-Winkler"),
    ("developer_jaccard_all", "Developer Jaccard (all)"),
    ("developer_jaccard_labeled", "Developer Jaccard (labeled)"),
)


@dataclass(frozen=True)
class Args:
    """脚本使用的类型化参数。"""

    titles: Path
    relations: Path
    developers: Path
    languages: tuple[str, ...]
    relation_types: tuple[str, ...] | None
    relation_official: str
    title_normalization: str
    output: Path

    @classmethod
    def from_ns(cls, namespace: argparse.Namespace) -> Self:
        """将 argparse 的原始字段聚合为脚本直接使用的参数。"""
        relation_types = None
        if namespace.relation_types is not None:
            relation_types = tuple(dict.fromkeys(namespace.relation_types))
        return cls(
            titles=namespace.titles,
            relations=namespace.relations,
            developers=namespace.developers,
            languages=tuple(dict.fromkeys(namespace.languages)),
            relation_types=relation_types,
            relation_official=namespace.relation_official,
            title_normalization=namespace.title_normalization,
            output=namespace.output,
        )

    def __post_init__(self) -> None:
        for path in (self.titles, self.relations, self.developers):
            if not path.is_file():
                raise FileNotFoundError(f"input file not found: {path}")
        if self.output.suffix.casefold() != ".png":
            raise ValueError("--output must use the .png extension")


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Describe title and developer similarities on VNDB truth pairs")
    parser.add_argument(
        "--titles",
        type=Path,
        default=DATA_RAW_DIR / "vn_titles_20260218.csv",
        help="Path to vn_titles CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--relations",
        type=Path,
        default=DATA_RAW_DIR / "vn_relations_20260218.csv",
        help="Path to vn_relations CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--developers",
        type=Path,
        default=DATA_RAW_DIR / "vn_developers_20260218.csv",
        help="Path to vn_developers CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=SUPPORTED_LANGUAGES,
        default=list(SUPPORTED_LANGUAGES),
        help="Language truth subsets to describe (default: %(default)s)",
    )
    parser.add_argument(
        "--relation-types",
        nargs="+",
        default=None,
        metavar="TYPE",
        help="Relation types included in truth; omit to include all types",
    )
    parser.add_argument(
        "--relation-official",
        choices=("all", "true", "false"),
        default="all",
        help="Official flag included in truth (default: %(default)s)",
    )
    parser.add_argument(
        "--title-normalization",
        choices=("basic", "none"),
        default="basic",
        help="Title preprocessing: NFKC/casefold/whitespace normalization or none (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "benchmark_results" / "truth_similarity.png",
        help="Path of the output image (default: %(default)s)",
    )
    return Args.from_ns(parser.parse_args())


def require_columns(frame: pd.DataFrame, expected: set[str], path: Path) -> None:
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")


def load_inputs(args: Args) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, frozenset[str]]]:
    titles = pd.read_csv(args.titles)
    relations = pd.read_csv(args.relations)
    developers = pd.read_csv(args.developers)

    require_columns(titles, {"id", "lang", "title"}, args.titles)
    require_columns(relations, {"id", "vid", "relation", "official"}, args.relations)
    require_columns(developers, {"vn_id", "developers"}, args.developers)

    developer_map: dict[str, frozenset[str]] = {}
    for row in developers.itertuples(index=False):
        try:
            value = json.loads(row.developers)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid developers JSON for {row.vn_id}") from exc
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"developers must be a JSON string list for {row.vn_id}")
        if row.vn_id in developer_map:
            raise ValueError(f"duplicate developer row: {row.vn_id}")
        developer_map[row.vn_id] = frozenset(value)

    return titles, relations, developer_map


def normalize_title(title: object, mode: str) -> str:
    if pd.isna(title):
        return ""
    value = str(title)
    if mode == "none":
        return value
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def vn_number(vn_id: str) -> int:
    if len(vn_id) < 2 or vn_id[0] != "v" or not vn_id[1:].isdigit():
        raise ValueError(f"invalid VN id: {vn_id}")
    return int(vn_id[1:])


def filter_relations(
    relations: pd.DataFrame,
    relation_types: tuple[str, ...] | None,
    official_mode: str,
) -> pd.DataFrame:
    available_types = sorted(relations["relation"].dropna().unique().tolist())
    selected_types = available_types if relation_types is None else sorted(set(relation_types))
    unknown_types = set(selected_types) - set(available_types)
    if unknown_types:
        raise ValueError(f"unknown relation types: {', '.join(sorted(unknown_types))}")

    filtered = relations.loc[relations["relation"].isin(selected_types)].copy()
    if official_mode != "all":
        official = filtered["official"]
        if official.dtype != bool:
            normalized = official.astype(str).str.casefold().map({"true": True, "false": False})
            if normalized.isna().any():
                raise ValueError("relation official column contains values other than true/false")
            official = normalized
        filtered = filtered.loc[official == (official_mode == "true")]
    return filtered


def build_undirected_truth_pairs(relations: pd.DataFrame) -> list[tuple[str, str]]:
    """将导出的单向边折叠为唯一无向真值对。"""
    pairs: set[tuple[str, str]] = set()
    for row in relations.itertuples(index=False):
        left_number = vn_number(row.id)
        right_number = vn_number(row.vid)
        if left_number == right_number:
            continue
        if left_number < right_number:
            pairs.add((row.id, row.vid))
        else:
            pairs.add((row.vid, row.id))
    return sorted(pairs, key=lambda pair: (vn_number(pair[0]), vn_number(pair[1])))


def developer_jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def describe_metric(name: str, values: np.ndarray) -> dict[str, int | float | str]:
    if values.size == 0:
        raise ValueError(f"cannot describe empty metric: {name}")
    nonzero_values = values[values != 0]
    result: dict[str, int | float | str] = {
        "metric": name,
        "count": int(nonzero_values.size),
        "zero_rate": float(np.mean(values == 0)),
        "one_rate": float(np.mean(values == 1)),
    }
    if nonzero_values.size == 0:
        result.update({name: math.nan for name, _ in PLOT_STATISTICS if name not in result})
        return result

    result.update(
        {
            "mean": float(np.mean(nonzero_values)),
            "std": float(np.std(nonzero_values)),
            "min": float(np.min(nonzero_values)),
        }
    )
    for quantile, value in zip(QUANTILES, np.quantile(nonzero_values, QUANTILES), strict=True):
        result[f"p{round(quantile * 100):02d}"] = float(value)
    result.update(
        {
            "max": float(np.max(nonzero_values)),
        }
    )
    return result


def analyze_language(
    language: str,
    titles: pd.DataFrame,
    truth_pairs: list[tuple[str, str]],
    developer_map: dict[str, frozenset[str]],
    title_normalization: str,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    language_titles = titles.loc[titles["lang"] == language, ["id", "title"]].copy()
    if language_titles["id"].duplicated().any():
        duplicate = language_titles.loc[language_titles["id"].duplicated(), "id"].iloc[0]
        raise ValueError(f"duplicate {language} title row: {duplicate}")

    title_map = {
        row.id: normalize_title(row.title, title_normalization)
        for row in language_titles.itertuples(index=False)
    }
    empty_titles = [vn_id for vn_id, title in title_map.items() if not title]
    if empty_titles:
        preview = ", ".join(empty_titles[:5])
        raise ValueError(f"{language} contains {len(empty_titles)} empty titles, including: {preview}")

    language_pairs = [pair for pair in truth_pairs if pair[0] in title_map and pair[1] in title_map]
    if not language_pairs:
        raise ValueError(f"{language} contains no truth pairs after filtering")

    title_similarities = np.empty(len(language_pairs), dtype=np.float64)
    developer_similarities = np.empty(len(language_pairs), dtype=np.float64)
    labeled_developer_similarities: list[float] = []
    left_missing_count = 0
    right_missing_count = 0
    both_missing_count = 0

    progress = tqdm(language_pairs, desc=f"[{language}] truth similarities", unit="pair", dynamic_ncols=True)
    for index, (left_id, right_id) in enumerate(progress):
        title_similarities[index] = JaroWinkler.normalized_similarity(title_map[left_id], title_map[right_id])

        left_developers = developer_map.get(left_id, frozenset())
        right_developers = developer_map.get(right_id, frozenset())
        developer_similarity = developer_jaccard(left_developers, right_developers)
        developer_similarities[index] = developer_similarity
        if left_developers and right_developers:
            labeled_developer_similarities.append(developer_similarity)
        elif not left_developers and not right_developers:
            both_missing_count += 1
        elif not left_developers:
            left_missing_count += 1
        else:
            right_missing_count += 1

    labeled_values = np.asarray(labeled_developer_similarities, dtype=np.float64)
    metric_rows = [
        describe_metric("title_jaro_winkler", title_similarities),
        describe_metric("developer_jaccard_all", developer_similarities),
    ]
    if labeled_values.size:
        metric_rows.append(describe_metric("developer_jaccard_labeled", labeled_values))

    pair_count = len(language_pairs)
    coverage = {
        "candidate_count": len(title_map),
        "truth_pair_count": pair_count,
        "both_developers_labeled": len(labeled_developer_similarities),
        "one_developer_missing": left_missing_count + right_missing_count,
        "both_developers_missing": both_missing_count,
        "both_developers_labeled_rate": len(labeled_developer_similarities) / pair_count,
    }
    return pd.DataFrame(metric_rows), coverage


def plot_statistics(
    results: dict[str, tuple[pd.DataFrame, dict[str, int | float]]],
    languages: tuple[str, ...],
    output_path: Path,
) -> None:
    colors = ("#4C78A8", "#F58518", "#54A24B")
    figure, axes = plt.subplots(
        nrows=len(languages),
        ncols=1,
        figsize=(19, 4 * len(languages)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    statistic_names = [name for name, _ in PLOT_STATISTICS]
    statistic_labels = [label for _, label in PLOT_STATISTICS]
    x_positions = np.arange(len(PLOT_STATISTICS))
    group_width = 0.84
    bar_width = group_width / len(PLOT_METRICS)

    for row_index, language in enumerate(languages):
        statistics, coverage = results[language]
        indexed_statistics = statistics.set_index("metric")
        axis = axes[row_index, 0]
        for metric_index, (metric_name, metric_label) in enumerate(PLOT_METRICS):
            offset = (metric_index - (len(PLOT_METRICS) - 1) / 2) * bar_width
            if metric_name in indexed_statistics.index:
                values = indexed_statistics.loc[metric_name, statistic_names].to_numpy(dtype=float)
            else:
                values = np.full(len(statistic_names), np.nan)
            bars = axis.bar(
                x_positions + offset,
                values,
                width=bar_width,
                color=colors[metric_index],
                label=metric_label,
            )
            axis.bar_label(
                bars,
                labels=["" if np.isnan(value) else f"{value:.4f}" for value in values],
                padding=2,
                fontsize=6,
                rotation=90,
            )

        axis.set_ylim(0, 1.13)
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylabel(f"{language}\nValue")
        axis.set_xticks(x_positions, statistic_labels, rotation=30, ha="right")
        axis.tick_params(axis="x", labelbottom=True)
        axis.set_title(
            "Candidates: {candidate_count:,} | Truth pairs: {truth_pair_count:,} | "
            "Both developers labeled: {both_developers_labeled_rate:.2%}".format(**coverage),
            loc="left",
            fontsize=10,
        )
    figure.suptitle(
        "VNDB truth-pair similarity statistics\n"
        "Distribution statistics exclude zero scores; boundary rates use all truth pairs",
        fontsize=15,
    )
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        title="Similarity metric",
        loc="center right",
        bbox_to_anchor=(0.995, 0.5),
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 0.82, 0.93))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    try:
        args = parse_args()
        titles, relations, developer_map = load_inputs(args)
        filtered_relations = filter_relations(relations, args.relation_types, args.relation_official)
        truth_pairs = build_undirected_truth_pairs(filtered_relations)
        if not truth_pairs:
            raise ValueError("no truth pairs remain after relation filtering")

        results: dict[str, tuple[pd.DataFrame, dict[str, int | float]]] = {}
        for language in args.languages:
            statistics, coverage = analyze_language(
                language,
                titles,
                truth_pairs,
                developer_map,
                args.title_normalization,
            )
            results[language] = (statistics, coverage)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        plot_statistics(results, args.languages, args.output)
        return 0
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
