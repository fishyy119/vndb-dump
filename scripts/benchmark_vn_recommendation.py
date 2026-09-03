"""
benchmark_vn_recommendation.py

功能：
    使用 VNDB 关系表作为真值，按语言分别评测游戏推荐候选算法。
    待评测查询集从该语言的全部 VN 中按 votecount 组成：一部分取最高排名，
    另一部分从剩余 VN 中固定种子随机抽样。候选集必定包含全部查询及其真值目标，
    然后从剩余无关 VN 中抽取与上述核心候选数量相同的负例。

输出：
    两个 PNG 图片：分别对应正确开发商标注和开发商相似度置零环境。
    每张图片均为三行语言子集，每行以分组柱状图对比 NDCG@5、
    NDCG@10、NDCG@20 和 Recall；各 NDCG 均仅对存在真值的查询取平均。
"""

import argparse
import hashlib
import json
import math
import sys
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Self

import matplotlib
import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import JaroWinkler
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_DIR / "data" / "raw"
SUPPORTED_LANGUAGES = ("zh-Hans", "ja", "en")
IRRELEVANT_CANDIDATE_RATIO = 1.0
NDCG_TOP_KS = (5, 10, 20)
NDCG_MAX_TOP_K = max(NDCG_TOP_KS)
NDCG_DISCOUNTS = 1 / np.log2(np.arange(2, NDCG_MAX_TOP_K + 2))
DEVELOPER_SCENARIOS = ("clean", "no_developer")
DEVELOPER_SCENARIO_TITLES = {
    "clean": "Correct developer annotations",
    "no_developer": "Developer similarity disabled",
}


@dataclass(frozen=True)
class CandidateResult:
    """候选算法对每个查询—候选对的输出。"""

    matched: np.ndarray
    score: np.ndarray
    rank: np.ndarray


@dataclass(frozen=True)
class CandidateAlgorithm:
    """使用两个预计算相似度分量的候选算法定义。"""

    name: str
    scorer: Callable[[np.ndarray, np.ndarray], CandidateResult]


def rank_scores(score: np.ndarray) -> np.ndarray:
    """按分数降序生成每个候选的一基排名；同分时保持候选 VN ID 顺序。"""
    order = np.argsort(-score, axis=1, kind="stable")
    rank = np.empty(order.shape, dtype=np.int32)
    rank_values = np.broadcast_to(np.arange(1, score.shape[1] + 1, dtype=np.int32), order.shape)
    np.put_along_axis(rank, order, rank_values, axis=1)
    return rank


def base_score(
    developer_similarity: np.ndarray,
    title_similarity: np.ndarray,
    *,
    threshold: float,
) -> CandidateResult:
    """基线算法：开发者 Jaccard 加权标题 Jaro-Winkler。"""
    score = threshold * developer_similarity + (1 - threshold) * title_similarity
    return CandidateResult(
        matched=score >= threshold,
        score=score,
        rank=rank_scores(score),
    )


def union_score(
    developer_similarity: np.ndarray,
    title_similarity: np.ndarray,
    *,
    weight: float,
    threshold: float,
    threshold_title: float,
) -> CandidateResult:
    score = weight * developer_similarity + (1 - weight) * title_similarity
    return CandidateResult(
        matched=(developer_similarity >= threshold)
        | (title_similarity >= threshold_title)
        | (score >= weight * threshold + (1 - weight) * threshold_title),
        score=score,
        rank=rank_scores(score),
    )


CANDIDATE_ALGORITHMS = (
    CandidateAlgorithm(
        name="base@0.3",
        scorer=partial(base_score, threshold=0.3),
    ),
    CandidateAlgorithm(
        name="union@354",
        scorer=partial(union_score, weight=0.3, threshold=0.5, threshold_title=0.4),
    ),
    CandidateAlgorithm(
        name="union@454",
        scorer=partial(union_score, weight=0.4, threshold=0.5, threshold_title=0.4),
    ),
    CandidateAlgorithm(
        name="union@554",
        scorer=partial(union_score, weight=0.5, threshold=0.5, threshold_title=0.4),
    ),
    CandidateAlgorithm(
        name="union@453",
        scorer=partial(union_score, weight=0.5, threshold=0.5, threshold_title=0.3),
    ),
    CandidateAlgorithm(
        name="union@455",
        scorer=partial(union_score, weight=0.5, threshold=0.5, threshold_title=0.5),
    ),
)


@dataclass(frozen=True)
class SimilarityBlock:
    """当前查询块与全量候选间的两个相似度分量。"""

    developer: np.ndarray
    title: np.ndarray


@dataclass(frozen=True)
class QuerySample:
    """单个待评测查询。"""

    candidate_index: int
    source: str
    votecount: int
    votecount_rank: int


@dataclass
class LanguageDataset:
    """单种语言的全量候选和抽样查询。"""

    language: str
    candidate_ids: list[str]
    candidate_titles: list[str]
    candidate_developers: list[frozenset[str]]
    queries: list[QuerySample]
    truth_by_query: list[frozenset[int]]
    shared_developer_scores: list[dict[int, float]]
    universe_candidate_count: int
    irrelevant_candidate_count: int


@dataclass(frozen=True)
class Args:
    """脚本使用的类型化参数。"""

    titles: Path
    relations: Path
    developers: Path
    votecount: Path
    languages: tuple[str, ...]
    query_count: int
    top_ratio: float
    seed: int
    relation_types: tuple[str, ...] | None
    relation_official: str
    title_normalization: str
    block_size: int
    workers: int
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
            votecount=namespace.votecount,
            languages=tuple(dict.fromkeys(namespace.languages)),
            query_count=namespace.query_count,
            top_ratio=namespace.top_percent / 100.0,
            seed=namespace.seed,
            relation_types=relation_types,
            relation_official=namespace.relation_official,
            title_normalization=namespace.title_normalization,
            block_size=namespace.block_size,
            workers=namespace.workers,
            output=namespace.output,
        )

    def __post_init__(self) -> None:
        if self.query_count <= 0:
            raise ValueError("--query-count must be positive")
        if not math.isfinite(self.top_ratio) or not 0 <= self.top_ratio <= 1:
            raise ValueError("--top-percent must be a finite number between 0 and 100")
        if self.block_size <= 0:
            raise ValueError("--block-size must be positive")
        if self.workers == 0 or self.workers < -1:
            raise ValueError("--workers must be -1 or a positive integer")
        for path in (self.titles, self.relations, self.developers, self.votecount):
            if not path.is_file():
                raise FileNotFoundError(f"input file not found: {path}")

    @property
    def random_ratio(self) -> float:
        return 1.0 - self.top_ratio


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Evaluate a VN recommendation algorithm against VNDB relations")
    parser.add_argument(
        "--titles",
        type=Path,
        default=DATA_RAW_DIR / "vn_titles_20260218.csv",
        help="Path to vn_titles CSV",
    )
    parser.add_argument(
        "--relations",
        type=Path,
        default=DATA_RAW_DIR / "vn_relations_20260218.csv",
        help="Path to vn_relations CSV",
    )
    parser.add_argument(
        "--developers",
        type=Path,
        default=DATA_RAW_DIR / "vn_developers_20260218.csv",
        help="Path to vn_developers CSV",
    )
    parser.add_argument(
        "--votecount",
        type=Path,
        default=DATA_RAW_DIR / "vn_votecount_20260218.csv",
        help="Path to VN votecount CSV",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=SUPPORTED_LANGUAGES,
        default=list(SUPPORTED_LANGUAGES),
        help="Language subsets to evaluate",
    )
    parser.add_argument(
        "--query-count",
        type=int,
        default=4096,
        help=(
            "Maximum evaluation queries sampled independently for each language; "
            "capped by the available language candidates (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=50.0,
        help="Top-ranked query percentage; the remainder is sampled randomly (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random sampling seed (default: %(default)s)")
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
        "--block-size",
        type=int,
        default=128,
        help="Number of queries scored in one matrix block (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="RapidFuzz workers; -1 uses all CPU cores (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "benchmark_results" / "metrics.png",
        help="Base path used to derive the clean and no-developer images (default: %(default)s)",
    )
    return Args.from_ns(parser.parse_args())


def require_columns(frame: pd.DataFrame, expected: set[str], path: Path) -> None:
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")


def load_inputs(args: Args) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, frozenset[str]], pd.DataFrame]:
    titles = pd.read_csv(args.titles)
    relations = pd.read_csv(args.relations)
    developers = pd.read_csv(args.developers)
    votes = pd.read_csv(args.votecount)

    require_columns(titles, {"id", "lang", "title"}, args.titles)
    require_columns(relations, {"id", "vid", "relation", "official"}, args.relations)
    require_columns(developers, {"vn_id", "developers"}, args.developers)
    require_columns(votes, {"vid", "votecount", "votecount_rank"}, args.votecount)

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

    if votes["vid"].duplicated().any():
        duplicate = votes.loc[votes["vid"].duplicated(), "vid"].iloc[0]
        raise ValueError(f"duplicate votecount row: {duplicate}")

    return titles, relations, developer_map, votes.set_index("vid", verify_integrity=True)


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


def stable_language_seed(seed: int, language: str) -> int:
    digest = hashlib.sha256(f"{seed}:{language}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def sample_queries(
    candidate_ids: list[str],
    votes: pd.DataFrame,
    query_count: int,
    top_ratio: float,
    seed: int,
) -> list[QuerySample]:
    query_count = min(query_count, len(candidate_ids))
    if query_count == 0:
        raise ValueError("cannot sample queries from an empty language candidate set")

    missing_votes = [vn_id for vn_id in candidate_ids if vn_id not in votes.index]
    if missing_votes:
        preview = ", ".join(missing_votes[:5])
        raise ValueError(f"votecount is missing {len(missing_votes)} candidate VNs, including: {preview}")

    ranking = pd.DataFrame(
        {
            "candidate_index": np.arange(len(candidate_ids), dtype=np.int64),
            "vid": candidate_ids,
            "vn_number": [vn_number(vn_id) for vn_id in candidate_ids],
            "votecount": [int(votes.at[vn_id, "votecount"]) for vn_id in candidate_ids],
            "votecount_rank": [int(votes.at[vn_id, "votecount_rank"]) for vn_id in candidate_ids],
        }
    ).sort_values(["votecount_rank", "vn_number"], kind="stable")

    top_count = min(query_count, int(math.floor(query_count * top_ratio + 0.5)))
    random_count = query_count - top_count
    top_rows = ranking.iloc[:top_count]
    remaining = ranking.iloc[top_count:]

    if random_count:
        rng = np.random.default_rng(seed)
        selected_positions = rng.choice(len(remaining), size=random_count, replace=False)
        random_rows = remaining.iloc[np.sort(selected_positions)]
    else:
        random_rows = remaining.iloc[:0]

    samples = [
        QuerySample(
            candidate_index=int(row.candidate_index),
            source="top",
            votecount=int(row.votecount),
            votecount_rank=int(row.votecount_rank),
        )
        for row in top_rows.itertuples(index=False)
    ]
    samples.extend(
        QuerySample(
            candidate_index=int(row.candidate_index),
            source="random",
            votecount=int(row.votecount),
            votecount_rank=int(row.votecount_rank),
        )
        for row in random_rows.itertuples(index=False)
    )
    return samples


def filter_relations(
    relations: pd.DataFrame,
    relation_types: Sequence[str] | None,
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

    return filtered.drop_duplicates(subset=["id", "vid"])


def build_truth_by_query(
    candidate_ids: list[str],
    queries: list[QuerySample],
    relations: pd.DataFrame,
) -> list[frozenset[int]]:
    candidate_index = {vn_id: index for index, vn_id in enumerate(candidate_ids)}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in relations.itertuples(index=False):
        adjacency[row.id].add(row.vid)
        adjacency[row.vid].add(row.id)

    truth_by_query: list[frozenset[int]] = []
    for query in queries:
        query_id = candidate_ids[query.candidate_index]
        truth = {
            candidate_index[target_id]
            for target_id in adjacency.get(query_id, set())
            if target_id in candidate_index and candidate_index[target_id] != query.candidate_index
        }
        truth_by_query.append(frozenset(truth))
    return truth_by_query


def sample_candidate_pool(
    universe_candidate_ids: list[str],
    universe_queries: list[QuerySample],
    universe_truth_by_query: list[frozenset[int]],
    seed: int,
) -> tuple[list[int], list[QuerySample], list[frozenset[int]], int]:
    """保留查询及其全部真值目标，再抽取等量的无关候选。"""
    core_indices = {query.candidate_index for query in universe_queries}
    for truth_targets in universe_truth_by_query:
        core_indices.update(truth_targets)

    irrelevant_indices = [
        candidate_index for candidate_index in range(len(universe_candidate_ids)) if candidate_index not in core_indices
    ]
    requested_irrelevant_count = round(len(core_indices) * IRRELEVANT_CANDIDATE_RATIO)
    irrelevant_candidate_count = min(requested_irrelevant_count, len(irrelevant_indices))
    if irrelevant_candidate_count:
        rng = np.random.default_rng(seed)
        sampled_irrelevant = rng.choice(
            irrelevant_indices,
            size=irrelevant_candidate_count,
            replace=False,
        )
        selected_indices = core_indices | {int(candidate_index) for candidate_index in sampled_irrelevant}
    else:
        selected_indices = core_indices

    ordered_indices = sorted(selected_indices)
    remapped_index = {
        universe_candidate_index: candidate_index
        for candidate_index, universe_candidate_index in enumerate(ordered_indices)
    }
    queries = [
        QuerySample(
            candidate_index=remapped_index[query.candidate_index],
            source=query.source,
            votecount=query.votecount,
            votecount_rank=query.votecount_rank,
        )
        for query in universe_queries
    ]
    truth_by_query = [
        frozenset(remapped_index[target] for target in truth_targets) for truth_targets in universe_truth_by_query
    ]
    return ordered_indices, queries, truth_by_query, irrelevant_candidate_count


def build_shared_developer_scores(
    candidate_developers: list[frozenset[str]],
    queries: list[QuerySample],
) -> list[dict[int, float]]:
    candidates_by_developer: dict[str, list[int]] = defaultdict(list)
    for candidate_index, developer_ids in enumerate(candidate_developers):
        for developer_id in developer_ids:
            candidates_by_developer[developer_id].append(candidate_index)

    result: list[dict[int, float]] = []
    for query in queries:
        query_developers = candidate_developers[query.candidate_index]
        shared_candidates: set[int] = set()
        for developer_id in query_developers:
            shared_candidates.update(candidates_by_developer[developer_id])
        shared_candidates.discard(query.candidate_index)

        scores: dict[int, float] = {}
        for candidate_index in shared_candidates:
            candidate_developer_ids = candidate_developers[candidate_index]
            intersection = len(query_developers & candidate_developer_ids)
            union = len(query_developers | candidate_developer_ids)
            scores[candidate_index] = intersection / union
        result.append(scores)
    return result


def build_language_dataset(
    language: str,
    titles: pd.DataFrame,
    votes: pd.DataFrame,
    developer_map: dict[str, frozenset[str]],
    relations: pd.DataFrame,
    args: Args,
) -> LanguageDataset:
    language_titles = titles.loc[titles["lang"] == language, ["id", "title"]].copy()
    if language_titles["id"].duplicated().any():
        duplicate = language_titles.loc[language_titles["id"].duplicated(), "id"].iloc[0]
        raise ValueError(f"duplicate {language} title row: {duplicate}")

    language_titles["vn_number"] = language_titles["id"].map(vn_number)
    language_titles = language_titles.sort_values("vn_number", kind="stable")
    universe_candidate_ids = language_titles["id"].tolist()
    universe_raw_titles = language_titles["title"].tolist()
    universe_candidate_titles = [normalize_title(title, args.title_normalization) for title in universe_raw_titles]
    empty_titles = [
        vn_id for vn_id, title in zip(universe_candidate_ids, universe_candidate_titles, strict=True) if not title
    ]
    if empty_titles:
        preview = ", ".join(empty_titles[:5])
        raise ValueError(f"{language} contains {len(empty_titles)} empty titles, including: {preview}")

    query_seed = stable_language_seed(args.seed, language)
    universe_queries = sample_queries(
        universe_candidate_ids,
        votes,
        args.query_count,
        args.top_ratio,
        query_seed,
    )
    universe_truth_by_query = build_truth_by_query(universe_candidate_ids, universe_queries, relations)
    candidate_seed = stable_language_seed(args.seed, f"{language}:irrelevant-candidates")
    selected_indices, queries, truth_by_query, irrelevant_candidate_count = sample_candidate_pool(
        universe_candidate_ids,
        universe_queries,
        universe_truth_by_query,
        candidate_seed,
    )
    candidate_ids = [universe_candidate_ids[index] for index in selected_indices]
    candidate_titles = [universe_candidate_titles[index] for index in selected_indices]
    candidate_developers = [developer_map.get(vn_id, frozenset()) for vn_id in candidate_ids]
    shared_developer_scores = build_shared_developer_scores(candidate_developers, queries)

    return LanguageDataset(
        language=language,
        candidate_ids=candidate_ids,
        candidate_titles=candidate_titles,
        candidate_developers=candidate_developers,
        queries=queries,
        truth_by_query=truth_by_query,
        shared_developer_scores=shared_developer_scores,
        universe_candidate_count=len(universe_candidate_ids),
        irrelevant_candidate_count=irrelevant_candidate_count,
    )


# 相似度计算层：所有候选算法共享同一份计算结果，禁止读取 truth_by_query。
def calculate_similarity_block(
    dataset: LanguageDataset,
    query_start: int,
    query_stop: int,
    workers: int,
) -> SimilarityBlock:
    """
    预先返回形状均为 [本批查询数, 抽样候选数] 的开发者 Jaccard 与标题 Jaro-Winkler 矩阵。
    """
    query_titles = [
        dataset.candidate_titles[dataset.queries[query_index].candidate_index]
        for query_index in range(query_start, query_stop)
    ]
    title_similarities = process.cdist(
        query_titles,
        dataset.candidate_titles,
        scorer=JaroWinkler.normalized_similarity,
        workers=workers,
        dtype=np.float32,
    )
    developer_similarities = np.zeros_like(title_similarities)

    for local_index, query_index in enumerate(range(query_start, query_stop)):
        for candidate_index, developer_similarity in dataset.shared_developer_scores[query_index].items():
            developer_similarities[local_index, candidate_index] = developer_similarity
        self_index = dataset.queries[query_index].candidate_index
        developer_similarities[local_index, self_index] = -math.inf
        title_similarities[local_index, self_index] = -math.inf
    return SimilarityBlock(developer=developer_similarities, title=title_similarities)


# 候选算法插槽：算法直接组合已经计算完毕的两个相似度矩阵。
def score_candidate_algorithm(
    similarities: SimilarityBlock,
    algorithm: CandidateAlgorithm,
) -> CandidateResult:
    return algorithm.scorer(similarities.developer, similarities.title)


def remove_developer_similarity(similarities: SimilarityBlock) -> SimilarityBlock:
    """将开发商相似度置零，同时保留用于排除查询自身的负无穷标记。"""
    developer = np.zeros_like(similarities.developer)
    developer[np.isneginf(similarities.title)] = -math.inf
    return SimilarityBlock(developer=developer, title=similarities.title)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def top_ranked_matches(matched: np.ndarray, rank: np.ndarray, top_k: int) -> np.ndarray:
    """从通过阈值的候选中，按算法排名返回前 K 个候选下标。"""
    matched_indices = np.flatnonzero(matched)
    if len(matched_indices) > top_k:
        selected_positions = np.argpartition(rank[matched_indices], top_k - 1)[:top_k]
        matched_indices = matched_indices[selected_positions]
    return matched_indices[np.argsort(rank[matched_indices], kind="stable")]


def calculate_ndcg(
    ranked_candidates: np.ndarray,
    truth_targets: frozenset[int],
    top_k: int,
) -> float:
    """计算单个查询在阈值过滤后推荐列表上的 NDCG@K。"""
    ideal_hit_count = min(len(truth_targets), top_k)
    if ideal_hit_count == 0:
        return 0.0

    top_candidates = ranked_candidates[:top_k]
    relevance = np.fromiter(
        (candidate_index in truth_targets for candidate_index in top_candidates),
        dtype=np.float64,
        count=len(top_candidates),
    )
    discounted_gain = float(np.dot(relevance, NDCG_DISCOUNTS[: len(top_candidates)]))
    ideal_discounted_gain = float(np.sum(NDCG_DISCOUNTS[:ideal_hit_count]))
    return discounted_gain / ideal_discounted_gain


def evaluate_language(
    dataset: LanguageDataset,
    algorithms: Sequence[CandidateAlgorithm],
    block_size: int,
    workers: int,
) -> dict[str, list[dict[str, int | float | str]]]:
    query_count = len(dataset.queries)
    truth_query_count = sum(bool(items) for items in dataset.truth_by_query)
    candidate_count = len(dataset.candidate_ids)
    ndcg_totals = {
        scenario: {algorithm.name: {top_k: 0.0 for top_k in NDCG_TOP_KS} for algorithm in algorithms}
        for scenario in DEVELOPER_SCENARIOS
    }
    recall_true_positives = {
        scenario: {algorithm.name: 0 for algorithm in algorithms} for scenario in DEVELOPER_SCENARIOS
    }

    with tqdm(total=query_count, desc=f"[{dataset.language}] scoring", unit="query", dynamic_ncols=True) as progress:
        for query_start in range(0, query_count, block_size):
            query_stop = min(query_start + block_size, query_count)
            similarities = calculate_similarity_block(dataset, query_start, query_stop, workers)
            expected_shape = (query_stop - query_start, candidate_count)
            if similarities.developer.shape != expected_shape or similarities.title.shape != expected_shape:
                raise ValueError(f"similarity layer returned an invalid matrix shape; expected {expected_shape}")

            scenario_similarities = {
                "clean": similarities,
                "no_developer": remove_developer_similarity(similarities),
            }
            for scenario, scenario_similarity in scenario_similarities.items():
                for algorithm in algorithms:
                    candidate_result = score_candidate_algorithm(scenario_similarity, algorithm)
                    result_shapes = {
                        candidate_result.matched.shape,
                        candidate_result.score.shape,
                        candidate_result.rank.shape,
                    }
                    if result_shapes != {expected_shape}:
                        raise ValueError(f"candidate algorithm {algorithm.name} returned an invalid result shape")
                    if candidate_result.matched.dtype != np.bool_:
                        raise ValueError(f"candidate algorithm {algorithm.name} returned a non-boolean matched matrix")
                    for local_index, query_index in enumerate(range(query_start, query_stop)):
                        predictions = candidate_result.matched[local_index]
                        candidate_rank = candidate_result.rank[local_index]
                        truth_targets = dataset.truth_by_query[query_index]
                        ranked_candidates = top_ranked_matches(predictions, candidate_rank, NDCG_MAX_TOP_K)
                        for top_k in NDCG_TOP_KS:
                            ndcg_totals[scenario][algorithm.name][top_k] += calculate_ndcg(
                                ranked_candidates,
                                truth_targets,
                                top_k,
                            )
                        recall_true_positives[scenario][algorithm.name] += sum(
                            bool(predictions[target]) for target in truth_targets
                        )

            progress.update(query_stop - query_start)

    truth_positives = sum(len(items) for items in dataset.truth_by_query)
    results: dict[str, list[dict[str, int | float | str]]] = {scenario: [] for scenario in DEVELOPER_SCENARIOS}
    for scenario in DEVELOPER_SCENARIOS:
        for algorithm in algorithms:
            algorithm_metrics: dict[str, int | float | str] = {
                "language": dataset.language,
                "algorithm": algorithm.name,
                "query_count": query_count,
                "truth_query_count": truth_query_count,
                "truth_coverage": safe_divide(truth_query_count, query_count),
                "candidate_count": candidate_count,
                "universe_candidate_count": dataset.universe_candidate_count,
                "irrelevant_candidate_count": dataset.irrelevant_candidate_count,
                "recall": safe_divide(recall_true_positives[scenario][algorithm.name], truth_positives),
            }
            for top_k in NDCG_TOP_KS:
                algorithm_metrics[f"ndcg_at_{top_k}"] = safe_divide(
                    ndcg_totals[scenario][algorithm.name][top_k],
                    truth_query_count,
                )
            results[scenario].append(algorithm_metrics)
    return results


def plot_metrics(
    metrics: pd.DataFrame,
    languages: Sequence[str],
    algorithms: Sequence[CandidateAlgorithm],
    output_path: Path,
    scenario_title: str,
) -> None:
    color_map = plt.get_cmap("tab10")
    colors = [color_map(index % color_map.N) for index in range(len(algorithms))]
    figure, axes = plt.subplots(
        nrows=len(languages),
        ncols=1,
        figsize=(12, 3.6 * len(languages)),
        sharey=True,
        squeeze=False,
    )
    metric_names = tuple(f"ndcg_at_{top_k}" for top_k in NDCG_TOP_KS) + ("recall",)
    metric_labels = tuple(f"NDCG@{top_k}" for top_k in NDCG_TOP_KS) + ("Recall",)
    x_positions = np.arange(len(metric_names))
    algorithm_names = [algorithm.name for algorithm in algorithms]
    group_width = 0.84
    bar_width = group_width / len(algorithms)

    for row_index, language in enumerate(languages):
        language_metrics = metrics.loc[metrics["language"] == language].set_index("algorithm")
        language_metrics = language_metrics.loc[algorithm_names]
        axis = axes[row_index, 0]
        for algorithm_index, algorithm_name in enumerate(algorithm_names):
            offset = (algorithm_index - (len(algorithms) - 1) / 2) * bar_width
            values = language_metrics.loc[algorithm_name, list(metric_names)].to_numpy(dtype=float)
            bars = axis.bar(
                x_positions + offset,
                values,
                width=bar_width,
                color=colors[algorithm_index],
                label=algorithm_name,
            )
            axis.bar_label(
                bars,
                labels=[f"{value:.4f}" for value in values],
                padding=2,
                fontsize=7,
                rotation=90,
            )
        axis.set_xticks(x_positions, metric_labels)
        axis.set_ylim(0, 1.12)
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylabel(f"{language}\nScore")
        truth_query_count = int(language_metrics["truth_query_count"].iloc[0])
        query_count = int(language_metrics["query_count"].iloc[0])
        truth_coverage = float(language_metrics["truth_coverage"].iloc[0])
        candidate_count = int(language_metrics["candidate_count"].iloc[0])
        universe_candidate_count = int(language_metrics["universe_candidate_count"].iloc[0])
        irrelevant_candidate_count = int(language_metrics["irrelevant_candidate_count"].iloc[0])
        axis.set_title(
            f"Truth coverage: {truth_query_count:,}/{query_count:,} ({truth_coverage:.2%}) | "
            f"Candidate pool: {candidate_count:,}/{universe_candidate_count:,} "
            f"({irrelevant_candidate_count:,} random negatives)",
            loc="left",
            fontsize=10,
        )

    figure.suptitle(f"VNDB recommendation benchmark: {scenario_title}", fontsize=15)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        title="Candidate algorithm",
        loc="center right",
        bbox_to_anchor=(0.995, 0.5),
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 0.8, 0.96))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def scenario_output_path(base_path: Path, scenario: str) -> Path:
    return base_path.with_name(f"{base_path.stem}_{scenario}{base_path.suffix}")


def main() -> int:
    try:
        args = parse_args()
        titles, relations, developer_map, votes = load_inputs(args)
        filtered_relations = filter_relations(
            relations,
            args.relation_types,
            args.relation_official,
        )
        algorithms = CANDIDATE_ALGORITHMS

        results = {scenario: [] for scenario in DEVELOPER_SCENARIOS}
        for language in args.languages:
            dataset = build_language_dataset(
                language,
                titles,
                votes,
                developer_map,
                filtered_relations,
                args,
            )
            language_results = evaluate_language(dataset, algorithms, args.block_size, args.workers)
            for scenario in DEVELOPER_SCENARIOS:
                results[scenario].extend(language_results[scenario])

        args.output.parent.mkdir(parents=True, exist_ok=True)
        for scenario in DEVELOPER_SCENARIOS:
            output_path = scenario_output_path(args.output, scenario)
            metrics = pd.DataFrame(results[scenario])
            plot_metrics(
                metrics,
                args.languages,
                algorithms,
                output_path,
                DEVELOPER_SCENARIO_TITLES[scenario],
            )
            print(f"Image written to: {output_path.resolve()}")
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
