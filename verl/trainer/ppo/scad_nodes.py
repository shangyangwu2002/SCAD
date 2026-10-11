"""Deterministic cross-rollout prefix-tree construction for SCAD.

The semantic key of a Worker result is

    (subtask_cluster, answer_type, canonical_answer)

and a trie node is that key under a particular parent.  Consequently, equal
subtasks/answers merge only when their complete prefixes agree.  The mapping
is frozen once per question-level rollout group before batch padding or
reordering and is exported through ``DataProto.non_tensor_batch``.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from dateutil import parser as date_parser


_IGNORED_ENTITY_WORDS = {
    "a",
    "an",
    "the",
    "of",
    "on",
    "in",
    "and",
    "&",
    "for",
    "to",
    "by",
    "with",
}
_ABSTAIN = re.compile(
    r"\b(?:not found|no specific information|could not find|"
    r"unable to determine|insufficient)\b|"
    r"(?:无法确定|未找到|没有找到)",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}
_NUMBER = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
_DATEISH = re.compile(
    r"\b(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|"
    r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})\b",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_RETURN_WRAPPER = re.compile(
    r"^\s*branch has finished its task(?:, the returned message is|\. the last message was)\s*:\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CanonicalAnswer:
    answer_type: str
    value: str
    normalized_text: str


@dataclass(frozen=True)
class NodeMergeConfig:
    semantic: bool = True
    subtask_similarity_threshold: float = 0.83
    entity_similarity_threshold: float = 0.90
    embedding_url: str = "http://127.0.0.1:8000/embed"
    embedding_timeout_s: float = 300.0
    embedding_batch_size: int = 64
    audit_dir: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "NodeMergeConfig":
        value = value or {}
        return cls(
            semantic=bool(value.get("semantic", True)),
            subtask_similarity_threshold=float(value.get("subtask_similarity_threshold", 0.83)),
            entity_similarity_threshold=float(value.get("entity_similarity_threshold", 0.90)),
            embedding_url=str(value.get("embedding_url", "http://127.0.0.1:8000/embed")),
            embedding_timeout_s=float(value.get("embedding_timeout_s", 300.0)),
            embedding_batch_size=int(value.get("embedding_batch_size", 64)),
            audit_dir=value.get("audit_dir") or None,
        )


@dataclass(frozen=True)
class _Step:
    index: int
    subtask: str
    answer: str


@dataclass(frozen=True)
class _CanonicalStep:
    index: int
    subtask: str
    answer: str
    normalized_subtask: str
    subtask_cluster: str
    answer_type: str
    answer_value: str
    answer_cluster: str

    @property
    def semantic_key(self) -> tuple[str, str, str]:
        return self.subtask_cluster, self.answer_type, self.answer_cluster


@dataclass
class NodeMergeResult:
    metadata: dict[str, np.ndarray]
    metrics: dict[str, float]
    audit: dict[str, Any]


EmbeddingFunction = Callable[[list[str]], np.ndarray]


def norm_text(value: str) -> str:
    """Normalize text with the same recipe as ``local_search.em_score``."""

    text = "".join(
        char
        for char in unicodedata.normalize("NFKD", str(value or ""))
        if not unicodedata.combining(char)
    ).lower()
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = re.sub(r"[“”\"'`]+", "", text)
    text = re.sub(r"[:–—\-_/.,;!()?]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_branch_wrapper(value: str) -> str:
    return _RETURN_WRAPPER.sub("", str(value or "")).strip()


def _parse_number_words(tokens: list[str]) -> float | None:
    """Parse the small cardinal-number vocabulary used by the canonicalizer."""

    if not tokens or any(token not in _NUMBER_WORDS for token in tokens):
        return None
    total = 0
    current = 0
    for token in tokens:
        value = _NUMBER_WORDS[token]
        if value == 100:
            current = max(current, 1) * 100
        else:
            current += value
    total += current
    return float(total)


def classify_answer(value: str) -> CanonicalAnswer:
    """Apply the hard answer-type gate before any embedding-based alias match.

    Abstention phrases count only in the report's first 200 characters.  Long
    reports (more than six normalized whitespace-delimited words) remain whole
    and go directly to ``entity``; short answers retain the precedence
    abstain > bool > date > number > entity.  Answers containing multiple
    numeric mentions deliberately fall back to ``entity`` instead of guessing
    which mention is the conclusion.
    """

    raw = strip_branch_wrapper(value)
    normalized = norm_text(raw)
    if _ABSTAIN.search(raw[:200]):
        return CanonicalAnswer("abstain", "ABSTAIN", normalized)

    normalized_words = normalized.split()
    if not normalized_words:
        return CanonicalAnswer("entity", "", normalized)
    if len(normalized_words) > 6:
        entity_tokens = [token for token in normalized_words if token not in _IGNORED_ENTITY_WORDS]
        return CanonicalAnswer("entity", " ".join(entity_tokens), normalized)

    head = normalized_words[0]
    if head in {"yes", "true"} or normalized.startswith(("是的", "正确")):
        return CanonicalAnswer("bool", "YES", normalized)
    if head in {"no", "false"} or normalized.startswith(("不是", "否", "错误")):
        return CanonicalAnswer("bool", "NO", normalized)

    date_match = _DATEISH.search(raw)
    if date_match:
        try:
            parsed = date_parser.parse(
                date_match.group(0),
                fuzzy=False,
                default=date_parser.parse("2000-01-01"),
            )
            return CanonicalAnswer("date", parsed.strftime("%Y-%m-%d"), normalized)
        except (ValueError, OverflowError):
            pass

    numeric_mentions = _NUMBER.findall(normalized)
    year_match = _YEAR.search(normalized)
    if (
        year_match
        and len(numeric_mentions) == 1
        and numeric_mentions[0].replace(",", "") == year_match.group(1)
    ):
        return CanonicalAnswer("date", f"{year_match.group(1)}-XX-XX", normalized)

    if len(numeric_mentions) == 1:
        number = float(numeric_mentions[0].replace(",", ""))
        return CanonicalAnswer("number", f"{number:.6g}", normalized)
    if not numeric_mentions and len(normalized_words) <= 3:
        number_tokens = [token for token in normalized_words if token in _NUMBER_WORDS]
        word_number = _parse_number_words(number_tokens)
        if word_number is not None:
            return CanonicalAnswer("number", f"{word_number:.6g}", normalized)

    entity_tokens = [token for token in normalized.split() if token not in _IGNORED_ENTITY_WORDS]
    return CanonicalAnswer("entity", " ".join(entity_tokens), normalized)


def _entity_em(left: str, right: str) -> bool:
    compact = lambda text: re.sub(r"\s+", "", text)
    if compact(left) == compact(right):
        return True
    left_tokens, right_tokens = left.split(), right.split()
    if left_tokens and right_tokens and Counter(left_tokens) == Counter(right_tokens):
        return True
    if len(left_tokens) >= 2 and len(right_tokens) >= 2 and left_tokens[-1] == right_tokens[-1]:
        first_left, first_right = left_tokens[0], right_tokens[0]
        if first_left == first_right or (
            min(len(first_left), len(first_right)) >= 4
            and (first_left.startswith(first_right) or first_right.startswith(first_left))
        ):
            return True
    return difflib.SequenceMatcher(None, compact(left), compact(right)).ratio() >= 0.95


def answers_compatible(
    left: CanonicalAnswer,
    right: CanonicalAnswer,
    *,
    entity_similarity: float | None = None,
    entity_threshold: float = 0.90,
) -> bool:
    if left.answer_type != right.answer_type:
        return False
    answer_type = left.answer_type
    if answer_type in {"abstain", "bool", "number"}:
        return left.value == right.value
    if answer_type == "date":
        if left.value == right.value:
            return True
        if left.value[:4] != right.value[:4]:
            return False
        return left.value.endswith("XX-XX") or right.value.endswith("XX-XX")
    if left.value == right.value or _entity_em(left.value, right.value):
        return True
    return entity_similarity is not None and entity_similarity >= entity_threshold


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"{prefix}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0 or not math.isfinite(denominator):
        return -1.0
    return float(np.dot(left, right) / denominator)


def _normalize_vectors(vectors: np.ndarray, expected_rows: int) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != expected_rows or vectors.shape[1] == 0:
        raise ValueError(
            f"Embedding response has shape {vectors.shape}; expected ({expected_rows}, dimension)"
        )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(~np.isfinite(vectors)) or np.any(norms <= 0):
        raise ValueError("Embedding response contains non-finite or zero-length vectors")
    return vectors / norms


def request_embeddings(
    texts: list[str],
    *,
    url: str,
    timeout_s: float,
    batch_size: int,
) -> np.ndarray:
    """Fetch one frozen bulk embedding map without inheriting proxy settings."""

    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    endpoint = url.rstrip("/")
    if not endpoint.endswith("/embed"):
        endpoint += "/embed"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    chunks = []
    for start in range(0, len(texts), 1024):
        chunk = texts[start : start + 1024]
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(
                {"texts": chunk, "batch_size": int(batch_size)}, ensure_ascii=False
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with opener.open(request, timeout=float(timeout_s)) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as error:
            raise RuntimeError(f"SCAD node embedding request failed at {endpoint}: {error}") from error
        if "vectors" not in payload:
            raise RuntimeError(f"SCAD /embed response has no 'vectors' field: {payload.keys()}")
        chunks.append(_normalize_vectors(np.asarray(payload["vectors"], dtype=np.float32), len(chunk)))
    return np.concatenate(chunks, axis=0)


def _coerce_trace(raw_trace: Any) -> list[_Step]:
    if raw_trace is None:
        return []
    if isinstance(raw_trace, np.ndarray):
        raw_trace = raw_trace.tolist()
    if isinstance(raw_trace, str):
        try:
            raw_trace = json.loads(raw_trace)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw_trace, (list, tuple)):
        return []
    steps: list[_Step] = []
    for position, entry in enumerate(raw_trace):
        if not isinstance(entry, Mapping):
            continue
        try:
            index = int(entry.get("index", position))
        except (TypeError, ValueError):
            index = position
        steps.append(
            _Step(
                index=index,
                subtask=str(entry.get("subtask", entry.get("subtask_text", "")) or ""),
                answer=str(
                    entry.get(
                        "answer",
                        entry.get("local_answer", entry.get("subtask_return", "")),
                    )
                    or ""
                ),
            )
        )
    return sorted(steps, key=lambda step: step.index)


def _coerce_planner_spans(raw_spans: Any) -> list[dict[str, Any]]:
    if raw_spans is None:
        return []
    if isinstance(raw_spans, np.ndarray):
        raw_spans = raw_spans.tolist()
    if not isinstance(raw_spans, (list, tuple)):
        return []
    spans = []
    for item in raw_spans:
        if not isinstance(item, Mapping):
            continue
        try:
            start = int(item.get("start", -1))
            end = int(item.get("end", -1))
            index = int(item.get("index", -1))
        except (TypeError, ValueError):
            continue
        if start < 0 or end <= start:
            continue
        spans.append(
            {
                "kind": str(item.get("kind", "pair")),
                "index": index,
                "start": start,
                "end": end,
            }
        )
    return spans


def _extract_paths(non_tensor_batch: Mapping[str, np.ndarray]) -> tuple[dict[tuple[str, str], list[_Step]], dict]:
    roles = np.asarray(non_tensor_batch["scad_role"], dtype=object)
    uids = np.asarray(non_tensor_batch["uid"], dtype=object)
    gen_uids = np.asarray(non_tensor_batch["gen_uid"], dtype=object)
    traces = non_tensor_batch.get("subtask_trace")
    subtask_indices = non_tensor_batch.get("subtask_index")
    subtasks = non_tensor_batch.get("subtask_text")
    local_answers = non_tensor_batch.get("local_answer")
    if local_answers is None:
        local_answers = non_tensor_batch.get("subtask_return")

    paths: dict[tuple[str, str], list[_Step]] = {}
    group_sample_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for sample_index, (uid, gen_uid) in enumerate(zip(uids, gen_uids)):
        key = str(uid), str(gen_uid)
        group_sample_indices[key].append(sample_index)
        if traces is not None:
            trace = _coerce_trace(traces[sample_index])
            if trace and (key not in paths or str(roles[sample_index]) == "planner"):
                paths[key] = trace

    fallback: dict[tuple[str, str], list[_Step]] = defaultdict(list)
    for sample_index, role in enumerate(roles):
        if str(role) != "worker":
            continue
        key = str(uids[sample_index]), str(gen_uids[sample_index])
        try:
            subtask_index = int(subtask_indices[sample_index]) if subtask_indices is not None else 0
        except (TypeError, ValueError):
            subtask_index = 0
        fallback[key].append(
            _Step(
                index=subtask_index,
                subtask=str(subtasks[sample_index] if subtasks is not None else ""),
                answer=str(local_answers[sample_index] if local_answers is not None else ""),
            )
        )
    for key, sample_indices in group_sample_indices.items():
        if key not in paths:
            paths[key] = sorted(fallback.get(key, []), key=lambda step: step.index)
    return paths, group_sample_indices


def _cluster_subtasks(
    normalized_subtasks: Sequence[str],
    embeddings: Mapping[str, np.ndarray],
    config: NodeMergeConfig,
) -> dict[str, str]:
    leaders: list[str] = []
    result: dict[str, str] = {}
    for subtask in sorted(set(normalized_subtasks)):
        selected = None
        if config.semantic:
            for leader in leaders:
                if not subtask or not leader:
                    continue
                if _cosine(embeddings[subtask], embeddings[leader]) >= config.subtask_similarity_threshold:
                    selected = leader
                    break
        elif subtask in leaders:
            selected = subtask
        if selected is None:
            selected = subtask
            leaders.append(subtask)
        result[subtask] = _stable_id("subtask", selected)
    return result


def _cluster_answers(
    raw_steps: Sequence[tuple[_Step, str, CanonicalAnswer]],
    embeddings: Mapping[str, np.ndarray],
    config: NodeMergeConfig,
) -> dict[tuple[str, str, str], str]:
    by_bucket: dict[tuple[str, str], dict[str, CanonicalAnswer]] = defaultdict(dict)
    for _, subtask_cluster, answer in raw_steps:
        by_bucket[(subtask_cluster, answer.answer_type)][answer.value] = answer

    result: dict[tuple[str, str, str], str] = {}
    for (subtask_cluster, answer_type), candidates in sorted(by_bucket.items()):
        leaders: list[str] = []
        for value in sorted(candidates):
            answer = candidates[value]
            selected = None
            for leader_value in leaders:
                similarity = None
                if (
                    answer_type == "entity"
                    and config.semantic
                    and value in embeddings
                    and leader_value in embeddings
                ):
                    similarity = _cosine(embeddings[value], embeddings[leader_value])
                if answers_compatible(
                    answer,
                    candidates[leader_value],
                    entity_similarity=similarity,
                    entity_threshold=config.entity_similarity_threshold,
                ):
                    selected = leader_value
                    break
            if selected is None:
                selected = value
                leaders.append(value)
            result[(subtask_cluster, answer_type, value)] = selected
    return result


def build_scad_node_metadata(
    non_tensor_batch: Mapping[str, np.ndarray],
    config: NodeMergeConfig,
    *,
    embed_fn: EmbeddingFunction | None = None,
) -> NodeMergeResult:
    """Build and freeze one semantic trie for every question-level UID."""

    required = {"uid", "gen_uid", "scad_role"}
    missing = sorted(required - set(non_tensor_batch))
    if missing:
        raise RuntimeError(f"SCAD node merge is missing rollout metadata: {missing}")

    roles = np.asarray(non_tensor_batch["scad_role"], dtype=object)
    uids = np.asarray(non_tensor_batch["uid"], dtype=object)
    gen_uids = np.asarray(non_tensor_batch["gen_uid"], dtype=object)
    batch_size = len(roles)
    if not (len(uids) == len(gen_uids) == batch_size):
        raise ValueError("SCAD node metadata arrays must have equal lengths")

    paths, group_sample_indices = _extract_paths(non_tensor_batch)
    paths_by_uid: dict[str, dict[str, list[_Step]]] = defaultdict(dict)
    for (uid, gen_uid), steps in paths.items():
        paths_by_uid[uid][gen_uid] = steps

    # Deduplicate across the whole PPO batch and use one bulk server request in
    # the common case.  Clustering remains isolated per question UID below.
    embeddings: dict[str, np.ndarray] = {}
    if config.semantic:
        all_steps = [
            step
            for uid_paths in paths_by_uid.values()
            for steps in uid_paths.values()
            for step in steps
        ]
        embedding_texts = sorted(
            {normalized for step in all_steps if (normalized := norm_text(step.subtask))}
            | {
                answer.value
                for step in all_steps
                if (answer := classify_answer(step.answer)).answer_type == "entity" and answer.value
            }
        )
        if embedding_texts:
            if embed_fn is None:
                vectors = request_embeddings(
                    embedding_texts,
                    url=config.embedding_url,
                    timeout_s=config.embedding_timeout_s,
                    batch_size=config.embedding_batch_size,
                )
            else:
                vectors = _normalize_vectors(embed_fn(embedding_texts), len(embedding_texts))
            embeddings = dict(zip(embedding_texts, vectors))

    node_ids = np.full(batch_size, "", dtype=object)
    parent_ids = np.full(batch_size, "", dtype=object)
    node_paths = np.empty(batch_size, dtype=object)
    node_paths[:] = [tuple() for _ in range(batch_size)]
    node_depths = np.full(batch_size, -1, dtype=np.int32)
    node_kinds = np.full(batch_size, "", dtype=object)
    subtask_clusters = np.full(batch_size, "", dtype=object)
    answer_types = np.full(batch_size, "", dtype=object)
    answer_values = np.full(batch_size, "", dtype=object)
    path_nodes = np.empty(batch_size, dtype=object)
    path_nodes[:] = [tuple() for _ in range(batch_size)]
    planner_node_spans = np.empty(batch_size, dtype=object)
    planner_node_spans[:] = [tuple() for _ in range(batch_size)]

    audit_questions: list[dict[str, Any]] = []
    all_pair_occurrences: list[tuple[str, str]] = []
    all_stop_occurrences: list[tuple[str, str]] = []
    total_subtask_clusters = 0

    for uid in sorted(paths_by_uid):
        uid_paths = paths_by_uid[uid]
        flattened = [step for steps in uid_paths.values() for step in steps]
        normalized_subtasks = [norm_text(step.subtask) for step in flattened]
        classified_answers = [classify_answer(step.answer) for step in flattened]

        subtask_cluster_map = _cluster_subtasks(normalized_subtasks, embeddings, config)
        total_subtask_clusters += len(set(subtask_cluster_map.values()))

        prepared: list[tuple[_Step, str, CanonicalAnswer]] = []
        cursor = 0
        for gen_uid in uid_paths:
            for step in uid_paths[gen_uid]:
                prepared.append(
                    (
                        step,
                        subtask_cluster_map[normalized_subtasks[cursor]],
                        classified_answers[cursor],
                    )
                )
                cursor += 1
        answer_cluster_map = _cluster_answers(prepared, embeddings, config)

        prepared_by_gen: dict[str, list[_CanonicalStep]] = defaultdict(list)
        for step, subtask_cluster, answer in prepared:
            answer_cluster = answer_cluster_map[(subtask_cluster, answer.answer_type, answer.value)]
            # Locate the rollout deterministically.  Step objects are immutable;
            # identity is safe here because they originate from uid_paths.
            owner = next(
                gen_uid
                for gen_uid, candidate_steps in uid_paths.items()
                if any(candidate is step for candidate in candidate_steps)
            )
            prepared_by_gen[owner].append(
                _CanonicalStep(
                    index=step.index,
                    subtask=step.subtask,
                    answer=step.answer,
                    normalized_subtask=norm_text(step.subtask),
                    subtask_cluster=subtask_cluster,
                    answer_type=answer.answer_type,
                    answer_value=answer.value,
                    answer_cluster=answer_cluster,
                )
            )

        root_id = _stable_id("root", uid)
        path_metadata: dict[str, dict[str, Any]] = {}
        question_nodes: dict[str, dict[str, Any]] = {}
        for gen_uid in sorted(uid_paths):
            parent_id = root_id
            pair_path: list[str] = []
            path_node_records: list[dict[str, Any]] = []
            step_metadata: dict[int, dict[str, Any]] = {}
            for depth, step in enumerate(
                sorted(prepared_by_gen.get(gen_uid, []), key=lambda item: item.index), start=1
            ):
                key = step.semantic_key
                node_id = _stable_id("node", uid, parent_id, *key)
                all_pair_occurrences.append((node_id, gen_uid))
                question_nodes.setdefault(
                    node_id,
                    {
                        "id": node_id,
                        "parent_id": parent_id,
                        "depth": depth,
                        "kind": "pair",
                        "subtask_cluster": step.subtask_cluster,
                        "normalized_subtask": step.normalized_subtask,
                        "answer_type": step.answer_type,
                        "answer_value": step.answer_value,
                        "answer_cluster": step.answer_cluster,
                    },
                )
                pair_path.append(node_id)
                step_metadata[step.index] = {
                    "node_id": node_id,
                    "parent_id": parent_id,
                    "depth": depth,
                    "path": tuple(pair_path),
                    "step": step,
                }
                path_node_records.append(
                    {
                        "node_id": node_id,
                        "parent_node_id": parent_id,
                        "kind": "pair",
                        "depth": depth,
                        "subtask_cluster_id": step.subtask_cluster,
                        "answer_type": step.answer_type,
                        "canonical_answer": step.answer_value,
                        "answer_cluster": step.answer_cluster,
                    }
                )
                parent_id = node_id

            stop_id = _stable_id("stop", uid, parent_id)
            all_stop_occurrences.append((stop_id, gen_uid))
            question_nodes.setdefault(
                stop_id,
                {
                    "id": stop_id,
                    "parent_id": parent_id,
                    "depth": len(pair_path) + 1,
                    "kind": "stop",
                },
            )
            path_node_records.append(
                {
                    "node_id": stop_id,
                    "parent_node_id": parent_id,
                    "kind": "stop",
                    "depth": len(pair_path) + 1,
                    "subtask_cluster_id": "__STOP__",
                    "answer_type": "stop",
                    "canonical_answer": "STOP",
                }
            )
            path_metadata[gen_uid] = {
                "steps": step_metadata,
                "stop_id": stop_id,
                "stop_parent": parent_id,
                "stop_depth": len(pair_path) + 1,
                "full_path": tuple(pair_path + [stop_id]),
                "path_nodes": tuple(path_node_records),
            }

        for (sample_uid, gen_uid), sample_indices in group_sample_indices.items():
            if sample_uid != uid:
                continue
            path = path_metadata[gen_uid]
            for sample_index in sample_indices:
                role = str(roles[sample_index])
                if role == "planner":
                    node_ids[sample_index] = path["stop_id"]
                    parent_ids[sample_index] = path["stop_parent"]
                    node_paths[sample_index] = path["full_path"]
                    node_depths[sample_index] = path["stop_depth"]
                    node_kinds[sample_index] = "stop"
                    path_nodes[sample_index] = path["path_nodes"]
                    mapped_spans = []
                    raw_spans = non_tensor_batch.get("planner_decision_spans")
                    for span in _coerce_planner_spans(
                        raw_spans[sample_index] if raw_spans is not None else None
                    ):
                        if span["kind"] == "stop":
                            node_record = path["path_nodes"][-1]
                        else:
                            step_item = path["steps"].get(span["index"])
                            if step_item is None:
                                continue
                            step = step_item["step"]
                            node_record = {
                                "node_id": step_item["node_id"],
                                "parent_node_id": step_item["parent_id"],
                                "kind": "pair",
                                "depth": step_item["depth"],
                                "subtask_cluster_id": step.subtask_cluster,
                                "answer_type": step.answer_type,
                                "canonical_answer": step.answer_value,
                                "answer_cluster": step.answer_cluster,
                            }
                        mapped_spans.append({**node_record, "start": span["start"], "end": span["end"]})
                    planner_node_spans[sample_index] = tuple(mapped_spans)
                elif role == "worker":
                    try:
                        subtask_index = int(non_tensor_batch["subtask_index"][sample_index])
                    except (KeyError, TypeError, ValueError):
                        subtask_index = 0
                    if subtask_index not in path["steps"]:
                        raise RuntimeError(
                            f"Worker sample {sample_index} has no trie step for "
                            f"uid={uid}, gen_uid={gen_uid}, subtask_index={subtask_index}"
                        )
                    item = path["steps"][subtask_index]
                    step = item["step"]
                    node_ids[sample_index] = item["node_id"]
                    parent_ids[sample_index] = item["parent_id"]
                    node_paths[sample_index] = item["path"]
                    node_depths[sample_index] = item["depth"]
                    node_kinds[sample_index] = "pair"
                    subtask_clusters[sample_index] = step.subtask_cluster
                    answer_types[sample_index] = step.answer_type
                    answer_values[sample_index] = step.answer_value

        audit_questions.append(
            {
                "uid": uid,
                "root_id": root_id,
                "nodes": sorted(question_nodes.values(), key=lambda node: (node["depth"], node["id"])),
                "paths": [
                    {
                        "gen_uid": gen_uid,
                        "node_path": list(path_metadata[gen_uid]["full_path"]),
                    }
                    for gen_uid in sorted(path_metadata)
                ],
            }
        )

    worker_sample_indices = [index for index, role in enumerate(roles) if str(role) == "worker"]
    missing_workers = [index for index in worker_sample_indices if not node_ids[index]]
    missing_planners = [
        index for index, role in enumerate(roles) if str(role) == "planner" and not node_ids[index]
    ]
    if missing_workers or missing_planners:
        raise RuntimeError(
            f"SCAD trie left samples unmapped: workers={missing_workers}, planners={missing_planners}"
        )

    pair_counts = Counter(node_id for node_id, _ in all_pair_occurrences)
    pair_rollouts: dict[str, set[str]] = defaultdict(set)
    for node_id, gen_uid in all_pair_occurrences:
        pair_rollouts[node_id].add(gen_uid)
    stop_counts = Counter(node_id for node_id, _ in all_stop_occurrences)
    pair_occurrence_count = len(all_pair_occurrences)
    stop_occurrence_count = len(all_stop_occurrences)
    cross_rollout_occurrences = sum(
        count for node_id, count in pair_counts.items() if len(pair_rollouts[node_id]) > 1
    )
    metrics = {
        "scad/node_pair_occurrences": float(pair_occurrence_count),
        "scad/node_unique_pairs": float(len(pair_counts)),
        "scad/node_merge_rate": (
            (pair_occurrence_count - len(pair_counts)) / max(pair_occurrence_count, 1)
        ),
        "scad/node_cross_rollout_fraction": (
            cross_rollout_occurrences / max(pair_occurrence_count, 1)
        ),
        "scad/node_subtask_clusters": float(total_subtask_clusters),
        "scad/node_stop_merge_rate": (
            (stop_occurrence_count - len(stop_counts)) / max(stop_occurrence_count, 1)
        ),
    }
    metadata = {
        "scad_node_id": node_ids,
        "scad_parent_node_id": parent_ids,
        "scad_node_path": node_paths,
        "scad_node_depth": node_depths,
        "scad_node_kind": node_kinds,
        "scad_subtask_cluster_id": subtask_clusters,
        "scad_answer_type": answer_types,
        "scad_canonical_answer": answer_values,
        "scad_path_nodes": path_nodes,
        "scad_planner_node_spans": planner_node_spans,
    }
    return NodeMergeResult(
        metadata=metadata,
        metrics=metrics,
        audit={"version": 1, "questions": audit_questions, "metrics": metrics},
    )


def annotate_scad_nodes(
    data: Any,
    config_value: Mapping[str, Any] | None,
    *,
    global_step: int | None = None,
    embed_fn: EmbeddingFunction | None = None,
) -> Any:
    """Attach frozen trie metadata to a ``DataProto`` in-place and return it."""

    config = NodeMergeConfig.from_mapping(config_value)
    result = build_scad_node_metadata(data.non_tensor_batch, config, embed_fn=embed_fn)
    data.non_tensor_batch.update(result.metadata)
    data.meta_info["scad_node_metrics"] = result.metrics
    if config.audit_dir:
        audit_dir = Path(config.audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        step_name = "unknown" if global_step is None else f"{int(global_step):06d}"
        target = audit_dir / f"step_{step_name}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(result.audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(target)
    return data
