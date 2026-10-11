#!/usr/bin/env python3
"""GPU-sharded local retrieval server for the unified SCAD corpus."""

from __future__ import annotations

import argparse
import asyncio
import bisect
import functools
import heapq
import json
import logging
import multiprocessing as mp
import os
import queue
import signal
import sqlite3
import threading
import time
import uuid
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

from envs.ann_retrieval import AnnRetriever


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SourceName = Literal["browsecomp_plus", "webwalker", "wiki18"]
VALID_SOURCES = ("browsecomp_plus", "webwalker", "wiki18")
QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


@dataclass(frozen=True)
class ShardSpec:
    source: str
    name: str
    embedding_path: str
    passage_path: str
    rows: int
    source_start: int


class QueryRequest(BaseModel):
    query: str
    k: int = 20
    source: SourceName


class OpenRequest(BaseModel):
    docid: Optional[str] = None
    url: Optional[str] = None
    source: Optional[SourceName] = None


class QueryResponse(BaseModel):
    results: list[dict[str, Any]]
    took_ms: float


class EmbedRequest(BaseModel):
    texts: list[str]
    dim: Optional[int] = None
    batch_size: int = 64


class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    took_ms: float


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if int(attention_mask[:, -1].sum().item()) == attention_mask.shape[0]:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_states[
        torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device),
        sequence_lengths,
    ]


def load_shard_specs(
    embedding_root: Path,
    passage_root: Path,
    sources: tuple[str, ...],
) -> list[ShardSpec]:
    manifest_path = embedding_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dimension = int(manifest.get("dimension", 0))
    if dimension <= 0:
        raise ValueError(f"Invalid embedding dimension in {manifest_path}")

    by_source: dict[str, list[dict]] = defaultdict(list)
    for shard in manifest.get("shards", []):
        if shard["source"] in sources:
            by_source[shard["source"]].append(shard)

    specs = []
    for source in sources:
        shards = sorted(by_source[source], key=lambda shard: shard["output_relative_path"])
        if not shards:
            raise ValueError(f"No embedding shards found for source={source}")
        source_start = 0
        for shard in shards:
            embedding_path = embedding_root / shard["output_relative_path"]
            passage_path = passage_root / shard["input_relative_path"]
            if not embedding_path.is_file() or not passage_path.is_file():
                raise FileNotFoundError(f"Missing paired shard: {embedding_path} / {passage_path}")
            array = np.load(embedding_path, mmap_mode="r")
            expected_shape = (int(shard["rows"]), dimension)
            if array.shape != expected_shape or array.dtype != np.float16:
                raise ValueError(
                    f"Unexpected embedding array {embedding_path}: {array.shape} {array.dtype}; "
                    f"expected {expected_shape} float16"
                )
            parquet_rows = pq.ParquetFile(passage_path).metadata.num_rows
            if parquet_rows != expected_shape[0]:
                raise ValueError(f"Embedding/passage row mismatch for {embedding_path}")
            specs.append(
                ShardSpec(
                    source=source,
                    name=Path(shard["output_relative_path"]).stem,
                    embedding_path=str(embedding_path),
                    passage_path=str(passage_path),
                    rows=expected_shape[0],
                    source_start=source_start,
                )
            )
            source_start += expected_shape[0]
    return specs


def assign_shards(specs: list[ShardSpec], num_workers: int) -> list[list[ShardSpec]]:
    assignments = [[] for _ in range(num_workers)]
    by_source: dict[str, list[ShardSpec]] = defaultdict(list)
    for spec in specs:
        by_source[spec.source].append(spec)
    for source_specs in by_source.values():
        for index, spec in enumerate(source_specs):
            assignments[index % num_workers].append(spec)
    return assignments


def _push_topk(heap: list[tuple], hit: tuple, k: int) -> None:
    if len(heap) < k:
        heapq.heappush(heap, hit)
    elif hit > heap[0]:
        heapq.heapreplace(heap, hit)


def retrieval_worker(
    worker_id: int,
    device_index: int,
    specs: list[ShardSpec],
    request_queue,
    result_queue,
    worker_top_k: int,
    batch_size: int,
    batch_timeout_s: float,
) -> None:
    try:
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
        loaded = []
        bytes_loaded = 0
        for spec in specs:
            array = np.load(spec.embedding_path, mmap_mode="r")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
                tensor = torch.from_numpy(array).to(device=device, dtype=torch.float16).contiguous()
            loaded.append((spec, tensor))
            bytes_loaded += tensor.numel() * tensor.element_size()
        by_source: dict[str, list[tuple[ShardSpec, torch.Tensor]]] = defaultdict(list)
        for spec, tensor in loaded:
            by_source[spec.source].append((spec, tensor))
        result_queue.put(
            {
                "type": "ready",
                "worker_id": worker_id,
                "device": device_index,
                "shards": len(specs),
                "bytes": bytes_loaded,
            }
        )

        while True:
            first = request_queue.get()
            if first is None:
                break
            jobs = [first]
            deadline = time.monotonic() + batch_timeout_s
            while len(jobs) < batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    job = request_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if job is None:
                    request_queue.put(None)
                    break
                jobs.append(job)

            grouped: dict[str, list[dict]] = defaultdict(list)
            for job in jobs:
                grouped[job["source"]].append(job)

            with torch.inference_mode():
                for source, source_jobs in grouped.items():
                    heaps = [[] for _ in source_jobs]
                    source_shards = by_source.get(source, [])
                    if source_shards:
                        queries = np.stack([job["vector"] for job in source_jobs])
                        query_tensor = torch.from_numpy(queries).to(device=device, dtype=torch.float16)
                        for spec, corpus in source_shards:
                            similarities = torch.mm(corpus, query_tensor.T)
                            local_k = min(worker_top_k, corpus.shape[0])
                            values, indices = torch.topk(similarities, k=local_k, dim=0)
                            values_np = values.float().cpu().numpy()
                            indices_np = indices.cpu().numpy()
                            for column, heap in enumerate(heaps):
                                for rank in range(local_k):
                                    _push_topk(
                                        heap,
                                        (
                                            float(values_np[rank, column]),
                                            spec.name,
                                            int(indices_np[rank, column]),
                                        ),
                                        worker_top_k,
                                    )
                            del similarities, values, indices
                    for job, heap in zip(source_jobs, heaps):
                        hits = [
                            {"score": score, "source": source, "shard": shard, "row": row}
                            for score, shard, row in sorted(heap, reverse=True)
                        ]
                        result_queue.put(
                            {
                                "type": "result",
                                "request_id": job["request_id"],
                                "worker_id": worker_id,
                                "hits": hits,
                            }
                        )
    except Exception as error:
        result_queue.put(
            {
                "type": "worker_error",
                "worker_id": worker_id,
                "error": f"{type(error).__name__}: {error}",
            }
        )


def bounded_embedding_batches(lengths: list[int], max_items: int, max_tokens: int):
    """Keep input order while bounding the padded token count of each forward."""
    if max_items < 1 or max_tokens < 0:
        raise ValueError("Embedding batch limits must be positive (zero tokens means unlimited)")
    start = 0
    longest = 0
    for index, length in enumerate(lengths):
        if max_tokens and length > max_tokens:
            raise ValueError("Embedding token budget must fit a single truncated input")
        candidate_longest = max(longest, length)
        size = index - start + 1
        if index > start and (size > max_items or (max_tokens and size * candidate_longest > max_tokens)):
            yield start, index
            start = index
            longest = length
        else:
            longest = candidate_longest
    if start < len(lengths):
        yield start, len(lengths)


class QueryEncoder:
    def __init__(self, model_path: Path, device_index: int, max_length: int = 8192):
        self.device = torch.device(f"cuda:{device_index}")
        self.max_length = max_length
        # The shared GPU also hosts the training/rollout processes. Bound both
        # short-query batches and long post-rollout report batches. Defaults
        # preserve legacy launches; the 1.7B profile opts into bounded forwards.
        self.max_batch_size = int(os.getenv("SCAD_ENCODER_MAX_BATCH_SIZE", "256"))
        self.max_batch_tokens = int(os.getenv("SCAD_ENCODER_MAX_BATCH_TOKENS", "0"))
        self.release_cuda_cache = os.getenv("SCAD_ENCODER_RELEASE_CACHE", "0").lower() in {"1", "true"}
        if self.max_batch_size < 1 or self.max_batch_tokens < 0:
            raise ValueError("Invalid SCAD_ENCODER batch limits")
        if self.max_batch_tokens and self.max_batch_tokens < self.max_length:
            raise ValueError("SCAD_ENCODER_MAX_BATCH_TOKENS must be >= encoder max_length")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs = {
            "local_files_only": True,
            "dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
        try:
            self.model = AutoModel.from_pretrained(
                model_path,
                attn_implementation="flash_attention_2",
                **model_kwargs,
            )
        except (ImportError, ValueError):
            logger.warning("FlashAttention unavailable for query encoder; using SDPA")
            self.model = AutoModel.from_pretrained(
                model_path,
                attn_implementation="sdpa",
                **model_kwargs,
            )
        self.model.config.use_cache = False
        self.model.eval().to(self.device)
        # The query batcher and post-rollout semantic canonicalizer share this
        # model.  Serialize forwards at the native CUDA boundary: post-hoc bulk
        # embedding runs while retrieval is idle, and concurrent forwards on a
        # single model have previously made the server unstable.
        self._encode_lock = threading.RLock()

    def _encode_token_batch(self, tokenized: dict) -> np.ndarray:
        batch = self.tokenizer.pad(tokenized, padding=True, return_tensors="pt")
        batch = {name: value.to(self.device) for name, value in batch.items()}
        with torch.inference_mode():
            outputs = self.model(**batch, use_cache=False)
            pooled = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            normalized = F.normalize(pooled.float(), p=2, dim=1)
        return normalized.to(dtype=torch.float16).cpu().numpy()

    def encode_many(self, queries: list[str]) -> np.ndarray:
        if not queries:
            raise ValueError("queries must be non-empty")
        with self._encode_lock:
            instructed = [f"Instruct: {QUERY_INSTRUCTION}\nQuery:{query}" for query in queries]
            # Preserve the original batch's left-padding/position layout when
            # splitting it. Re-padding each slice to a shorter length changes
            # BF16 RoPE rounding and can flip near-threshold node similarities.
            tokenized = self.tokenizer(
                instructed, padding=True, truncation=True, max_length=self.max_length,
            )
            lengths = [len(ids) for ids in tokenized["input_ids"]]
            try:
                vectors = [
                    self._encode_token_batch({name: values[start:end] for name, values in tokenized.items()})
                    for start, end in bounded_embedding_batches(
                        lengths, self.max_batch_size, self.max_batch_tokens,
                    )
                ]
                return np.concatenate(vectors, axis=0)
            finally:
                # The helper frame has returned, so its CUDA temporaries are
                # gone. Return unused allocator blocks to other GPU processes;
                # keep model weights and the CPU numpy results intact.
                if self.release_cuda_cache and self.device.type == "cuda":
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()

    def encode(self, query: str) -> np.ndarray:
        return self.encode_many([query])[0]


def bulk_encode(
    encoder: QueryEncoder,
    texts: list[str],
    *,
    batch_size: int = 64,
    dim: int | None = None,
) -> np.ndarray:
    """Encode a bounded post-rollout burst using the resident search model."""

    if not texts or len(texts) > 1024:
        raise ValueError("texts must contain 1..1024 items")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("every embedding text must be a non-empty string")
    batch_size = int(batch_size)
    if batch_size < 1 or batch_size > 256:
        raise ValueError("batch_size must be between 1 and 256")
    chunks = [
        encoder.encode_many(texts[start : start + batch_size])
        for start in range(0, len(texts), batch_size)
    ]
    vectors = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if dim is not None:
        dim = int(dim)
        if dim < 1 or dim > vectors.shape[1]:
            raise ValueError(f"dim must be between 1 and {vectors.shape[1]}")
        vectors = vectors[:, :dim]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, np.finfo(np.float32).tiny)
    return vectors


@dataclass
class EncodeJob:
    query: str
    event: threading.Event
    vector: np.ndarray | None = None
    error: Exception | None = None


class BatchedQueryEncoder:
    def __init__(
        self,
        encoder: QueryEncoder,
        batch_size: int = 16,
        batch_timeout_s: float = 0.01,
    ):
        self.encoder = encoder
        self.batch_size = batch_size
        self.batch_timeout_s = batch_timeout_s
        self.queue: queue.Queue[EncodeJob | None] = queue.Queue(maxsize=2000)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            first = self.queue.get()
            if first is None:
                return
            jobs = [first]
            deadline = time.monotonic() + self.batch_timeout_s
            while len(jobs) < self.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    job = self.queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if job is None:
                    self.queue.put(None)
                    break
                jobs.append(job)
            try:
                vectors = self.encoder.encode_many([job.query for job in jobs])
                for job, vector in zip(jobs, vectors):
                    job.vector = vector
            except Exception as error:
                for job in jobs:
                    job.error = error
            finally:
                for job in jobs:
                    job.event.set()

    def encode(self, query: str, timeout_s: float) -> np.ndarray:
        job = EncodeJob(query=query, event=threading.Event())
        self.queue.put(job, timeout=10)
        if not job.event.wait(timeout_s):
            raise TimeoutError(f"Query encoding timed out after {timeout_s}s")
        if job.error is not None:
            raise job.error
        if job.vector is None:
            raise RuntimeError("Query encoder returned no vector")
        return job.vector

    def shutdown(self) -> None:
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=10)


class PassageStore:
    def __init__(self, specs: list[ShardSpec], cache_groups: int = 8):
        self.specs_by_key = {(spec.source, spec.name): spec for spec in specs}
        self.files = {key: pq.ParquetFile(spec.passage_path) for key, spec in self.specs_by_key.items()}
        # PyArrow ParquetFile instances are shared by all request workers.
        # Concurrent row-group reads on the same native handle can crash the
        # process, so protect only that I/O/cache boundary. The OS page cache
        # already holds the parquet bytes; a small decoded-object cache avoids
        # retaining tens of GiB of Python dictionaries.
        self.read_lock = threading.RLock()
        self.group_starts = {}
        self.docid_ranges: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        self.source_specs: dict[str, list[ShardSpec]] = defaultdict(list)
        for key, parquet_file in self.files.items():
            starts = []
            total = 0
            for group_index in range(parquet_file.num_row_groups):
                starts.append(total)
                total += parquet_file.metadata.row_group(group_index).num_rows
            self.group_starts[key] = starts
            spec = self.specs_by_key[key]
            first_group = parquet_file.read_row_group(0, columns=["docid"])
            last_group = parquet_file.read_row_group(parquet_file.num_row_groups - 1, columns=["docid"])
            first_docid = int(first_group.column("docid")[0].as_py())
            last_docid = int(last_group.column("docid")[-1].as_py())
            self.docid_ranges[spec.source].append((first_docid, last_docid, spec.name))
            self.source_specs[spec.source].append(spec)
        for source in self.docid_ranges:
            self.docid_ranges[source].sort()
            self.source_specs[source].sort(key=lambda spec: spec.source_start)
        self._read_group = functools.lru_cache(maxsize=cache_groups)(self._read_group_uncached)
        self.wiki_title_index_lock = threading.Lock()
        self.wiki_title_index = self._open_wiki_title_index()

    def _wiki_title_index_path(self) -> Path | None:
        specs = self.source_specs.get("wiki18", [])
        if not specs:
            return None
        parents = {Path(spec.passage_path).parent for spec in specs}
        if len(parents) != 1:
            raise ValueError("Wiki18 passage shards must share one directory")
        return parents.pop() / "title_index.sqlite3"

    @staticmethod
    def _wiki_index_matches(connection: sqlite3.Connection, expected_rows: int, expected_shards: int) -> bool:
        try:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        except sqlite3.DatabaseError:
            return False
        return metadata == {
            "schema_version": "wiki18_title_index_v1",
            "rows": str(expected_rows),
            "shards": str(expected_shards),
        }

    def _build_wiki_title_index(self, path: Path, specs: list[ShardSpec]) -> None:
        temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary_path.unlink(missing_ok=True)
        connection = sqlite3.connect(temporary_path)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=MEMORY")
            connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute(
                "CREATE TABLE title_positions (title TEXT NOT NULL, source_position INTEGER NOT NULL)"
            )
            connection.execute("BEGIN")
            for spec in specs:
                parquet_file = self.files[("wiki18", spec.name)]
                starts = self.group_starts[("wiki18", spec.name)]
                for group_index, group_start in enumerate(starts):
                    with self.read_lock:
                        titles = parquet_file.read_row_group(group_index, columns=["title"]).column("title")
                    connection.executemany(
                        "INSERT INTO title_positions(title, source_position) VALUES (?, ?)",
                        (
                            (str(title), spec.source_start + group_start + offset)
                            for offset, title in enumerate(titles.to_pylist())
                        ),
                    )
            connection.commit()
            connection.execute(
                "CREATE INDEX title_positions_lookup ON title_positions(title, source_position)"
            )
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                [
                    ("schema_version", "wiki18_title_index_v1"),
                    ("rows", str(sum(spec.rows for spec in specs))),
                    ("shards", str(len(specs))),
                ],
            )
            connection.commit()
        finally:
            connection.close()
        temporary_path.replace(path)

    def _open_wiki_title_index(self) -> sqlite3.Connection | None:
        path = self._wiki_title_index_path()
        if path is None:
            return None
        specs = self.source_specs["wiki18"]
        expected_rows = sum(spec.rows for spec in specs)
        if path.is_file():
            connection = sqlite3.connect(path, check_same_thread=False)
            if self._wiki_index_matches(connection, expected_rows, len(specs)):
                return connection
            connection.close()
        logger.info("Building Wiki18 title index at %s", path)
        self._build_wiki_title_index(path, specs)
        connection = sqlite3.connect(path, check_same_thread=False)
        if not self._wiki_index_matches(connection, expected_rows, len(specs)):
            connection.close()
            raise RuntimeError(f"Invalid Wiki18 title index after build: {path}")
        return connection

    def _read_group_uncached(self, source: str, shard: str, group_index: int):
        # Keep Arrow's compact columnar representation in the LRU. Expanding an
        # entire row group with to_pylist() retained hundreds of MiB of Python
        # objects even though a search needs only a few selected rows.
        return self.files[(source, shard)].read_row_group(group_index)

    def get(self, source: str, shard: str, row: int) -> dict:
        key = (source, shard)
        starts = self.group_starts[key]
        group_index = bisect.bisect_right(starts, row) - 1
        if group_index < 0:
            raise IndexError(row)
        local_row = row - starts[group_index]
        with self.read_lock:
            table = self._read_group(source, shard, group_index)
        if local_row >= table.num_rows:
            raise IndexError(row)
        return {
            name: table.column(name)[local_row].as_py()
            for name in table.column_names
        }

    def locate_docid(self, docid: str, source: str | None = None) -> tuple[str, str, int] | None:
        if not str(docid).isdigit():
            return None
        numeric = int(docid)
        sources = [source] if source else list(self.docid_ranges)
        for candidate_source in sources:
            ranges = self.docid_ranges.get(candidate_source, [])
            starts = [item[0] for item in ranges]
            range_index = bisect.bisect_right(starts, numeric) - 1
            if range_index < 0:
                continue
            first, last, shard = ranges[range_index]
            if first <= numeric <= last:
                spec = self.specs_by_key[(candidate_source, shard)]
                row = numeric - first
                if row < spec.rows:
                    return candidate_source, shard, row
        return None

    def global_locator(self, source: str, position: int) -> tuple[str, str, int] | None:
        specs = self.source_specs[source]
        starts = [spec.source_start for spec in specs]
        index = bisect.bisect_right(starts, position) - 1
        if index < 0:
            return None
        spec = specs[index]
        local_row = position - spec.source_start
        if local_row >= spec.rows:
            return None
        return source, spec.name, local_row

    def open_document(self, locator: tuple[str, str, int], max_passages: int = 64) -> dict:
        source, shard, row = locator
        seed = self.get(source, shard, row)
        spec = self.specs_by_key[(source, shard)]
        seed_position = spec.source_start + row
        if source == "wiki18":
            # The public Wiki18 retrieval corpus is already passage-level and
            # assigns every row a distinct parent_docid. Reconstruct a page by
            # joining all passages with the same Wikipedia title, which can be
            # interleaved and can cross parquet shard boundaries.
            if self.wiki_title_index is None:
                raise RuntimeError("Wiki18 title index is unavailable")
            with self.wiki_title_index_lock:
                positions = [
                    int(result[0])
                    for result in self.wiki_title_index.execute(
                        "SELECT source_position FROM title_positions WHERE title = ? ORDER BY source_position",
                        (seed["title"],),
                    )
                ]
            if len(positions) > max_passages:
                positions = sorted(
                    sorted(positions, key=lambda position: (abs(position - seed_position), position))[:max_passages]
                )
        else:
            parent_docid = seed["parent_docid"]
            positions = [seed_position]
            position = seed_position - 1
            while position >= 0 and len(positions) < max_passages:
                previous_locator = self.global_locator(source, position)
                if previous_locator is None or self.get(*previous_locator)["parent_docid"] != parent_docid:
                    break
                positions.insert(0, position)
                position -= 1
            position = seed_position + 1
            while len(positions) < max_passages:
                next_locator = self.global_locator(source, position)
                if next_locator is None or self.get(*next_locator)["parent_docid"] != parent_docid:
                    break
                positions.append(position)
                position += 1
        passages = [self.get(*self.global_locator(source, position)) for position in positions]
        text = "\n\n".join(passage["text"] for passage in passages)
        return {
            "docid": seed["docid"],
            "url": seed["url"],
            "title": seed["title"],
            "text": text,
            "source": source,
            "passages": len(passages),
        }

    def close(self) -> None:
        if self.wiki_title_index is not None:
            self.wiki_title_index.close()
            self.wiki_title_index = None


@dataclass
class PendingRequest:
    expected: int
    event: threading.Event
    results: list[dict]
    error: str | None = None


class UnifiedSearchEngine:
    def __init__(
        self,
        model_path: Path,
        embedding_root: Path,
        passage_root: Path,
        devices: list[int],
        encoder_device: int,
        sources: tuple[str, ...],
        worker_top_k: int,
        worker_batch_size: int,
        batch_timeout_s: float,
        query_batch_size: int,
        request_timeout_s: float,
    ):
        self.sources = sources
        self.request_timeout_s = request_timeout_s
        self.specs = load_shard_specs(embedding_root, passage_root, sources)
        self.store = PassageStore(self.specs)
        self.stats = Counter()
        self.stats_lock = threading.Lock()
        self.pending: dict[str, PendingRequest] = {}
        self.pending_lock = threading.Lock()
        self.url_to_locator: dict[tuple[str, str], tuple[str, str, int]] = {}
        self.ctx = mp.get_context("spawn")
        self.result_queue = self.ctx.Queue(maxsize=20000)
        self.request_queues = [self.ctx.Queue(maxsize=2000) for _ in devices]
        assignments = assign_shards(self.specs, len(devices))
        self.workers = []
        for worker_id, (device, specs, request_queue) in enumerate(
            zip(devices, assignments, self.request_queues)
        ):
            process = self.ctx.Process(
                target=retrieval_worker,
                args=(
                    worker_id,
                    device,
                    specs,
                    request_queue,
                    self.result_queue,
                    worker_top_k,
                    worker_batch_size,
                    batch_timeout_s,
                ),
                name=f"scad-retrieval-gpu-{device}",
            )
            process.start()
            self.workers.append(process)
        self.worker_info = self._wait_for_workers(timeout_s=1800)
        logger.info("All retrieval workers are ready: %s", self.worker_info)
        self.encoder = QueryEncoder(model_path, encoder_device)
        self.query_batcher = BatchedQueryEncoder(
            self.encoder,
            batch_size=query_batch_size,
            batch_timeout_s=batch_timeout_s,
        )
        self.collector = threading.Thread(target=self._collect_results, daemon=True)
        self.collector.start()

    def _wait_for_workers(self, timeout_s: float) -> list[dict]:
        ready = {}
        deadline = time.monotonic() + timeout_s
        while len(ready) < len(self.workers):
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out while loading retrieval shards")
            try:
                message = self.result_queue.get(timeout=5)
            except queue.Empty:
                failed = [process for process in self.workers if not process.is_alive()]
                if failed:
                    raise RuntimeError(f"Retrieval workers exited during startup: {failed}")
                continue
            if message["type"] == "worker_error":
                raise RuntimeError(f"Retrieval worker failed: {message}")
            if message["type"] == "ready":
                ready[message["worker_id"]] = message
        return [ready[index] for index in sorted(ready)]

    def _collect_results(self) -> None:
        while True:
            try:
                message = self.result_queue.get(timeout=1)
            except queue.Empty:
                continue
            if message is None:
                return
            if message["type"] == "worker_error":
                with self.pending_lock:
                    for pending in self.pending.values():
                        pending.error = message["error"]
                        pending.event.set()
                continue
            if message["type"] != "result":
                continue
            with self.pending_lock:
                pending = self.pending.get(message["request_id"])
                if pending is None:
                    continue
                pending.results.append(message)
                if len(pending.results) == pending.expected:
                    pending.event.set()

    def _merge_hits(self, worker_results: list[dict], k: int) -> list[dict]:
        hits = [hit for result in worker_results for hit in result["hits"]]
        hits.sort(key=lambda hit: (-hit["score"], hit["source"], hit["shard"], hit["row"]))
        return hits[:k]

    def search_sync(self, query: str, source: str, k: int) -> list[dict]:
        if source not in self.sources:
            raise ValueError(f"Source {source!r} is not loaded")
        if not query.strip():
            raise ValueError("Query must not be empty")
        k = min(max(int(k), 1), 100)
        vector = self.query_batcher.encode(query, self.request_timeout_s)
        request_id = uuid.uuid4().hex
        pending = PendingRequest(len(self.workers), threading.Event(), [])
        with self.pending_lock:
            self.pending[request_id] = pending
        job = {"request_id": request_id, "source": source, "vector": vector}
        try:
            for request_queue in self.request_queues:
                request_queue.put(job, timeout=10)
            if not pending.event.wait(self.request_timeout_s):
                raise TimeoutError(f"Retrieval timed out after {self.request_timeout_s}s")
            if pending.error:
                raise RuntimeError(pending.error)
            hits = self._merge_hits(pending.results, k)
            results = []
            for hit in hits:
                row = self.store.get(hit["source"], hit["shard"], hit["row"])
                locator = (hit["source"], hit["shard"], hit["row"])
                self.url_to_locator[(hit["source"], row["url"])] = locator
                results.append(
                    {
                        "docid": row["docid"],
                        "url": row["url"],
                        "title": row["title"],
                        "text": row["text"],
                        "score": hit["score"],
                        "source": hit["source"],
                    }
                )
            with self.stats_lock:
                self.stats[f"search/{source}"] += 1
            return results
        finally:
            with self.pending_lock:
                self.pending.pop(request_id, None)

    async def search(self, query: str, source: str, k: int) -> list[dict]:
        return await asyncio.to_thread(self.search_sync, query, source, k)

    def open(self, docid: str | None, url: str | None, source: str | None) -> list[dict]:
        locator = self.store.locate_docid(docid, source) if docid else None
        if locator is None and url:
            if source:
                locator = self.url_to_locator.get((source, url))
            else:
                for candidate_source in self.sources:
                    locator = self.url_to_locator.get((candidate_source, url))
                    if locator is not None:
                        break
        if locator is None:
            return [
                {
                    "docid": docid,
                    "url": url,
                    "text": "Document not found. Open a docid or URL returned by search.",
                }
            ]
        with self.stats_lock:
            self.stats[f"open/{locator[0]}"] += 1
        return [self.store.open_document(locator)]

    def shutdown(self) -> None:
        self.query_batcher.shutdown()
        self.store.close()
        for request_queue in self.request_queues:
            try:
                request_queue.put_nowait(None)
            except queue.Full:
                pass
        for process in self.workers:
            process.join(timeout=20)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        try:
            self.result_queue.put_nowait(None)
        except queue.Full:
            pass


class AnnSearchEngine:
    """CPU FAISS candidate retrieval with exact full-dimensional mmap reranking."""

    def __init__(
        self,
        model_path: Path,
        embedding_root: Path,
        passage_root: Path,
        ann_root: Path,
        encoder_device: int,
        sources: tuple[str, ...],
        candidate_k: int,
        nprobe: int,
        faiss_threads: int,
        batch_timeout_s: float,
        query_batch_size: int,
        request_timeout_s: float,
    ):
        self.sources = sources
        self.request_timeout_s = request_timeout_s
        self.specs = load_shard_specs(embedding_root, passage_root, sources)
        self.store = PassageStore(self.specs)
        self.stats = Counter()
        self.stats_lock = threading.Lock()
        self.pending: dict[str, bool] = {}
        self.pending_lock = threading.Lock()
        self.url_to_locator: dict[tuple[str, str], tuple[str, str, int]] = {}
        self.retriever = AnnRetriever(
            specs=self.specs,
            embedding_root=embedding_root,
            ann_root=ann_root,
            sources=sources,
            candidate_k=candidate_k,
            nprobe=nprobe,
            threads=faiss_threads,
        )
        self.worker_info = [self.retriever.info()]
        logger.info("ANN retrieval index is ready: %s", self.worker_info[0])
        self.encoder = QueryEncoder(model_path, encoder_device)
        self.query_batcher = BatchedQueryEncoder(
            self.encoder,
            batch_size=query_batch_size,
            batch_timeout_s=batch_timeout_s,
        )
        # Candidate matrices are large. Running one rerank in every FastAPI
        # request thread made native thread arenas retain tens of GiB and caused
        # a SIGSEGV under training load. A small persistent pool bounds both
        # native concurrency and retained memory while still allowing overlap.
        self.retrieval_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="scad-ann-rerank",
        )

    def _validate_search(self, query: str, source: str, k: int) -> int:
        if source not in self.sources:
            raise ValueError(f"Source {source!r} is not loaded")
        if not query.strip():
            raise ValueError("Query must not be empty")
        return min(max(int(k), 1), 100)

    def _search_vector(self, vector: np.ndarray, source: str, k: int) -> list[dict]:
        hits = self.retriever.search(vector, source, k)
        results = []
        for hit in hits:
            row = self.store.get(hit["source"], hit["shard"], hit["row"])
            locator = (hit["source"], hit["shard"], hit["row"])
            self.url_to_locator[(hit["source"], row["url"])] = locator
            results.append(
                {
                    "docid": row["docid"],
                    "url": row["url"],
                    "title": row["title"],
                    "text": row["text"],
                    "score": hit["score"],
                    "source": hit["source"],
                }
            )
        with self.stats_lock:
            self.stats[f"search/{source}"] += 1
        return results

    def search_sync(self, query: str, source: str, k: int) -> list[dict]:
        k = self._validate_search(query, source, k)
        request_id = uuid.uuid4().hex
        with self.pending_lock:
            self.pending[request_id] = True
        try:
            vector = self.query_batcher.encode(query, self.request_timeout_s)
            return self._search_vector(vector, source, k)
        finally:
            with self.pending_lock:
                self.pending.pop(request_id, None)

    async def search(self, query: str, source: str, k: int) -> list[dict]:
        k = self._validate_search(query, source, k)
        request_id = uuid.uuid4().hex
        with self.pending_lock:
            self.pending[request_id] = True
        try:
            # Keep concurrent callers here so BatchedQueryEncoder can combine
            # them into one GPU forward pass.
            vector = await asyncio.to_thread(
                self.query_batcher.encode,
                query,
                self.request_timeout_s,
            )
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self.retrieval_executor,
                self._search_vector,
                vector,
                source,
                k,
            )
        finally:
            with self.pending_lock:
                self.pending.pop(request_id, None)

    def open(self, docid: str | None, url: str | None, source: str | None) -> list[dict]:
        locator = self.store.locate_docid(docid, source) if docid else None
        if locator is None and url:
            if source:
                locator = self.url_to_locator.get((source, url))
            else:
                for candidate_source in self.sources:
                    locator = self.url_to_locator.get((candidate_source, url))
                    if locator is not None:
                        break
        if locator is None:
            return [
                {
                    "docid": docid,
                    "url": url,
                    "text": "Document not found. Open a docid or URL returned by search.",
                }
            ]
        with self.stats_lock:
            self.stats[f"open/{locator[0]}"] += 1
        return [self.store.open_document(locator)]

    def shutdown(self) -> None:
        self.query_batcher.shutdown()
        self.retrieval_executor.shutdown(wait=True, cancel_futures=True)
        self.store.close()


SERVER_CONFIG: dict[str, Any] = {}
search_engine: UnifiedSearchEngine | AnnSearchEngine | None = None


@asynccontextmanager
async def lifespan(_app):
    global search_engine
    backend = SERVER_CONFIG["backend"]
    engine_config = SERVER_CONFIG["engine_config"]
    if backend == "ann_cpu":
        search_engine = AnnSearchEngine(**engine_config)
    else:
        search_engine = UnifiedSearchEngine(**engine_config)
    try:
        yield
    finally:
        search_engine.shutdown()


app = FastAPI(title="SCAD Search", lifespan=lifespan)


@app.post("/search", response_model=QueryResponse)
async def search_endpoint(request: QueryRequest) -> QueryResponse:
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine is not ready")
    start = time.monotonic()
    try:
        results = await search_engine.search(request.query, request.source, request.k)
    except (ValueError, TimeoutError, RuntimeError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return QueryResponse(results=results, took_ms=(time.monotonic() - start) * 1000)


@app.post("/open", response_model=QueryResponse)
async def open_endpoint(request: OpenRequest) -> QueryResponse:
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine is not ready")
    start = time.monotonic()
    results = await asyncio.to_thread(search_engine.open, request.docid, request.url, request.source)
    return QueryResponse(results=results, took_ms=(time.monotonic() - start) * 1000)


@app.post("/embed", response_model=EmbedResponse)
async def embed_endpoint(request: EmbedRequest) -> EmbedResponse:
    """Bulk embeddings for post-rollout SCAD node canonicalization."""

    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine is not ready")
    start = time.monotonic()
    try:
        vectors = await asyncio.to_thread(
            bulk_encode,
            search_engine.encoder,
            request.texts,
            batch_size=request.batch_size,
            dim=request.dim,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    with search_engine.stats_lock:
        search_engine.stats["embed/requests"] += 1
        search_engine.stats["embed/texts"] += len(request.texts)
    return EmbedResponse(
        vectors=vectors.tolist(),
        took_ms=(time.monotonic() - start) * 1000,
    )


@app.get("/health")
async def health_endpoint() -> dict[str, Any]:
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine is not ready")
    rows = Counter()
    for spec in search_engine.specs:
        rows[spec.source] += spec.rows
    return {
        "status": "healthy",
        "sources": dict(rows),
        "workers": search_engine.worker_info,
    }


@app.get("/stats")
async def stats_endpoint() -> dict[str, Any]:
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine is not ready")
    with search_engine.pending_lock:
        pending = len(search_engine.pending)
    return {"requests": dict(search_engine.stats), "pending": pending}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/Qwen/Qwen3-Embedding-4B"),
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=Path("runtime/text/embeddings"),
    )
    parser.add_argument(
        "--passage-root",
        type=Path,
        default=Path("../SCAD_DATA/text/corpus/passages"),
    )
    parser.add_argument("--backend", choices=("ann_cpu", "exact_gpu"), default="ann_cpu")
    parser.add_argument(
        "--ann-root",
        type=Path,
        default=Path("runtime/text/indices"),
    )
    parser.add_argument("--ann-candidates", type=int, default=8192)
    parser.add_argument("--ann-nprobe", type=int, default=256)
    parser.add_argument("--faiss-threads", type=int, default=104)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--encoder-device", type=int, default=0)
    parser.add_argument("--sources", default=",".join(VALID_SOURCES))
    parser.add_argument("--worker-top-k", type=int, default=64)
    parser.add_argument("--worker-batch-size", type=int, default=32)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--batch-timeout-ms", type=float, default=10.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = [int(value) for value in args.devices.split(",") if value.strip()]
    sources = tuple(value.strip() for value in args.sources.split(",") if value.strip())
    unknown_sources = set(sources) - set(VALID_SOURCES)
    if unknown_sources:
        raise ValueError(f"Unknown sources: {sorted(unknown_sources)}")
    if args.backend == "exact_gpu" and not devices:
        raise ValueError("At least one retrieval device is required")
    required_devices = devices + [args.encoder_device] if args.backend == "exact_gpu" else [args.encoder_device]
    if max(required_devices) >= torch.cuda.device_count():
        raise ValueError(
            f"Requested CUDA devices {devices} / encoder {args.encoder_device}, "
            f"but only {torch.cuda.device_count()} are visible"
        )
    global SERVER_CONFIG
    common_config = {
        "model_path": args.model.resolve(),
        "embedding_root": args.embedding_root.resolve(),
        "passage_root": args.passage_root.resolve(),
        "encoder_device": args.encoder_device,
        "sources": sources,
        "batch_timeout_s": args.batch_timeout_ms / 1000,
        "query_batch_size": args.query_batch_size,
        "request_timeout_s": args.request_timeout,
    }
    if args.backend == "ann_cpu":
        engine_config = {
            **common_config,
            "ann_root": args.ann_root.resolve(),
            "candidate_k": args.ann_candidates,
            "nprobe": args.ann_nprobe,
            "faiss_threads": args.faiss_threads,
        }
    else:
        engine_config = {
            **common_config,
            "devices": devices,
            "worker_top_k": args.worker_top_k,
            "worker_batch_size": args.worker_batch_size,
        }
    SERVER_CONFIG = {"backend": args.backend, "engine_config": engine_config}

    def handle_sigterm(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)
    uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
