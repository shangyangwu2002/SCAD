"""FAISS ANN indexing and full-dimensional reranking for the SCAD corpus."""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


logger = logging.getLogger(__name__)


def _import_faiss():
    try:
        import faiss
    except ImportError as error:  # pragma: no cover - exercised by deployment preflight
        raise RuntimeError(
            "faiss-cpu is required for ANN retrieval; install faiss-cpu==1.12.0"
        ) from error
    return faiss


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.divide(vectors, np.maximum(norms, 1e-12), out=vectors)
    return vectors


def _manifest_sha256(embedding_root: Path) -> str:
    return hashlib.sha256((embedding_root / "manifest.json").read_bytes()).hexdigest()


class EmbeddingStore:
    """Random and sequential access to source-global positions in sharded NPY files."""

    def __init__(self, specs: Iterable[Any]):
        by_source: dict[str, list[Any]] = defaultdict(list)
        for spec in specs:
            by_source[spec.source].append(spec)
        self.specs = {
            source: sorted(source_specs, key=lambda item: item.source_start)
            for source, source_specs in by_source.items()
        }
        self.starts = {
            source: np.asarray([spec.source_start for spec in source_specs], dtype=np.int64)
            for source, source_specs in self.specs.items()
        }
        self.arrays: dict[str, np.ndarray] = {}
        first_spec = next(iter(next(iter(self.specs.values()))))
        self.dimension = int(self._array(first_spec).shape[1])

    def _array(self, spec: Any) -> np.ndarray:
        array = self.arrays.get(spec.embedding_path)
        if array is None:
            array = np.load(spec.embedding_path, mmap_mode="r")
            self.arrays[spec.embedding_path] = array
        return array

    def rows(self, source: str) -> int:
        source_specs = self.specs[source]
        last = source_specs[-1]
        return int(last.source_start + last.rows)

    def locator(self, source: str, position: int) -> tuple[Any, int]:
        source_specs = self.specs[source]
        shard_index = bisect.bisect_right(self.starts[source], position) - 1
        if shard_index < 0:
            raise IndexError(position)
        spec = source_specs[shard_index]
        local_row = int(position - spec.source_start)
        if local_row < 0 or local_row >= spec.rows:
            raise IndexError(position)
        return spec, local_row

    def get_vectors(
        self,
        source: str,
        positions: np.ndarray,
        dimension: int | None = None,
    ) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.int64)
        if positions.ndim != 1:
            raise ValueError("positions must be one-dimensional")
        output_dimension = self.dimension if dimension is None else int(dimension)
        if output_dimension <= 0 or output_dimension > self.dimension:
            raise ValueError(f"Invalid output dimension: {output_dimension}")
        total = self.rows(source)
        if positions.size and (positions.min() < 0 or positions.max() >= total):
            raise IndexError(f"Position outside source={source} with {total} rows")
        output = np.empty((len(positions), output_dimension), dtype=np.float32)
        if not len(positions):
            return output
        shard_indices = np.searchsorted(self.starts[source], positions, side="right") - 1
        for shard_index in np.unique(shard_indices):
            mask = shard_indices == shard_index
            spec = self.specs[source][int(shard_index)]
            local_rows = positions[mask] - spec.source_start
            output[mask] = self._array(spec)[local_rows, :output_dimension]
        return output

    def iter_batches(
        self,
        source: str,
        batch_size: int,
        dimension: int,
    ) -> Iterable[np.ndarray]:
        for spec in self.specs[source]:
            array = self._array(spec)
            for start in range(0, spec.rows, batch_size):
                stop = min(start + batch_size, spec.rows)
                yield np.asarray(array[start:stop, :dimension], dtype=np.float32)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_index_collection(
    specs: Iterable[Any],
    embedding_root: Path,
    output_root: Path,
    sources: tuple[str, ...],
    ann_dimension: int = 512,
    pq_m: int = 64,
    pq_bits: int = 8,
    nlist_by_source: dict[str, int] | None = None,
    flat_threshold: int = 100_000,
    train_vectors_per_centroid: int = 50,
    train_iterations: int = 20,
    add_batch_size: int = 65_536,
    coarse_hnsw_m: int = 32,
    coarse_ef_construction: int = 200,
    coarse_ef_search: int = 512,
    threads: int = 104,
    seed: int = 42,
    force: bool = False,
) -> dict[str, Any]:
    """Build resumable per-source indexes whose IDs are source-global row positions."""

    faiss = _import_faiss()
    faiss.omp_set_num_threads(threads)
    specs = list(specs)
    store = EmbeddingStore(specs)
    if ann_dimension > store.dimension:
        raise ValueError(f"ANN dimension {ann_dimension} exceeds embedding dimension {store.dimension}")
    if ann_dimension % pq_m:
        raise ValueError(f"ANN dimension {ann_dimension} must be divisible by pq_m={pq_m}")
    nlist_by_source = nlist_by_source or {"browsecomp_plus": 4096, "wiki18": 4096}
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    signature = {
        "schema_version": "scad_ann_ivfpq_v1",
        "embedding_manifest_sha256": _manifest_sha256(embedding_root),
        "embedding_dimension": store.dimension,
        "ann_dimension": ann_dimension,
        "pq_m": pq_m,
        "pq_bits": pq_bits,
        "flat_threshold": flat_threshold,
        "seed": seed,
    }
    if existing and any(existing.get(key) != value for key, value in signature.items()):
        if not force:
            raise ValueError(f"Existing ANN manifest is incompatible: {manifest_path}; use --force")
        existing = {}
    manifest = {**signature, "sources": dict(existing.get("sources", {}))}

    for source_index, source in enumerate(sources):
        total_rows = store.rows(source)
        index_path = output_root / f"{source}.faiss"
        previous = manifest["sources"].get(source, {})
        if (
            not force
            and index_path.is_file()
            and int(previous.get("rows", -1)) == total_rows
            and int(previous.get("ann_dimension", -1)) == ann_dimension
        ):
            logger.info("Keeping completed ANN index for %s: %s", source, index_path)
            continue

        started = time.monotonic()
        if total_rows <= flat_threshold:
            index = faiss.IndexFlatIP(ann_dimension)
            index_type = "flat_ip"
            nlist = 0
            train_samples = 0
        else:
            nlist = int(nlist_by_source.get(source, 4096))
            if nlist <= 0:
                raise ValueError(f"nlist must be positive for source={source}")
            quantizer = faiss.IndexHNSWFlat(
                ann_dimension,
                coarse_hnsw_m,
                faiss.METRIC_INNER_PRODUCT,
            )
            quantizer.hnsw.efConstruction = coarse_ef_construction
            quantizer.hnsw.efSearch = coarse_ef_search
            index = faiss.IndexIVFPQ(
                quantizer,
                ann_dimension,
                nlist,
                pq_m,
                pq_bits,
                faiss.METRIC_INNER_PRODUCT,
            )
            index.cp.niter = train_iterations
            index.cp.seed = seed + source_index
            index.cp.verbose = True
            index.pq.cp.niter = train_iterations
            index.pq.cp.seed = seed + source_index
            index.pq.cp.verbose = True
            train_samples = min(total_rows, max(65_536, nlist * train_vectors_per_centroid))
            rng = np.random.default_rng(seed + source_index)
            train_positions = np.sort(rng.choice(total_rows, size=train_samples, replace=False))
            logger.info(
                "Training %s IVF-PQ: rows=%d dim=%d nlist=%d m=%d samples=%d",
                source,
                total_rows,
                ann_dimension,
                nlist,
                pq_m,
                train_samples,
            )
            training_vectors = _normalize_rows(
                store.get_vectors(source, train_positions, dimension=ann_dimension)
            )
            index.train(training_vectors)
            del training_vectors, train_positions
            index_type = "ivfpq_hnsw_ip"

        added = 0
        for vectors in store.iter_batches(source, add_batch_size, ann_dimension):
            vectors = _normalize_rows(vectors)
            index.add(vectors)
            added += len(vectors)
            if added == total_rows or added % 1_000_000 < len(vectors):
                logger.info("Adding %s vectors: %d/%d", source, added, total_rows)
        if index.ntotal != total_rows:
            raise RuntimeError(f"Index row mismatch for {source}: {index.ntotal} != {total_rows}")

        temporary = output_root / f".{source}.faiss.tmp"
        faiss.write_index(index, str(temporary))
        os.replace(temporary, index_path)
        manifest["sources"][source] = {
            "ann_dimension": ann_dimension,
            "build_seconds": round(time.monotonic() - started, 3),
            "file": index_path.name,
            "index_type": index_type,
            "nlist": nlist,
            "coarse_hnsw_m": coarse_hnsw_m if nlist else 0,
            "coarse_ef_search": coarse_ef_search if nlist else 0,
            "rows": total_rows,
            "train_samples": train_samples,
        }
        _write_json_atomic(manifest_path, manifest)
        logger.info("Completed %s ANN index: %s", source, index_path)

    missing = set(sources) - set(manifest["sources"])
    if missing:
        raise RuntimeError(f"ANN manifest is incomplete: {sorted(missing)}")
    return manifest


class AnnRetriever:
    """ANN candidate generation followed by exact full-dimensional cosine reranking."""

    def __init__(
        self,
        specs: Iterable[Any],
        embedding_root: Path,
        ann_root: Path,
        sources: tuple[str, ...],
        candidate_k: int = 8192,
        nprobe: int = 256,
        threads: int = 104,
    ):
        self.faiss = _import_faiss()
        self.faiss.omp_set_num_threads(threads)
        self.store = EmbeddingStore(specs)
        self.sources = sources
        self.candidate_k = candidate_k
        self.nprobe = nprobe
        self.lock = threading.Lock()
        manifest_path = ann_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"ANN index manifest is missing: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha = _manifest_sha256(embedding_root)
        if self.manifest.get("embedding_manifest_sha256") != expected_sha:
            raise ValueError("ANN index was built from a different embedding manifest")
        self.ann_dimension = int(self.manifest["ann_dimension"])
        self.indexes = {}
        for source in sources:
            metadata = self.manifest.get("sources", {}).get(source)
            if metadata is None:
                raise ValueError(f"ANN index does not contain source={source}")
            if int(metadata["rows"]) != self.store.rows(source):
                raise ValueError(f"ANN/source row mismatch for source={source}")
            index = self.faiss.read_index(str(ann_root / metadata["file"]))
            if index.ntotal != self.store.rows(source) or index.d != self.ann_dimension:
                raise ValueError(f"Invalid FAISS index for source={source}")
            if hasattr(index, "nprobe"):
                index.nprobe = min(nprobe, int(metadata["nlist"]))
                quantizer = self.faiss.downcast_index(index.quantizer)
                if hasattr(quantizer, "hnsw"):
                    quantizer.hnsw.efSearch = max(
                        index.nprobe,
                        int(metadata.get("coarse_ef_search", index.nprobe)),
                    )
            self.indexes[source] = index

    def search(self, vector: np.ndarray, source: str, k: int) -> list[dict[str, Any]]:
        if source not in self.indexes:
            raise ValueError(f"Source {source!r} is not indexed")
        full_query = np.asarray(vector, dtype=np.float32)
        if full_query.shape != (self.store.dimension,):
            raise ValueError(
                f"Query vector shape {full_query.shape} does not match {(self.store.dimension,)}"
            )
        ann_query = _normalize_rows(full_query[None, : self.ann_dimension])
        candidate_count = min(self.candidate_k, self.store.rows(source))
        with self.lock:
            _, candidate_ids = self.indexes[source].search(ann_query, candidate_count)
        positions = candidate_ids[0]
        positions = positions[positions >= 0]
        if not len(positions):
            return []
        positions = np.unique(positions)
        candidates = self.store.get_vectors(source, positions)
        exact_scores = candidates @ full_query
        take = min(k, len(positions))
        if take < len(positions):
            selected = np.argpartition(exact_scores, -take)[-take:]
        else:
            selected = np.arange(len(positions))
        selected = selected[np.lexsort((positions[selected], -exact_scores[selected]))]
        output = []
        for index in selected:
            spec, local_row = self.store.locator(source, int(positions[index]))
            output.append(
                {
                    "score": float(exact_scores[index]),
                    "source": source,
                    "shard": spec.name,
                    "row": local_row,
                    "position": int(positions[index]),
                }
            )
        return output

    def info(self) -> dict[str, Any]:
        return {
            "backend": "faiss_ann_cpu_full_rerank",
            "ann_dimension": self.ann_dimension,
            "candidate_k": self.candidate_k,
            "nprobe": self.nprobe,
            "sources": {
                source: {
                    "rows": self.store.rows(source),
                    "index_type": self.manifest["sources"][source]["index_type"],
                    "nlist": self.manifest["sources"][source]["nlist"],
                }
                for source in self.sources
            },
        }
