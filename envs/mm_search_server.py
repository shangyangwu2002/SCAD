#!/usr/bin/env python3
"""Isolated multimodal retrieval server over the four SCAD corpus sources."""

from __future__ import annotations

import argparse
import asyncio
import signal
from concurrent.futures import ThreadPoolExecutor
import sys
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.ann_retrieval import _normalize_rows
from envs.search_server import AnnSearchEngine, UnifiedSearchEngine, bulk_encode


VALID_SOURCES = ("mm_opd_v1", "browsecomp_plus", "webwalker", "wiki18")
FROZEN_MM_SOURCES = ("mm_opd_v1", "browsecomp_plus", "webwalker")


def parse_sources(value: str | None) -> tuple[str, ...]:
    if not value or value.strip() == "unified_mm":
        return VALID_SOURCES
    sources = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    unknown = set(sources) - set(VALID_SOURCES)
    if unknown:
        raise ValueError(f"Unknown sources: {sorted(unknown)}")
    return sources


def merge_source_results(results: dict[str, list[dict]], k: int) -> list[dict]:
    rows = [dict(row) for source_rows in results.values() for row in source_rows]
    rows.sort(key=lambda row: (-float(row.get("score", float("-inf"))), str(row.get("source", "")), str(row.get("docid", ""))))
    return rows[: min(max(int(k), 1), 100)]


class ParentDocumentStore:
    """Document lookup for Lens' 5e9 parent document IDs."""

    def __init__(self, document_root: Path) -> None:
        self.rows: dict[str, dict] = {}
        document_root = Path(document_root)
        paths = sorted(document_root.glob("*.parquet"))
        if not paths:
            paths = sorted((document_root / "mm_opd_v1").glob("*.parquet"))
        for path in paths:
            table = pq.read_table(path)
            for row in table.to_pylist():
                self.rows[str(row["docid"])] = row

    def open(self, docid: str | None) -> list[dict] | None:
        row = self.rows.get(str(docid))
        return [dict(row)] if row is not None else None


def _ann_search_capped(retriever, vector: np.ndarray, source: str, k: int, candidate_k: int) -> list[dict]:
    """AnnRetriever.search with a per-call candidate budget (flat indexes need far fewer rerank candidates)."""
    full_query = np.asarray(vector, dtype=np.float32)
    ann_query = _normalize_rows(full_query[None, : retriever.ann_dimension])
    candidate_count = min(int(candidate_k), retriever.store.rows(source))
    with retriever.lock:
        _, candidate_ids = retriever.indexes[source].search(ann_query, candidate_count)
    positions = candidate_ids[0]
    positions = np.unique(positions[positions >= 0])
    if not len(positions):
        return []
    exact_scores = retriever.store.get_vectors(source, positions) @ full_query
    take = min(k, len(positions))
    selected = np.argpartition(exact_scores, -take)[-take:] if take < len(positions) else np.arange(len(positions))
    selected = selected[np.lexsort((positions[selected], -exact_scores[selected]))]
    output = []
    for index in selected:
        spec, local_row = retriever.store.locator(source, int(positions[index]))
        output.append({"score": float(exact_scores[index]), "source": source, "shard": spec.name, "row": local_row, "position": int(positions[index])})
    return output


class MultiSourceEngine:
    """Fan a query out over several sources: encode once, rerank in a bounded pool, cap flat-index candidates."""

    def __init__(self, engine, parent_documents: ParentDocumentStore, rerank_workers: int = 4, flat_candidate_k: int = 4096) -> None:
        self.engine, self.parent_documents = engine, parent_documents
        self.flat_candidate_k = int(flat_candidate_k)
        self.executor = ThreadPoolExecutor(max_workers=max(1, int(rerank_workers)), thread_name_prefix="mm-rerank")
        self._ann = hasattr(engine, "retriever") and hasattr(engine, "query_batcher") and hasattr(engine, "_search_vector")

    def _search_source(self, vector: np.ndarray, source: str, k: int) -> list[dict]:
        retriever = self.engine.retriever
        index = retriever.indexes[source]
        if hasattr(index, "nprobe") or retriever.candidate_k <= self.flat_candidate_k:
            return self.engine._search_vector(vector, source, k)
        hits = _ann_search_capped(retriever, vector, source, k, self.flat_candidate_k)
        results = []
        for hit in hits:
            row = self.engine.store.get(hit["source"], hit["shard"], hit["row"])
            self.engine.url_to_locator[(hit["source"], row["url"])] = (hit["source"], hit["shard"], hit["row"])
            results.append({"docid": row["docid"], "url": row["url"], "title": row["title"], "text": row["text"],
                            "score": hit["score"], "source": hit["source"]})
        with self.engine.stats_lock:
            self.engine.stats[f"search/{source}"] += 1
        return results

    async def search(self, query: str, source: str, k: int) -> list[dict]:
        sources = parse_sources(source)
        if source == "unified_mm":
            sources = FROZEN_MM_SOURCES
        if not self._ann:
            per_source = await asyncio.gather(*(self.engine.search(query, item, k) for item in sources))
            return merge_source_results(dict(zip(sources, per_source)), k)
        k = min(self.engine._validate_search(query, item, k) for item in sources)
        vector = await asyncio.to_thread(self.engine.query_batcher.encode, query, self.engine.request_timeout_s)
        loop = asyncio.get_running_loop()
        per_source = await asyncio.gather(*(loop.run_in_executor(self.executor, self._search_source, vector, item, k) for item in sources))
        return merge_source_results(dict(zip(sources, per_source)), k)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def open(self, docid: str | None, url: str | None, source: str | None) -> list[dict]:
        if docid:
            direct = self.parent_documents.open(docid)
            if direct is not None:
                return direct
        normalized_source = None if source in (None, "", "unified_mm") or "," in str(source) else source
        return self.engine.open(docid, url, normalized_source)


class QueryRequest(BaseModel):
    query: str
    k: int = 20
    source: str = "unified_mm"


class OpenRequest(BaseModel):
    docid: str | None = None
    url: str | None = None
    source: str | None = None


class QueryResponse(BaseModel):
    results: list[dict[str, Any]]
    took_ms: float


class EmbedRequest(BaseModel):
    texts: list[str]
    dim: int | None = None
    batch_size: int = 64


SERVER_CONFIG: dict[str, Any] = {}
search_engine: MultiSourceEngine | None = None


@asynccontextmanager
async def lifespan(_app):
    global search_engine
    config = dict(SERVER_CONFIG["engine_config"])
    documents = ParentDocumentStore(SERVER_CONFIG["document_root"])
    engine_cls = AnnSearchEngine if SERVER_CONFIG["backend"] == "ann_cpu" else UnifiedSearchEngine
    underlying = engine_cls(**config)
    search_engine = MultiSourceEngine(underlying, documents, rerank_workers=SERVER_CONFIG.get("rerank_workers", 4),
                                      flat_candidate_k=SERVER_CONFIG.get("flat_candidate_k", 4096))
    try:
        yield
    finally:
        search_engine.shutdown()
        underlying.shutdown()
        search_engine = None


app = FastAPI(title="Unified MM Search", lifespan=lifespan)


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


@app.post("/embed")
async def embed_endpoint(request: EmbedRequest) -> dict:
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine is not ready")
    start = time.monotonic()
    vectors = await asyncio.to_thread(bulk_encode, search_engine.engine.encoder, request.texts, batch_size=request.batch_size, dim=request.dim)
    return {"vectors": vectors.tolist(), "took_ms": (time.monotonic() - start) * 1000}


@app.get("/health")
async def health_endpoint() -> dict:
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine is not ready")
    rows = Counter()
    for spec in search_engine.engine.specs:
        rows[spec.source] += spec.rows
    return {"status": "healthy", "sources": dict(rows), "workers": search_engine.engine.worker_info}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/Qwen/Qwen3-Embedding-4B"))
    parser.add_argument("--embedding-root", type=Path, default=Path("runtime/multimodal/embeddings"))
    parser.add_argument("--passage-root", type=Path, default=Path("../SCAD_DATA/multimodal/corpus/passages"))
    parser.add_argument("--document-root", type=Path, default=Path("../SCAD_DATA/multimodal/corpus/documents"))
    parser.add_argument("--backend", choices=("ann_cpu", "exact_gpu"), default="ann_cpu")
    parser.add_argument("--ann-root", type=Path, default=Path("runtime/multimodal/indices"))
    parser.add_argument("--ann-candidates", type=int, default=8192)
    parser.add_argument("--ann-nprobe", type=int, default=256)
    parser.add_argument("--faiss-threads", type=int, default=104)
    parser.add_argument("--rerank-workers", type=int, default=4, help="bounded pool for per-source candidate reranking")
    parser.add_argument("--flat-candidate-k", type=int, default=4096, help="rerank candidates for flat (exact) indexes")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--encoder-device", type=int, default=0)
    parser.add_argument("--sources", default="unified_mm")
    parser.add_argument("--worker-top-k", type=int, default=64)
    parser.add_argument("--worker-batch-size", type=int, default=32)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--batch-timeout-ms", type=float, default=10.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.port == 17897:
        raise ValueError("Port 17897 is forbidden for multimodal retrieval")
    sources = parse_sources(args.sources)
    devices = [int(value) for value in args.devices.split(",") if value.strip()]
    if args.backend == "exact_gpu" and not devices:
        raise ValueError("At least one retrieval device is required")
    required = devices + [args.encoder_device] if args.backend == "exact_gpu" else [args.encoder_device]
    if max(required) >= torch.cuda.device_count():
        raise ValueError(f"Requested CUDA devices {devices} / encoder {args.encoder_device}, but only {torch.cuda.device_count()} are visible")
    common = dict(model_path=args.model.resolve(), embedding_root=args.embedding_root.resolve(), passage_root=args.passage_root.resolve(),
                  encoder_device=args.encoder_device, sources=sources, batch_timeout_s=args.batch_timeout_ms / 1000,
                  query_batch_size=args.query_batch_size, request_timeout_s=args.request_timeout)
    if args.backend == "ann_cpu":
        engine_config = {**common, "ann_root": args.ann_root.resolve(), "candidate_k": args.ann_candidates,
                         "nprobe": args.ann_nprobe, "faiss_threads": args.faiss_threads}
    else:
        engine_config = {**common, "devices": devices, "worker_top_k": args.worker_top_k,
                         "worker_batch_size": args.worker_batch_size}
    global SERVER_CONFIG
    SERVER_CONFIG = {"backend": args.backend, "engine_config": engine_config, "document_root": args.document_root.resolve(),
                     "rerank_workers": args.rerank_workers, "flat_candidate_k": args.flat_candidate_k}
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
    uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
