#!/usr/bin/env python3
"""Build resumable FAISS indexes for the unified SCAD embedding collection."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from envs.ann_retrieval import build_index_collection
from envs.search_server import load_shard_specs

VALID_SOURCES = ("wiki18", "browsecomp_plus", "webwalker", "mm_opd_v1")


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runtime/text/indices"),
    )
    parser.add_argument("--sources", default="wiki18,browsecomp_plus,webwalker")
    parser.add_argument("--ann-dimension", type=int, default=512)
    parser.add_argument("--pq-m", type=int, default=64)
    parser.add_argument("--pq-bits", type=int, default=8)
    parser.add_argument("--browsecomp-nlist", type=int, default=4096)
    parser.add_argument("--wiki18-nlist", type=int, default=4096)
    parser.add_argument("--flat-threshold", type=int, default=100_000)
    parser.add_argument("--train-vectors-per-centroid", type=int, default=50)
    parser.add_argument("--train-iterations", type=int, default=20)
    parser.add_argument("--add-batch-size", type=int, default=65_536)
    parser.add_argument("--coarse-hnsw-m", type=int, default=32)
    parser.add_argument("--coarse-ef-construction", type=int, default=200)
    parser.add_argument("--coarse-ef-search", type=int, default=512)
    parser.add_argument("--threads", type=int, default=104)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = tuple(value.strip() for value in args.sources.split(",") if value.strip())
    unknown = set(sources) - set(VALID_SOURCES)
    if unknown:
        raise ValueError(f"Unknown sources: {sorted(unknown)}")
    specs = load_shard_specs(args.embedding_root, args.passage_root, sources)
    manifest = build_index_collection(
        specs=specs,
        embedding_root=args.embedding_root,
        output_root=args.output_root,
        sources=sources,
        ann_dimension=args.ann_dimension,
        pq_m=args.pq_m,
        pq_bits=args.pq_bits,
        nlist_by_source={
            "browsecomp_plus": args.browsecomp_nlist,
            "wiki18": args.wiki18_nlist,
        },
        flat_threshold=args.flat_threshold,
        train_vectors_per_centroid=args.train_vectors_per_centroid,
        train_iterations=args.train_iterations,
        add_batch_size=args.add_batch_size,
        coarse_hnsw_m=args.coarse_hnsw_m,
        coarse_ef_construction=args.coarse_ef_construction,
        coarse_ef_search=args.coarse_ef_search,
        threads=args.threads,
        seed=args.seed,
        force=args.force,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
