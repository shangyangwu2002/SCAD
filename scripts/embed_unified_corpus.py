#!/usr/bin/env python3
"""Encode unified SCAD passages with Qwen3-Embedding-4B."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq


SOURCE_ORDER = ("wiki18", "browsecomp_plus", "webwalker")


@dataclass(frozen=True)
class ShardSpec:
    source: str
    input_path: Path
    relative_path: str
    rows: int


@dataclass(frozen=True)
class WorkerContext:
    rank: int
    world_size: int
    local_rank: int
    device: str


def resolve_worker_context(
    environment: Mapping[str, str],
    cuda_device_count: int,
) -> WorkerContext:
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in required if name not in environment]
    if missing:
        raise ValueError(f"Missing torchrun environment variables: {missing}")
    try:
        rank, world_size, local_rank = (int(environment[name]) for name in required)
    except (TypeError, ValueError) as error:
        raise ValueError("torchrun rank values must be integers") from error
    if cuda_device_count <= 0:
        raise ValueError("No CUDA devices are visible")
    if world_size <= 0 or world_size > cuda_device_count:
        raise ValueError(
            f"WORLD_SIZE={world_size} is incompatible with {cuda_device_count} CUDA devices"
        )
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK={rank} is outside WORLD_SIZE={world_size}")
    if local_rank < 0 or local_rank >= cuda_device_count:
        raise ValueError(
            f"LOCAL_RANK={local_rank} is outside {cuda_device_count} CUDA devices"
        )
    return WorkerContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=f"cuda:{local_rank}",
    )


def discover_shards(passages_root: Path, sources=SOURCE_ORDER) -> list[ShardSpec]:
    passages_root = Path(passages_root)
    for source in sources:
        if source not in (*SOURCE_ORDER, "mm_opd_v1"):
            raise ValueError(f"Unknown corpus source: {source}")
    flat_paths = sorted(passages_root.glob("part-*.parquet"))
    if flat_paths:
        by_source = {source: [] for source in (*SOURCE_ORDER, "mm_opd_v1")}
        for path in flat_paths:
            parquet_file = pq.ParquetFile(path)
            missing = {"docid", "text", "source"} - set(parquet_file.schema_arrow.names)
            if missing:
                raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
            values = parquet_file.read(columns=["source"]).column("source").unique().to_pylist()
            if len(values) != 1 or values[0] not in by_source:
                raise ValueError(f"{path} must contain exactly one supported source value")
            source = values[0]
            by_source[source].append(ShardSpec(source, path, path.name, parquet_file.metadata.num_rows))
        for source in sources:
            if not by_source[source]:
                raise FileNotFoundError(f"No passage shards found for {source}: {passages_root}")
        return [shard for source in sources for shard in by_source[source]]

    shards = []
    for source in sources:
        source_dir = passages_root / source
        paths = sorted(source_dir.glob("part-*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No passage shards found for {source}: {source_dir}")
        for path in paths:
            parquet_file = pq.ParquetFile(path)
            names = set(parquet_file.schema_arrow.names)
            missing = {"docid", "text"} - names
            if missing:
                raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
            shards.append(
                ShardSpec(
                    source=source,
                    input_path=path,
                    relative_path=path.relative_to(passages_root).as_posix(),
                    rows=parquet_file.metadata.num_rows,
                )
            )
    return shards


def assign_shards(
    shards: Sequence[ShardSpec],
    rank: int,
    world_size: int,
) -> list[ShardSpec]:
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid rank/world_size: rank={rank}, world_size={world_size}")
    return [shard for index, shard in enumerate(shards) if index % world_size == rank]


def plan_batches(
    lengths: Sequence[int],
    max_batch_tokens: int,
    max_batch_size: int,
) -> list[list[int]]:
    if max_batch_tokens <= 0 or max_batch_size <= 0:
        raise ValueError("Batch limits must be positive")
    ordered = sorted(range(len(lengths)), key=lambda index: (lengths[index], index))
    batches: list[list[int]] = []
    current: list[int] = []
    for index in ordered:
        length = int(lengths[index])
        if length <= 0:
            raise ValueError(f"Token length must be positive: {length}")
        if length > max_batch_tokens:
            raise ValueError(
                f"Sequence length {length} exceeds max_batch_tokens {max_batch_tokens}"
            )
        candidate_size = len(current) + 1
        if current and (
            candidate_size > max_batch_size
            or length * candidate_size > max_batch_tokens
        ):
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches


def last_token_pool(last_hidden_state, attention_mask):
    """Pool the final real token for uniformly left- or right-padded batches."""
    import torch

    batch_size = last_hidden_state.shape[0]
    if int(attention_mask[:, -1].sum().item()) == batch_size:
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_state[
        torch.arange(batch_size, device=last_hidden_state.device),
        sequence_lengths,
    ]


def build_embedding_config(
    model_path: Path | str,
    dimension: int,
    max_length: int,
    max_batch_tokens: int,
    max_batch_size: int,
    attention_backend: str = "flash_attention_2",
) -> dict:
    return {
        "model": Path(model_path).name,
        "dimension": dimension,
        "dtype": "float16",
        "model_dtype": "bfloat16",
        "max_length": max_length,
        "pooling": "last_token",
        "normalized": True,
        "document_instruction": "",
        "attention_backend": attention_backend,
        "max_batch_tokens": max_batch_tokens,
        "max_batch_size": max_batch_size,
    }


class QwenPassageEncoder:
    def __init__(
        self,
        tokenizer,
        model,
        device: str,
        model_identity: str,
        dimension: int = 2_560,
        max_length: int = 512,
        max_batch_tokens: int = 32_768,
        max_batch_size: int = 256,
        attention_backend: str = "injected",
    ) -> None:
        import torch

        if int(model.config.hidden_size) != dimension:
            raise ValueError(
                f"Model hidden_size={model.config.hidden_size} does not match dimension={dimension}"
            )
        if max_length <= 0 or max_batch_tokens < max_length or max_batch_size <= 0:
            raise ValueError("Invalid sequence or batch limits")
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.device = device
        self.dimension = dimension
        self.max_length = max_length
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_size = max_batch_size
        self._torch = torch
        self.config = build_embedding_config(
            model_path=model_identity,
            dimension=dimension,
            max_length=max_length,
            max_batch_tokens=max_batch_tokens,
            max_batch_size=max_batch_size,
            attention_backend=attention_backend,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: Path,
        device: str,
        dimension: int = 2_560,
        max_length: int = 512,
        max_batch_tokens: int = 32_768,
        max_batch_size: int = 256,
    ) -> "QwenPassageEncoder":
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_path = Path(model_path).resolve()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=True,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = False
        model.to(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        return cls(
            tokenizer=tokenizer,
            model=model,
            device=device,
            model_identity=str(model_path),
            dimension=dimension,
            max_length=max_length,
            max_batch_tokens=max_batch_tokens,
            max_batch_size=max_batch_size,
            attention_backend="flash_attention_2",
        )

    def encode_indexed(self, texts: Sequence[str]):
        tokenized = self.tokenizer(
            list(texts),
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        lengths = [int(sum(mask)) for mask in tokenized["attention_mask"]]
        batches = plan_batches(
            lengths,
            max_batch_tokens=self.max_batch_tokens,
            max_batch_size=self.max_batch_size,
        )
        torch = self._torch
        for indices in batches:
            selected = {
                name: [values[index] for index in indices]
                for name, values in tokenized.items()
            }
            padded = self.tokenizer.pad(
                selected,
                padding=True,
                return_tensors="pt",
            )
            padded = {
                name: value.to(self.device, non_blocking=True)
                for name, value in padded.items()
            }
            with torch.inference_mode():
                outputs = self.model(**padded, use_cache=False)
                pooled = last_token_pool(
                    outputs.last_hidden_state,
                    padded["attention_mask"],
                )
                normalized = torch.nn.functional.normalize(
                    pooled.float(),
                    p=2,
                    dim=1,
                )
            yield indices, normalized.to(dtype=torch.float16).cpu().numpy()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_paths(spec: ShardSpec, output_root: Path) -> tuple[Path, Path]:
    stem = spec.input_path.stem
    array_path = Path(output_root) / "embeddings" / spec.source / f"{stem}.npy"
    metadata_path = Path(output_root) / "metadata" / spec.source / f"{stem}.json"
    return array_path, metadata_path


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.stem}.tmp-{os.getpid()}{path.suffix}")
    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.stem}.tmp-{os.getpid()}{path.suffix}")
    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_completed_shard(
    spec: ShardSpec,
    output_root: Path,
    expected_config: dict,
) -> dict | None:
    array_path, metadata_path = _output_paths(spec, Path(output_root))
    if not array_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != "scad_embeddings_v1":
            return None
        for key, expected_value in expected_config.items():
            if metadata.get(key) != expected_value:
                return None
        if metadata.get("input_relative_path") != spec.relative_path:
            return None
        if metadata.get("input_rows") != spec.rows:
            return None
        if metadata.get("input_sha256") != _sha256_file(spec.input_path):
            return None
        if metadata.get("output_bytes") != array_path.stat().st_size:
            return None
        if metadata.get("output_sha256") != _sha256_file(array_path):
            return None
        vectors = np.load(array_path, mmap_mode="r")
        if vectors.shape != (spec.rows, int(expected_config["dimension"])):
            return None
        if vectors.dtype != np.dtype("float16"):
            return None
        norm_sample = metadata.get("norm_sample", {})
        if not (
            int(norm_sample.get("count", 0)) > 0
            and 0.999 <= float(norm_sample.get("min", 0.0)) <= 1.001
            and 0.999 <= float(norm_sample.get("max", 0.0)) <= 1.001
        ):
            return None
        return metadata
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _norm_sample(vectors: np.ndarray, rows: int) -> dict:
    count = min(rows, 1_024)
    indices = np.linspace(0, rows - 1, num=count, dtype=np.int64)
    sampled = np.asarray(vectors[indices], dtype=np.float32)
    norms = np.linalg.norm(sampled, axis=1)
    return {
        "count": count,
        "min": round(float(norms.min()), 6),
        "max": round(float(norms.max()), 6),
        "mean": round(float(norms.mean()), 6),
    }


def encode_shard(
    spec: ShardSpec,
    output_root: Path,
    encoder,
    dimension: int,
    buffer_rows: int = 8_192,
) -> dict:
    if dimension <= 0 or buffer_rows <= 0:
        raise ValueError("dimension and buffer_rows must be positive")
    expected_config = dict(encoder.config)
    if expected_config.get("dimension") != dimension:
        raise ValueError("Encoder dimension does not match requested dimension")
    completed = validate_completed_shard(spec, output_root, expected_config)
    if completed is not None:
        return completed

    array_path, metadata_path = _output_paths(spec, Path(output_root))
    if array_path.exists() or metadata_path.exists():
        raise ValueError(f"Existing output is incomplete or corrupt: {array_path}")
    array_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_array = array_path.with_name(f"{array_path.stem}.tmp-{os.getpid()}.npy")
    input_sha256 = _sha256_file(spec.input_path)
    memory_map = None
    try:
        memory_map = np.lib.format.open_memmap(
            temporary_array,
            mode="w+",
            dtype=np.dtype("<f2"),
            shape=(spec.rows, dimension),
        )
        input_file = pq.ParquetFile(spec.input_path)
        row_offset = 0
        for record_batch in input_file.iter_batches(columns=["text"], batch_size=buffer_rows):
            texts = record_batch.column(0).to_pylist()
            written = np.zeros(len(texts), dtype=bool)
            for local_indices, embeddings in encoder.encode_indexed(texts):
                indices = np.asarray(local_indices, dtype=np.int64)
                values = np.asarray(embeddings)
                if values.shape != (len(indices), dimension):
                    raise ValueError(
                        f"Encoder returned shape {values.shape}, expected {(len(indices), dimension)}"
                    )
                if np.any(indices < 0) or np.any(indices >= len(texts)):
                    raise ValueError("Encoder returned an out-of-range row index")
                if np.any(written[indices]):
                    raise ValueError("Encoder returned a duplicate row index")
                if not np.isfinite(values).all():
                    raise ValueError("Encoder returned non-finite embeddings")
                memory_map[row_offset + indices] = values.astype(np.float16, copy=False)
                written[indices] = True
            if not written.all():
                raise ValueError("Encoder did not return every input row")
            row_offset += len(texts)
        if row_offset != spec.rows:
            raise ValueError(
                f"Input row count changed: manifest={spec.rows}, encoded={row_offset}"
            )
        memory_map.flush()
        del memory_map
        memory_map = None
        with temporary_array.open("rb") as stream:
            os.fsync(stream.fileno())

        vectors = np.load(temporary_array, mmap_mode="r")
        norm_sample = _norm_sample(vectors, spec.rows)
        if norm_sample["min"] < 0.999 or norm_sample["max"] > 1.001:
            raise ValueError(f"Embedding norm validation failed: {norm_sample}")
        output_sha256 = _sha256_file(temporary_array)
        metadata = {
            "schema_version": "scad_embeddings_v1",
            **expected_config,
            "source": spec.source,
            "input_relative_path": spec.relative_path,
            "input_rows": spec.rows,
            "input_sha256": input_sha256,
            "output_relative_path": array_path.relative_to(output_root).as_posix(),
            "output_bytes": temporary_array.stat().st_size,
            "output_sha256": output_sha256,
            "norm_sample": norm_sample,
        }
        os.replace(temporary_array, array_path)
        _atomic_json(metadata_path, metadata)
        return metadata
    finally:
        if memory_map is not None:
            memory_map.flush()
        temporary_array.unlink(missing_ok=True)


def finalize_outputs(
    shards: Sequence[ShardSpec],
    output_root: Path,
    corpus_manifest_path: Path,
    model_path: Path,
    expected_config: dict,
) -> dict:
    output_root = Path(output_root)
    corpus_manifest_path = Path(corpus_manifest_path)
    model_path = Path(model_path)
    if not corpus_manifest_path.is_file():
        raise FileNotFoundError(f"Corpus manifest not found: {corpus_manifest_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    completed_shards = []
    rows_by_source: dict[str, int] = {}
    checksum_entries: list[tuple[str, str]] = []
    for spec in shards:
        metadata = validate_completed_shard(spec, output_root, expected_config)
        if metadata is None:
            raise ValueError(f"Missing or invalid completed shard: {spec.relative_path}")
        array_path, metadata_path = _output_paths(spec, output_root)
        rows_by_source[spec.source] = rows_by_source.get(spec.source, 0) + spec.rows
        completed_shards.append(
            {
                "source": spec.source,
                "input_relative_path": spec.relative_path,
                "rows": spec.rows,
                "input_sha256": metadata["input_sha256"],
                "output_relative_path": metadata["output_relative_path"],
                "output_bytes": metadata["output_bytes"],
                "output_sha256": metadata["output_sha256"],
                "metadata_relative_path": metadata_path.relative_to(output_root).as_posix(),
            }
        )
        checksum_entries.append(
            (metadata["output_sha256"], array_path.relative_to(output_root).as_posix())
        )
        checksum_entries.append(
            (_sha256_file(metadata_path), metadata_path.relative_to(output_root).as_posix())
        )

    model_files = []
    for path in sorted(item for item in model_path.rglob("*") if item.is_file()):
        model_files.append(
            {
                "relative_path": path.relative_to(model_path).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not model_files:
        raise ValueError(f"Model directory contains no files: {model_path}")

    manifest = {
        "schema_version": "scad_embedding_collection_v1",
        **expected_config,
        "corpus_manifest": corpus_manifest_path.name,
        "corpus_manifest_sha256": _sha256_file(corpus_manifest_path),
        "model_path": model_path.name,
        "model_files": model_files,
        "total_rows": sum(spec.rows for spec in shards),
        "total_shards": len(shards),
        "rows_by_source": rows_by_source,
        "output_bytes": sum(item["output_bytes"] for item in completed_shards),
        "shards": completed_shards,
    }
    manifest_path = output_root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    checksum_entries.append((_sha256_file(manifest_path), "manifest.json"))
    checksum_text = "".join(
        f"{checksum}  {relative_path}\n"
        for checksum, relative_path in sorted(checksum_entries, key=lambda item: item[1])
    )
    _atomic_text(output_root / "checksums.sha256", checksum_text)
    return manifest


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed SCAD passage shards with Qwen3-Embedding-4B",
    )
    parser.add_argument("--passages-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--sources", default=",".join(SOURCE_ORDER))
    parser.add_argument("--dimension", type=int, default=2_560)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--buffer-rows", type=int, default=8_192)
    parser.add_argument("--max-batch-tokens", type=int, default=32_768)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--finalize-only", action="store_true")
    return parser


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )


def run_worker(args: argparse.Namespace) -> None:
    import torch

    context = resolve_worker_context(os.environ, torch.cuda.device_count())
    torch.cuda.set_device(context.local_rank)
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // context.world_size)))
    passages_root = args.passages_root.resolve()
    output_root = args.output_root.resolve()
    model_path = args.model_path.resolve()
    shards = discover_shards(passages_root, tuple(args.sources.split(",")))
    assigned = assign_shards(shards, context.rank, context.world_size)
    assigned_rows = sum(spec.rows for spec in assigned)
    logging.info(
        "rank=%d/%d device=%s assigned_shards=%d assigned_rows=%d",
        context.rank,
        context.world_size,
        context.device,
        len(assigned),
        assigned_rows,
    )
    if not assigned:
        return

    load_started = time.monotonic()
    encoder = QwenPassageEncoder.from_pretrained(
        model_path=model_path,
        device=context.device,
        dimension=args.dimension,
        max_length=args.max_length,
        max_batch_tokens=args.max_batch_tokens,
        max_batch_size=args.max_batch_size,
    )
    torch.cuda.synchronize(context.local_rank)
    logging.info(
        "rank=%d model_loaded_seconds=%.2f free_gpu_gib=%.2f",
        context.rank,
        time.monotonic() - load_started,
        torch.cuda.mem_get_info(context.local_rank)[0] / (1024**3),
    )

    worker_started = time.monotonic()
    processed_rows = 0
    for shard_number, spec in enumerate(assigned, start=1):
        array_path, metadata_path = _output_paths(spec, output_root)
        was_present = array_path.is_file() and metadata_path.is_file()
        shard_started = time.monotonic()
        logging.info(
            "rank=%d shard=%d/%d source=%s rows=%d input=%s action=%s",
            context.rank,
            shard_number,
            len(assigned),
            spec.source,
            spec.rows,
            spec.relative_path,
            "validate_resume" if was_present else "encode",
        )
        encode_shard(
            spec=spec,
            output_root=output_root,
            encoder=encoder,
            dimension=args.dimension,
            buffer_rows=args.buffer_rows,
        )
        torch.cuda.synchronize(context.local_rank)
        elapsed = time.monotonic() - shard_started
        processed_rows += spec.rows
        logging.info(
            "rank=%d shard_complete=%s rows=%d seconds=%.2f rows_per_second=%.1f progress=%d/%d peak_gpu_gib=%.2f",
            context.rank,
            spec.relative_path,
            spec.rows,
            elapsed,
            spec.rows / elapsed if elapsed else 0.0,
            processed_rows,
            assigned_rows,
            torch.cuda.max_memory_allocated(context.local_rank) / (1024**3),
        )
        torch.cuda.reset_peak_memory_stats(context.local_rank)
    worker_elapsed = time.monotonic() - worker_started
    logging.info(
        "rank=%d worker_complete rows=%d seconds=%.2f rows_per_second=%.1f",
        context.rank,
        processed_rows,
        worker_elapsed,
        processed_rows / worker_elapsed if worker_elapsed else 0.0,
    )


def run_finalizer(args: argparse.Namespace) -> None:
    passages_root = args.passages_root.resolve()
    output_root = args.output_root.resolve()
    model_path = args.model_path.resolve()
    corpus_manifest_path = (
        args.corpus_manifest.resolve()
        if args.corpus_manifest
        else passages_root.parent / "manifest.json"
    )
    expected_config = build_embedding_config(
        model_path=model_path,
        dimension=args.dimension,
        max_length=args.max_length,
        max_batch_tokens=args.max_batch_tokens,
        max_batch_size=args.max_batch_size,
    )
    shards = discover_shards(passages_root, tuple(args.sources.split(",")))
    logging.info("finalize_start shards=%d rows=%d", len(shards), sum(s.rows for s in shards))
    started = time.monotonic()
    manifest = finalize_outputs(
        shards=shards,
        output_root=output_root,
        corpus_manifest_path=corpus_manifest_path,
        model_path=model_path,
        expected_config=expected_config,
    )
    logging.info(
        "finalize_complete shards=%d rows=%d output_gib=%.2f seconds=%.2f",
        manifest["total_shards"],
        manifest["total_rows"],
        manifest["output_bytes"] / (1024**3),
        time.monotonic() - started,
    )


def main() -> None:
    args = _build_argument_parser().parse_args()
    _configure_logging()
    if args.finalize_only:
        run_finalizer(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
