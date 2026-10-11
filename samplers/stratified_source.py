"""Deterministic source-stratified sampler for SCAD training.

Each optimizer step sees the same source mixture.  For the default eight-prompt
batch this is four BrowseComp prompts, two WebWalker prompts, and two prompts
rotated evenly across the remaining QA datasets.  Rows are independently
shuffled inside every source pool and reshuffled only when that pool is
exhausted.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterator, Sized
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from verl.experimental.dataset.sampler import AbstractSampler


class _StatefulSequenceIterator(Iterator[int]):
    """Iterator whose exact position is saved by StatefulDataLoader."""

    def __init__(self, sequence: tuple[int, ...]):
        self.sequence = sequence
        self.yielded = 0

    def __iter__(self) -> "_StatefulSequenceIterator":
        return self

    def __next__(self) -> int:
        if self.yielded >= len(self.sequence):
            raise StopIteration
        value = self.sequence[self.yielded]
        self.yielded += 1
        return value

    def state_dict(self) -> dict[str, int]:
        return {"yielded": self.yielded}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        yielded = int(state_dict["yielded"])
        if yielded < 0 or yielded > len(self.sequence):
            raise ValueError(f"Invalid restored sampler position: {yielded}")
        self.yielded = yielded


def _row_metadata(row: dict[str, Any], index: int) -> dict[str, str]:
    extra_info = row.get("extra_info") or {}
    source = str(row.get("data_source") or extra_info.get("dataset") or "unknown")
    stable_id = str(
        row.get("stable_hash")
        or extra_info.get("stable_hash")
        or f"{source}:{row.get('instance_id', extra_info.get('instance_id', index))}"
    )
    return {
        "source": source,
        "instance_id": str(row.get("instance_id", extra_info.get("instance_id", index))),
        "stable_id": stable_id,
    }


class _PoolStream:
    """Draw shuffled source rows without replacement, then start a new cycle."""

    def __init__(self, indices: list[int], rng: np.random.Generator):
        if not indices:
            raise ValueError("Cannot sample from an empty source pool")
        self.indices = tuple(indices)
        self.rng = rng
        self.order: list[int] = []
        self.position = 0
        self.cycles = 0

    def draw(self) -> int:
        if self.position >= len(self.order):
            self.order = self.rng.permutation(self.indices).tolist()
            self.position = 0
            self.cycles += 1
        value = self.order[self.position]
        self.position += 1
        return value


class StratifiedSourceBatchSampler(AbstractSampler):
    """Build fixed-ratio batches while preserving the dataset-level mixture."""

    def __init__(self, data_source: Sized, data_config: DictConfig):
        self.data_source = data_source
        sampler_config = data_config.sampler
        self.seed = int(sampler_config.get("seed", data_config.get("seed", 42)))
        self.total_steps = int(sampler_config.get("total_steps", 0))
        self.batch_size = int(
            sampler_config.get(
                "batch_size",
                data_config.get("gen_batch_size", data_config.get("train_batch_size", 0)),
            )
        )
        self.primary_source = str(sampler_config.get("primary_source", "browsecomp_plus"))
        self.secondary_source = str(sampler_config.get("secondary_source", "webwalker_qa"))
        self.primary_per_batch = int(sampler_config.get("primary_per_batch", 4))
        self.secondary_per_batch = int(sampler_config.get("secondary_per_batch", 2))
        self.other_per_batch = self.batch_size - self.primary_per_batch - self.secondary_per_batch

        if self.total_steps <= 0:
            raise ValueError("data.sampler.total_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("data.sampler.batch_size must be positive")
        if self.primary_per_batch <= 0 or self.secondary_per_batch <= 0:
            raise ValueError("Primary and secondary per-batch quotas must be positive")
        if self.other_per_batch <= 0:
            raise ValueError(
                "Source quotas must leave at least one batch position for other datasets: "
                f"batch_size={self.batch_size}, primary={self.primary_per_batch}, "
                f"secondary={self.secondary_per_batch}"
            )

        dataframe = getattr(data_source, "dataframe", None)
        if dataframe is None:
            raise TypeError("StratifiedSourceBatchSampler requires RLHFDataset.dataframe")
        self.metadata = tuple(_row_metadata(dataframe[index], index) for index in range(len(data_source)))
        pools: dict[str, list[int]] = {}
        for index, metadata in enumerate(self.metadata):
            pools.setdefault(metadata["source"], []).append(index)

        for required_source in (self.primary_source, self.secondary_source):
            if required_source not in pools:
                raise ValueError(f"Required source pool is missing: {required_source}")
        self.other_sources = tuple(
            sorted(source for source in pools if source not in {self.primary_source, self.secondary_source})
        )
        if not self.other_sources:
            raise ValueError("At least one non-primary/non-secondary source pool is required")

        sequence, audit = self._build_sequence(pools)
        self.sequence = tuple(sequence)
        self.audit = audit
        self._write_or_verify_audit(sampler_config.get("audit_path"))

        print(
            "[STRATIFIED SOURCE SAMPLER] "
            f"steps={self.total_steps} batch_size={self.batch_size} seed={self.seed} "
            f"per_batch={self.primary_source}:{self.primary_per_batch},"
            f"{self.secondary_source}:{self.secondary_per_batch},other:{self.other_per_batch} "
            f"fingerprint={self.audit['sequence_sha256']}"
        )
        print(f"[STRATIFIED SOURCE SAMPLER] totals={self.audit['total_source_counts']}")

    def _build_sequence(self, pools: dict[str, list[int]]) -> tuple[list[int], dict[str, Any]]:
        rng = np.random.default_rng(self.seed)
        streams = {source: _PoolStream(indices, rng) for source, indices in pools.items()}

        # Repeating a seeded permutation of the auxiliary source names gives
        # every source one slot per cycle.  Since adjacent names differ even at
        # the cycle boundary, a two-slot auxiliary allocation never collapses
        # onto a single dataset.
        other_schedule = rng.permutation(self.other_sources).tolist()
        other_position = 0
        sequence: list[int] = []
        records: list[dict[str, Any]] = []

        for step in range(1, self.total_steps + 1):
            batch: list[int] = []
            batch.extend(streams[self.primary_source].draw() for _ in range(self.primary_per_batch))
            batch.extend(streams[self.secondary_source].draw() for _ in range(self.secondary_per_batch))
            for _ in range(self.other_per_batch):
                source = other_schedule[other_position % len(other_schedule)]
                other_position += 1
                batch.append(streams[source].draw())

            # Source quotas are fixed, but batch slots should not expose a
            # constant source order to rollout workers.
            batch = rng.permutation(batch).tolist()
            start = len(sequence)
            sequence.extend(batch)
            for slot, index in enumerate(batch):
                records.append(
                    {
                        "step": step,
                        "batch_slot": slot,
                        "dataset_index": index,
                        **self.metadata[index],
                    }
                )
            counts = Counter(self.metadata[index]["source"] for index in sequence[start:])
            if counts[self.primary_source] != self.primary_per_batch:
                raise RuntimeError(f"Primary source quota failed at step {step}: {counts}")
            if counts[self.secondary_source] != self.secondary_per_batch:
                raise RuntimeError(f"Secondary source quota failed at step {step}: {counts}")
            if sum(counts[source] for source in self.other_sources) != self.other_per_batch:
                raise RuntimeError(f"Other source quota failed at step {step}: {counts}")

        expected_length = self.total_steps * self.batch_size
        if len(sequence) != expected_length:
            raise RuntimeError(
                f"Stratified sampler generated {len(sequence)} rows, expected {expected_length}"
            )
        fingerprint_payload = "\n".join(record["stable_id"] for record in records).encode("utf-8")
        step_source_counts = []
        for step in range(1, self.total_steps + 1):
            start = (step - 1) * self.batch_size
            end = step * self.batch_size
            step_source_counts.append(
                {
                    "step": step,
                    "source_counts": dict(
                        sorted(Counter(record["source"] for record in records[start:end]).items())
                    ),
                }
            )
        audit = {
            "schema_version": "scad_stratified_source_batch_v1",
            "seed": self.seed,
            "total_steps": self.total_steps,
            "batch_size": self.batch_size,
            "primary_source": self.primary_source,
            "secondary_source": self.secondary_source,
            "other_sources": list(self.other_sources),
            "per_batch": {
                self.primary_source: self.primary_per_batch,
                self.secondary_source: self.secondary_per_batch,
                "other": self.other_per_batch,
            },
            "pool_counts": {source: len(indices) for source, indices in sorted(pools.items())},
            "total_source_counts": dict(
                sorted(Counter(record["source"] for record in records).items())
            ),
            "step_source_counts": step_source_counts,
            "sequence_sha256": hashlib.sha256(fingerprint_payload).hexdigest(),
            "sequence": records,
        }
        return sequence, audit

    def _write_or_verify_audit(self, audit_path: Any) -> None:
        if not audit_path:
            return
        path = Path(os.path.expanduser(str(audit_path)))
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("sequence_sha256") != self.audit["sequence_sha256"]:
                raise RuntimeError(f"Existing sampler audit does not match this run: {path}")
            return
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def __iter__(self) -> _StatefulSequenceIterator:
        return _StatefulSequenceIterator(self.sequence)

    def __len__(self) -> int:
        return len(self.sequence)
