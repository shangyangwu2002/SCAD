#!/usr/bin/env python3
"""Select the best validation checkpoint and summarize held-out SCAD results."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import re
from pathlib import Path


def read_metrics(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if "step" in record and isinstance(record.get("data"), dict):
                records.append(record)
    if not records:
        raise ValueError(f"No metrics found in {path}")
    return records


def dataset_scores(data: dict) -> dict[str, float]:
    scores = {}
    pattern = re.compile(r"^val-core/([^/]+)/reward/mean@1$")
    for key, value in data.items():
        match = pattern.match(key)
        if match:
            scores[match.group(1)] = float(value)
    return scores


def validation_score(record: dict) -> float | None:
    data = record["data"]
    # Match RayPPOTrainer._validation_selection and checkpoint pruning:
    # Unequal source sizes mean val/reward (micro) can disagree with the macro score.
    scores = {name: score for name, score in dataset_scores(data).items() if name != "unknown"}
    if scores:
        return sum(scores.values()) / len(scores)
    if "val/reward" in data:
        return float(data["val/reward"])
    return None


def select_checkpoint(metrics_path: Path, checkpoint_root: Path, output_path: Path, counts=None) -> dict:
    metrics_path, checkpoint_root, output_path = map(Path, (metrics_path, checkpoint_root, output_path))
    candidates = []
    for record in read_metrics(metrics_path):
        if counts is not None and set(dataset_scores(record["data"])) != set(counts):
            continue
        score = validation_score(record)
        checkpoint = checkpoint_root / f"global_step_{record['step']}"
        if score is not None and checkpoint.is_dir():
            candidates.append((score, int(record["step"]), checkpoint, dataset_scores(record["data"])))
    if not candidates:
        raise ValueError("No validation metric has a matching saved checkpoint")
    score, step, checkpoint, per_dataset = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
    result = {
        "selection_rule": "highest validation dataset macro; earliest step breaks ties",
        "metrics": str(metrics_path.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "step": step,
        "validation_score": score,
        "validation_by_dataset": per_dataset,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result["checkpoint"])
    return result


def split_counts(data_file):
    import pyarrow.parquet as pq
    return dict(Counter(pq.read_table(data_file, columns=["data_source"])["data_source"].to_pylist()))


def _write_summary(output_dir, split, step, counts, scores, source):
    total = sum(counts.values())
    result = {"split": split, "step": int(step), "metric": "final_answer_llm_judge", "source": source,
              "counts": counts, "scores": scores, "total": total,
              "macro_accuracy": sum(scores.values()) / len(scores),
              "micro_accuracy": sum(counts[k] * scores[k] for k in counts) / total}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_results.json").write_text(json.dumps(result, indent=2) + "\n")
    rows = [(key, counts[key], 100 * scores[key]) for key in sorted(counts)]
    with (output_dir / "test_results.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "n", "accuracy_percent"])
        writer.writerows(rows)
    lines = ["| Dataset | N | Accuracy (%) |", "|---|---:|---:|"]
    lines.extend(f"| {key} | {count} | {score:.2f} |" for key, count, score in rows)
    lines.extend([f"| Macro | — | {100 * result['macro_accuracy']:.2f} |",
                  f"| Micro | {total} | {100 * result['micro_accuracy']:.2f} |"])
    (output_dir / "test_results.md").write_text("\n".join(lines) + "\n")
    return result


def summarize(metrics_path, output_dir, counts=None, split="test"):
    if counts is None:
        selection = json.loads((Path(__file__).resolve().parents[1] / "configs/data_selection.json").read_text())
        counts = dict(Counter(row["dataset"] for row in selection["samples"]
                              if row["modality"] == "text" and row["split"] == split))
    scored = [(r, dataset_scores(r["data"])) for r in read_metrics(Path(metrics_path))]
    scored = [(r, s) for r, s in scored if s]
    if not scored:
        raise ValueError("No held-out dataset metrics found")
    record, scores = scored[-1]
    if set(scores) != set(counts):
        raise ValueError(f"Held-out datasets differ: expected {sorted(counts)}, got {sorted(scores)}")
    return _write_summary(output_dir, split, record["step"], counts, scores, "test_metrics")


def summarize_from_predictions(predictions_dir, output_dir, counts, split):
    """Rebuild the summary from test_predictions/*.jsonl when the trainer crashed after dumping predictions.

    Every row carries the judged `score` and its `data_source`; counts must match the split exactly."""
    rows = [json.loads(line) for path in sorted(Path(predictions_dir).glob("*.jsonl"))
            for line in path.open(encoding="utf-8") if line.strip()]
    if not rows:
        raise ValueError(f"No predictions found under {predictions_dir}")
    per_dataset = {}
    for row in rows:
        per_dataset.setdefault(str(row["data_source"]), []).append(float(row["score"]))
    found = {key: len(values) for key, values in per_dataset.items()}
    if found != dict(counts):
        raise ValueError(f"Prediction counts differ: expected {dict(counts)}, got {found}")
    scores = {key: sum(values) / len(values) for key, values in per_dataset.items()}
    return _write_summary(output_dir, split, rows[0].get("step", 0), counts, scores, "test_predictions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["select", "summarize", "summarize-predictions"])
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--predictions", type=Path, help="test_predictions directory (summarize-predictions)")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint-root", type=Path)
    args = parser.parse_args()
    counts = split_counts(args.data)
    if args.command == "select":
        result = select_checkpoint(args.metrics, args.checkpoint_root, args.output_dir / "best_checkpoint.json", counts)
    elif args.command == "summarize-predictions":
        result = summarize_from_predictions(args.predictions, args.output_dir, counts, args.split)
    else:
        result = summarize(args.metrics, args.output_dir, counts, args.split)
    print(json.dumps(result, indent=2))
