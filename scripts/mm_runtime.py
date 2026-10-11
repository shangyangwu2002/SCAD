"""Runtime environment and preflight checks for multimodal training."""
import os
import json
import hashlib
import shutil
from pathlib import Path
from urllib.parse import urlsplit

SUPPORTED_PROCESSORS = ("Qwen3VLProcessor", "Qwen2_5_VLProcessor")

def validate_output_directory(output, resume_mode):
    output = Path(output)
    artifacts = any((output / name).exists() for name in ("metrics.jsonl", "checkpoints", "best_checkpoint.json"))
    marker = output / "mm_run.json"
    if artifacts and (not marker.is_file() or json.loads(marker.read_text()).get("schema") != "mm_scad_v1"):
        raise ValueError("Output directory belongs to another experiment; choose a separate multimodal OUTPUT_DIR")
    checkpoints = output / "checkpoints"
    has_checkpoints = checkpoints.is_dir() and any(checkpoints.iterdir())
    if resume_mode == "disable" and ((output / "metrics.jsonl").exists() or has_checkpoints or (output / "best_checkpoint.json").exists()):
        raise ValueError("Choose a new multimodal OUTPUT_DIR (or RESUME_MODE=auto); existing training metrics/checkpoints are protected")


def run_env(s, output, evaluation=False):
    env = dict(os.environ)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(name, None)
    env.update(s)
    env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
                "RAY_ADDRESS": "local",
                "RAY_TMPDIR": "/tmp/mmray-" + hashlib.sha256(str(Path(output).resolve()).encode()).hexdigest()[:8],
                "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
                "MM_IMAGE_SEARCH_CACHE": s["MM_IMAGE_SEARCH_CACHE"],
                "VERL_FILE_LOGGER_PATH": str(Path(output) / ("test_metrics.jsonl" if evaluation else "metrics.jsonl")),
                "VERL_FILE_LOGGER_APPEND": "1" if s["RESUME_MODE"] != "disable" and not evaluation else "0",
                "TENSORBOARD_DIR": str(Path(output) / "tensorboard")})
    return env


def check_hardware(s):
    import torch
    need = int(s["N_GPUS"])
    memories = [torch.cuda.get_device_properties(i).total_memory / 2**30 for i in range(torch.cuda.device_count())]
    if len(memories) != need or min(memories, default=0) < float(s["MIN_GPU_MEMORY_GIB"]):
        raise ValueError(f"Need {need} visible GPUs >= {s['MIN_GPU_MEMORY_GIB']} GiB each; found {memories}")
    free = shutil.disk_usage(s["OUTPUT_DIR"]).free / 2**30
    if free < float(s["MIN_FREE_DISK_GIB"]):
        raise ValueError(f"Insufficient checkpoint space: {free:.1f} GiB free; need {s['MIN_FREE_DISK_GIB']}")
    if not (os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise ValueError("Load JUDGE_API_KEY from JUDGE_ENV_FILE before training")

def validate_endpoints(settings):
    parsed = urlsplit(settings["LOCAL_SEARCH_URL"])
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.port != int(settings["SEARCH_PORT"]):
        raise ValueError("LOCAL_SEARCH_URL must point to the dedicated local SEARCH_PORT")


def validate_models(student_path, teacher_path=None, require_weights=True):
    from transformers import AutoProcessor, AutoTokenizer
    student_path = Path(student_path)
    student = AutoTokenizer.from_pretrained(student_path, local_files_only=True)
    processor = AutoProcessor.from_pretrained(student_path, local_files_only=True)
    if type(processor).__name__ not in SUPPORTED_PROCESSORS:
        raise ValueError(f"This experiment requires a Qwen3-VL or Qwen2.5-VL model, got {type(processor).__name__}")
    paths = [student_path]
    if teacher_path:
        paths.append(Path(teacher_path))
        teacher = AutoTokenizer.from_pretrained(teacher_path, local_files_only=True)
        teacher_processor = AutoProcessor.from_pretrained(teacher_path, local_files_only=True)
        if student.get_vocab() != teacher.get_vocab() or student.eos_token_id != teacher.eos_token_id:
            raise ValueError("SCAD student/teacher vocabulary and EOS must be identical")
        if processor.image_processor.to_dict() != teacher_processor.image_processor.to_dict():
            raise ValueError("SCAD student/teacher visual preprocessing must match")
        for name in ("image_token_id", "video_token_id", "vision_start_token_id"):
            if getattr(processor, name, None) != getattr(teacher_processor, name, None):
                raise ValueError(f"SCAD student/teacher {name} must match")
    if require_weights:
        from safetensors import safe_open
        for path in paths:
            index_path = path / "model.safetensors.index.json"
            files = sorted(set(json.loads(index_path.read_text())["weight_map"].values())) if index_path.exists() else ["model.safetensors"]
            for name in files:
                if not (path / name).is_file():
                    raise ValueError(f"Model download incomplete: {path / name}")
                with safe_open(path / name, framework="pt", device="cpu") as reader:
                    if not list(reader.keys()):
                        raise ValueError(f"Empty model shard: {path / name}")
    digest = hashlib.sha256(json.dumps(student.get_vocab(), sort_keys=True).encode()).hexdigest()
    return {"student": str(student_path), "teacher": str(teacher_path), "vocabulary_size": len(student),
            "vocabulary_sha256": digest, "eos_token_id": student.eos_token_id,
            "image_processor": type(processor.image_processor).__name__, "weights_checked": require_weights}
