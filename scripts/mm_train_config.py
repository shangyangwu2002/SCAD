"""Build the dedicated multimodal experiment's VERL command without shell eval."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def settings_from_env():
    cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    package = Path(os.environ.get("SCAD_DATA_ROOT", ROOT.parent / "SCAD_DATA")) / "multimodal"
    runtime = Path(os.environ.get("SCAD_RUNTIME_ROOT", ROOT / "runtime")) / "multimodal"
    models = Path(os.environ.get("MODELS_ROOT", ROOT / "models"))
    defaults = {
        "PYTHON_BIN": sys.executable,
        "MM_DATASET_ROOT": str(package / "dataset"),
        "MM_CORPUS_ROOT": str(package / "corpus"),
        "MM_IMAGE_SEARCH_CACHE": str(package / "corpus/image_search_results.json"),
        "DATA_DIR": str(runtime / "dataset"),
        "OUTPUT_DIR": str(ROOT / "outputs/scad_mm_2b_seed42"),
        "MODEL_PATH": str(models / "Qwen/Qwen3-VL-2B-Instruct"),
        "TEACHER_MODEL_PATH": str(models / "Qwen/Qwen3-VL-4B-Instruct"),
        "SEARCH_PORT": "8010", "LOCAL_SEARCH_URL": "http://127.0.0.1:8010",
        "PASSAGE_ROOT": str(package / "corpus/passages"),
        "EMBEDDING_ROOT": str(runtime / "embeddings"),
        "ANN_ROOT": str(runtime / "indices"),
        "EMBEDDING_MODEL_PATH": str(models / "Qwen/Qwen3-Embedding-4B"),
        "CUDA_VISIBLE_DEVICES": "0,1", "N_GPUS": "2", "SEED": "42",
        "CPU_BUDGET": str(cpu_count), "FAISS_THREADS": str(min(100, cpu_count)),
        "ANN_CANDIDATES": "32768", "ANN_NPROBE": "512",
        "AGENT_LOOP_WORKERS": "32", "GPU_MEMORY_UTILIZATION": "0.5",
        "PROMPT_LENGTH": "8192", "RESPONSE_LENGTH": "32768", "BRANCH_LEN": "16384",
        "MAX_TURN": "80", "MAX_SESSION": "6", "MM_MAX_IMAGE_PIXELS": "524288", "MM_LIMIT_IMAGES": "5",
        "TRAIN_BATCH_SIZE": "8", "ROLLOUT_N": "8", "TOTAL_STEPS": "60", "VAL_FREQ": "10", "SAVE_FREQ": "10",
        "VAL_BATCH_SIZE": "85", "VAL_MAX_SAMPLES": "-1", "RESUME_MODE": "disable", "RESUME_FROM_PATH": "",
        "JUDGE_ENV_FILE": ".env",
        "MM_OPEN_PAGE_PROXY": "",  # optional forward proxy for test-time open_page fetches; serpapi.com stays direct
        "ROLLOUT_TP": "1",  # vLLM tensor parallel size
        "ACTOR_PARAM_OFFLOAD": "False",
        "ACTOR_MODEL_DTYPE": "",  # Empty uses the framework default.
        "MM_IMAGE_URL_MAP": str(runtime / "dataset/image_url_map.json"),
        "MIN_FREE_DISK_GIB": "115", "MIN_GPU_MEMORY_GIB": "80",
        "MM_RERANK_WORKERS": "4", "MM_FLAT_CANDIDATE_K": "4096", "VAL_BEFORE_TRAIN": "False",
        # per-batch source mixture like the text run (browsecomp 4 / webwalker 2 / others 2):
        # Primary vdr_bench 2, secondary mmsearch_plus 3, remaining 3 rotate over other sets
        "MM_OPEN_PAGE_TOKENS": "2500", "MM_NODE_MERGE_EMBED_BATCH": "16",
        "MM_SAMPLER": "stratified", "MM_SAMPLER_PRIMARY": "vdr_bench", "MM_SAMPLER_PRIMARY_PER_BATCH": "2",
        "MM_SAMPLER_SECONDARY": "mmsearch_plus", "MM_SAMPLER_SECONDARY_PER_BATCH": "3",
    }
    settings = {k: os.environ.get(k, v) for k, v in defaults.items()}
    if "SEARCH_PORT" in os.environ and "LOCAL_SEARCH_URL" not in os.environ:
        settings["LOCAL_SEARCH_URL"] = f"http://127.0.0.1:{settings['SEARCH_PORT']}"
    return settings


def train_command(s, *, split="val", checkpoint=None, output_dir=None, dry_run=False, model_path=None):
    """Build SCAD training or unified-test evaluation arguments."""
    if split not in {"val", "test"}:
        raise ValueError("Expected val or test")
    evaluation = split == "test"
    model_only = evaluation and model_path is not None
    if model_only and checkpoint is not None:
        raise ValueError("Use either model_path or checkpoint for evaluation")
    out = Path(output_dir or s["OUTPUT_DIR"])
    number = lambda key: int(s[key])
    length = number("PROMPT_LENGTH") + number("RESPONSE_LENGTH")
    args = {
        "algorithm.adv_estimator": "scad",
        "algorithm.norm_adv_by_std_in_grpo": True,
        "algorithm.scad.enable": not evaluation,
        "algorithm.scad.distillation_coef": 1.0, "algorithm.scad.rl_coef": 1.0, "algorithm.scad.distillation_clip": 5.0,
        "algorithm.scad.node_merge.enable": not evaluation,
        "algorithm.scad.node_merge.semantic": True,
        "algorithm.scad.node_merge.subtask_similarity_threshold": 0.83,
        "algorithm.scad.node_merge.entity_similarity_threshold": 0.90,
        "algorithm.scad.node_merge.embedding_url": s["LOCAL_SEARCH_URL"] + "/embed",
        "algorithm.scad.node_merge.embedding_timeout_s": 300,
        "algorithm.scad.node_merge.embedding_batch_size": number("MM_NODE_MERGE_EMBED_BATCH"),
        "algorithm.scad.node_merge.audit_dir": str(out / "node_trees"),
        "algorithm.scad.tree_credit.enable": not evaluation,
        "algorithm.scad.tree_credit.episode_coef": 1.0, "algorithm.scad.tree_credit.planner_coef": 1.0,
        "algorithm.scad.tree_credit.prior_count": 2.0, "algorithm.scad.tree_credit.normalize": True,
        "algorithm.scad.tree_credit.clip": 5.0,
        "actor_rollout_ref.model.path": model_path or s["MODEL_PATH"], "actor_rollout_ref.model.trust_remote_code": True,
        "actor_rollout_ref.rollout.agent.default_agent_loop": "mm_scad_agent",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(ROOT / "configs/mm_scad_agent_loop.yaml"),
        "actor_rollout_ref.rollout.agent.num_workers": number("AGENT_LOOP_WORKERS"),
        "+actor_rollout_ref.rollout.agent.sticky_sessions": True,
        "+actor_rollout_ref.rollout.agent.hash_sticky_sessions": True,
        "actor_rollout_ref.rollout.name": "vllm", "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "actor_rollout_ref.rollout.prompt_length": number("PROMPT_LENGTH"),
        "actor_rollout_ref.rollout.response_length": number("RESPONSE_LENGTH"),
        "actor_rollout_ref.rollout.max_model_len": length,
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": length,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": number("ROLLOUT_TP"),
        "actor_rollout_ref.rollout.n": number("ROLLOUT_N"),
        "actor_rollout_ref.rollout.gpu_memory_utilization": float(s["GPU_MEMORY_UTILIZATION"]),
        "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 65536,
        "actor_rollout_ref.rollout.max_num_seqs": 32,
        "+actor_rollout_ref.rollout.limit_images": number("MM_LIMIT_IMAGES"),
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs.min_pixels": 65536,
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs.max_pixels": number("MM_MAX_IMAGE_PIXELS"),
        "actor_rollout_ref.rollout.val_kwargs.temperature": 0,
        "actor_rollout_ref.rollout.val_kwargs.top_p": 1.0,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": False, "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "data.train_files": str(Path(s["DATA_DIR"]) / "train.parquet"),
        "data.val_files": str(Path(s["DATA_DIR"]) / f"{split}.parquet"),
        "data.custom_cls.path": str(ROOT / "scripts/train_mm_scad.py"), "data.custom_cls.name": "MMRLHFDataset",
        "data.return_multi_modal_inputs": False, "data.return_raw_chat": True,
        "data.train_batch_size": number("TRAIN_BATCH_SIZE"), "data.val_batch_size": number("VAL_BATCH_SIZE"),
        "data.val_max_samples": number("VAL_MAX_SAMPLES"), "data.max_prompt_length": number("PROMPT_LENGTH"),
        "data.max_response_length": number("RESPONSE_LENGTH"), "data.filter_overlong_prompts": False,
        "data.dataloader_num_workers": 0, "data.shuffle": True, "data.seed": number("SEED"),
        "+data.apply_chat_template_kwargs.enable_thinking": False,
        "actor_rollout_ref.actor.ppo_mini_batch_size": number("TRAIN_BATCH_SIZE") * number("ROLLOUT_N"),
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.fsdp_config.param_offload": str(s.get("ACTOR_PARAM_OFFLOAD", "False")).lower() == "true",
        **({"actor_rollout_ref.actor.fsdp_config.model_dtype": s["ACTOR_MODEL_DTYPE"]} if s.get("ACTOR_MODEL_DTYPE") else {}),
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.actor.fsdp_config.seed": number("SEED"),
        "actor_rollout_ref.actor.checkpoint.save_contents": ["model", "extra"] if evaluation else ["model", "optimizer", "extra"],
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": length,
        "actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu": length,
        "trainer.val_before_train": True if evaluation else s.get("VAL_BEFORE_TRAIN", "True").lower() == "true", "trainer.val_only": evaluation,
        "+trainer.save_initial_checkpoint": not evaluation,
        "trainer.n_gpus_per_node": number("N_GPUS"), "trainer.nnodes": 1,
        "trainer.total_training_steps": number("TOTAL_STEPS"), "trainer.test_freq": number("VAL_FREQ"),
        "trainer.save_freq": number("SAVE_FREQ"), "trainer.max_actor_ckpt_to_keep": None,
        "+trainer.track_best_checkpoint": not evaluation, "+trainer.keep_best_and_latest": not evaluation,
        "trainer.resume_mode": "disable" if model_only else ("resume_path" if evaluation else s["RESUME_MODE"]),
        "trainer.default_local_dir": str(out / "checkpoints"),
        "trainer.rollout_data_dir": str(out / "rollouts"),
        "trainer.validation_data_dir": str(out / ("test_predictions" if evaluation else "validation")),
        "trainer.logger": ["console", "file", "tensorboard"], "trainer.project_name": "SCAD",
        "trainer.experiment_name": out.name,
    }
    plugin = {
        "enable_thinking": False,
        "must_search": False,
        "turn_max_new_tokens": 2048,
        "max_session": number("MAX_SESSION"), "val_max_session": number("MAX_SESSION"),
        "max_turn": number("MAX_TURN"), "val_max_turn": number("MAX_TURN"),
        "val_response_length": number("RESPONSE_LENGTH"), "session_timeout": 3600,
        "branch_len": number("BRANCH_LEN"), "branch_return_reserve_tokens": 4096,
        "branch_min_observation_tokens": 1024, "search_snippet_tokens": 512, "seen_search_snippet_tokens": 128,
        "open_page_tokens": number("MM_OPEN_PAGE_TOKENS"), "tool_observation_tokens": 6144, "mm_max_image_pixels": number("MM_MAX_IMAGE_PIXELS"),
    }
    args.update({"+actor_rollout_ref.rollout.plugin." + k: v for k, v in plugin.items()})
    if not evaluation and s.get("MM_SAMPLER", "random") == "stratified":
        args.update({"data.sampler.class_path": str(ROOT / "samplers/stratified_source.py"),
                     "data.sampler.class_name": "StratifiedSourceBatchSampler",
                     "+data.sampler.seed": number("SEED"), "+data.sampler.total_steps": number("TOTAL_STEPS"),
                     "+data.sampler.batch_size": number("TRAIN_BATCH_SIZE"),
                     "+data.sampler.primary_source": s["MM_SAMPLER_PRIMARY"], "+data.sampler.secondary_source": s["MM_SAMPLER_SECONDARY"],
                     "+data.sampler.primary_per_batch": number("MM_SAMPLER_PRIMARY_PER_BATCH"),
                     "+data.sampler.secondary_per_batch": number("MM_SAMPLER_SECONDARY_PER_BATCH"),
                     "+data.sampler.audit_path": str(out / "stratified_source_audit.json")})
    if not evaluation:
        args.update({"+actor_rollout_ref.ref.model.path": s["TEACHER_MODEL_PATH"],
                     "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
                     "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": length,
                     "actor_rollout_ref.ref.fsdp_config.param_offload": True,
                     "actor_rollout_ref.ref.fsdp_config.model_dtype": "bfloat16"})
    resume = None if model_only else (checkpoint or s["RESUME_FROM_PATH"])
    if evaluation and not model_only and not resume:
        raise ValueError("Evaluation requires a selected checkpoint")
    if resume:
        args["trainer.resume_from_path"] = str(resume)
    command = [s["PYTHON_BIN"], "-m", "scripts.train_scad"]
    command.extend(f"{key}={json.dumps(value)}" for key, value in args.items())
    if dry_run:
        command.extend(["--cfg", "job", "--resolve"])
    return command
