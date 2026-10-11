#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
# Shared internal launcher; public entry points load the local environment.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
MODEL_PATH="${MODEL_PATH:-models/Qwen/Qwen3-4B}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-models/Qwen/Qwen3-4B-Instruct-2507}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/runtime/text/dataset}"
SEED="${SEED:-42}"
STRATIFIED_SOURCE_SAMPLING="${STRATIFIED_SOURCE_SAMPLING:-True}"
case "${STRATIFIED_SOURCE_SAMPLING,,}" in
  true|false) ;;
  *)
    echo "STRATIFIED_SOURCE_SAMPLING must be True or False, got: ${STRATIFIED_SOURCE_SAMPLING}" >&2
    exit 1
    ;;
esac
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/scad_text_4b_seed${SEED}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-scad_text_4b_seed${SEED}}"
AGENT_LOOP_CONFIG="${AGENT_LOOP_CONFIG:-${PROJECT_ROOT}/configs/scad_agent_loop.yaml}"
LOCAL_SEARCH_URL="${LOCAL_SEARCH_URL:-http://127.0.0.1:8000}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/val.parquet}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-130}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
VAL_ONLY="${VAL_ONLY:-False}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
METRICS_FILE="${METRICS_FILE:-${OUTPUT_DIR}/metrics.jsonl}"
VALIDATION_DIR="${VALIDATION_DIR:-${OUTPUT_DIR}/validation}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${OUTPUT_DIR}/rollouts}"
N_GPUS="${N_GPUS:-4}"
AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-16}"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-scad}"
SCAD_ENABLE="${SCAD_ENABLE:-True}"
SCAD_DISTILLATION_COEF="${SCAD_DISTILLATION_COEF:-1.0}"
SCAD_RL_COEF="${SCAD_RL_COEF:-1.0}"
SCAD_DISTILLATION_CLIP="${SCAD_DISTILLATION_CLIP:-5.0}"
SCAD_NODE_MERGE="${SCAD_NODE_MERGE:-${SCAD_ENABLE}}"
SCAD_NODE_SEMANTIC="${SCAD_NODE_SEMANTIC:-True}"
SCAD_SUBTASK_THRESHOLD="${SCAD_SUBTASK_THRESHOLD:-0.83}"
SCAD_ENTITY_THRESHOLD="${SCAD_ENTITY_THRESHOLD:-0.90}"
SCAD_EMBED_TIMEOUT="${SCAD_EMBED_TIMEOUT:-300}"
SCAD_EMBED_BATCH_SIZE="${SCAD_EMBED_BATCH_SIZE:-64}"
SCAD_NODE_AUDIT_DIR="${SCAD_NODE_AUDIT_DIR:-${OUTPUT_DIR}/node_trees}"
SCAD_TREE_CREDIT="${SCAD_TREE_CREDIT:-${SCAD_NODE_MERGE}}"
SCAD_TREE_EPISODE_COEF="${SCAD_TREE_EPISODE_COEF:-1.0}"
SCAD_TREE_PLANNER_COEF="${SCAD_TREE_PLANNER_COEF:-1.0}"
SCAD_TREE_PRIOR_COUNT="${SCAD_TREE_PRIOR_COUNT:-2.0}"
SCAD_TREE_NORMALIZE="${SCAD_TREE_NORMALIZE:-True}"
SCAD_TREE_CLIP="${SCAD_TREE_CLIP:-5.0}"

PROMPT_LENGTH="${PROMPT_LENGTH:-8192}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-32768}"
BRANCH_LEN="${BRANCH_LEN:-16384}"
MAX_LENGTH=$((PROMPT_LENGTH + RESPONSE_LENGTH))
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TOTAL_STEPS="${TOTAL_STEPS:-60}"
VAL_FREQ="${VAL_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-10}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-null}"
TRACK_BEST_CHECKPOINT="${TRACK_BEST_CHECKPOINT:-True}"
KEEP_BEST_AND_LATEST="${KEEP_BEST_AND_LATEST:-True}"
MAX_TURN="${MAX_TURN:-80}"
MAX_SESSION="${MAX_SESSION:-6}"
SEARCH_SNIPPET_TOKENS="${SEARCH_SNIPPET_TOKENS:-512}"
SEEN_SEARCH_SNIPPET_TOKENS="${SEEN_SEARCH_SNIPPET_TOKENS:-128}"
OPEN_PAGE_TOKENS="${OPEN_PAGE_TOKENS:-2500}"
TOOL_OBSERVATION_TOKENS="${TOOL_OBSERVATION_TOKENS:-6144}"
BRANCH_RETURN_RESERVE_TOKENS="${BRANCH_RETURN_RESERVE_TOKENS:-4096}"
BRANCH_MIN_OBSERVATION_TOKENS="${BRANCH_MIN_OBSERVATION_TOKENS:-1024}"
RESUME_MODE="${RESUME_MODE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.72}"
PARAM_OFFLOAD="${PARAM_OFFLOAD:-False}"
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-False}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"
CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-[model,optimizer,extra]}"

case "${SCAD_ENABLE,,}" in
  true|false) ;;
  *)
    echo "SCAD_ENABLE must be True or False; got: ${SCAD_ENABLE}" >&2
    exit 1
    ;;
esac
case "${SCAD_NODE_MERGE,,}:${SCAD_NODE_SEMANTIC,,}" in
  true:true|true:false|false:true|false:false) ;;
  *)
    echo "SCAD_NODE_MERGE and SCAD_NODE_SEMANTIC must be True or False" >&2
    exit 1
    ;;
esac
case "${SCAD_TREE_CREDIT,,}:${SCAD_TREE_NORMALIZE,,}" in
  true:true|true:false|false:true|false:false) ;;
  *)
    echo "SCAD_TREE_CREDIT and SCAD_TREE_NORMALIZE must be True or False" >&2
    exit 1
    ;;
esac
if [[ "${SCAD_TREE_CREDIT,,}" == "true" && "${SCAD_NODE_MERGE,,}" != "true" ]]; then
  echo "SCAD_TREE_CREDIT=True requires SCAD_NODE_MERGE=True" >&2
  exit 1
fi
case "${TRACK_BEST_CHECKPOINT,,}:${KEEP_BEST_AND_LATEST,,}" in
  true:true|true:false|false:true|false:false) ;;
  *)
    echo "TRACK_BEST_CHECKPOINT and KEEP_BEST_AND_LATEST must be True or False" >&2
    exit 1
    ;;
esac

if [[ "${DRY_RUN:-0}" != "1" ]]; then
for required in "${PYTHON_BIN}" "${MODEL_PATH}/config.json" "${DATA_DIR}/train.parquet" "${VAL_FILE}" "${AGENT_LOOP_CONFIG}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 1
  fi
done
if [[ "${SCAD_ENABLE,,}" == "true" && ! -s "${TEACHER_MODEL_PATH}/config.json" ]]; then
  echo "Missing SCAD teacher model: ${TEACHER_MODEL_PATH}/config.json" >&2
  exit 1
fi
if [[ "${SCAD_ENABLE,,}" == "true" ]]; then
  "${PYTHON_BIN}" - "${MODEL_PATH}" "${TEACHER_MODEL_PATH}" <<'PY'
import sys
from transformers import AutoTokenizer

student = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
teacher = AutoTokenizer.from_pretrained(sys.argv[2], trust_remote_code=True)
if student.get_vocab() != teacher.get_vocab():
    raise SystemExit("SCAD student and teacher tokenizers must have identical vocabularies")
if student.eos_token_id != teacher.eos_token_id:
    raise SystemExit("SCAD student and teacher must use the same EOS token id")
print(
    f"SCAD tokenizer compatibility: {len(student)} tokens, eos={student.eos_token_id}",
    file=sys.stderr,
)
PY
fi
fi
if [[ "${STRATIFIED_SOURCE_SAMPLING,,}" == "true" && ! -f "${PROJECT_ROOT}/samplers/stratified_source.py" ]]; then
  echo "Missing stratified source sampler: ${PROJECT_ROOT}/samplers/stratified_source.py" >&2
  exit 1
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  mkdir -p "${OUTPUT_DIR}/checkpoints" "${VALIDATION_DIR}" "${ROLLOUT_DATA_DIR}" "${OUTPUT_DIR}/tensorboard"
fi
# Ray starts many Python workers. Prevent each worker from creating a full
# machine-sized BLAS/OpenMP pool; retrieval runs in a separate process and
# keeps its own FAISS thread setting.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export LOCAL_SEARCH_URL
export VERL_FILE_LOGGER_PATH="${METRICS_FILE}"
export TENSORBOARD_DIR="${OUTPUT_DIR}/tensorboard"
if [[ "${RESUME_MODE}" == "auto" && -s "${VERL_FILE_LOGGER_PATH}" ]]; then
  export VERL_FILE_LOGGER_APPEND=1
else
  export VERL_FILE_LOGGER_APPEND=0
fi

RESUME_ARGS=(trainer.resume_mode="${RESUME_MODE}")
if [[ "${RESUME_MODE}" == "resume_path" ]]; then
  if [[ -z "${RESUME_FROM_PATH:-}" ]]; then
    echo "RESUME_FROM_PATH is required when RESUME_MODE=resume_path" >&2
    exit 1
  fi
  RESUME_ARGS+=(trainer.resume_from_path="${RESUME_FROM_PATH}")
fi
HYDRA_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  HYDRA_ARGS+=(--cfg job --resolve)
fi
SAMPLER_ARGS=()
if [[ "${STRATIFIED_SOURCE_SAMPLING,,}" == "true" ]]; then
  SAMPLER_ARGS+=(
    data.shuffle=False
    data.sampler.class_path="${PROJECT_ROOT}/samplers/stratified_source.py"
    data.sampler.class_name=StratifiedSourceBatchSampler
    +data.sampler.seed="${SEED}"
    +data.sampler.total_steps="${TOTAL_STEPS}"
    +data.sampler.batch_size="${TRAIN_BATCH_SIZE}"
    +data.sampler.primary_source=browsecomp_plus
    +data.sampler.secondary_source=webwalker_qa
    +data.sampler.primary_per_batch=4
    +data.sampler.secondary_per_batch=2
    +data.sampler.audit_path="${OUTPUT_DIR}/stratified_source_audit.json"
  )

else
  SAMPLER_ARGS+=(data.shuffle=True)
fi
SCAD_ARGS=(
  algorithm.adv_estimator="${ADV_ESTIMATOR}"
  algorithm.scad.enable="${SCAD_ENABLE}"
  algorithm.scad.distillation_coef="${SCAD_DISTILLATION_COEF}"
  algorithm.scad.rl_coef="${SCAD_RL_COEF}"
  algorithm.scad.distillation_clip="${SCAD_DISTILLATION_CLIP}"
  algorithm.scad.node_merge.enable="${SCAD_NODE_MERGE}"
  algorithm.scad.node_merge.semantic="${SCAD_NODE_SEMANTIC}"
  algorithm.scad.node_merge.subtask_similarity_threshold="${SCAD_SUBTASK_THRESHOLD}"
  algorithm.scad.node_merge.entity_similarity_threshold="${SCAD_ENTITY_THRESHOLD}"
  algorithm.scad.node_merge.embedding_url="${LOCAL_SEARCH_URL}/embed"
  algorithm.scad.node_merge.embedding_timeout_s="${SCAD_EMBED_TIMEOUT}"
  algorithm.scad.node_merge.embedding_batch_size="${SCAD_EMBED_BATCH_SIZE}"
  algorithm.scad.node_merge.audit_dir="${SCAD_NODE_AUDIT_DIR}"
  algorithm.scad.tree_credit.enable="${SCAD_TREE_CREDIT}"
  algorithm.scad.tree_credit.episode_coef="${SCAD_TREE_EPISODE_COEF}"
  algorithm.scad.tree_credit.planner_coef="${SCAD_TREE_PLANNER_COEF}"
  algorithm.scad.tree_credit.prior_count="${SCAD_TREE_PRIOR_COUNT}"
  algorithm.scad.tree_credit.normalize="${SCAD_TREE_NORMALIZE}"
  algorithm.scad.tree_credit.clip="${SCAD_TREE_CLIP}"
)
if [[ "${SCAD_ENABLE,,}" == "true" ]]; then
  SCAD_ARGS+=(
    +actor_rollout_ref.ref.model.path="${TEACHER_MODEL_PATH}"
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_LENGTH}"
    actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}"
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16
  )
fi

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m scripts.train_scad \
  "${SCAD_ARGS[@]}" \
  algorithm.norm_adv_by_std_in_grpo=True \
  actor_rollout_ref.rollout.agent.default_agent_loop=scad_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.rollout.prompt_length="${PROMPT_LENGTH}" \
  actor_rollout_ref.rollout.response_length="${RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_LENGTH}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_LENGTH}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_WORKERS}" \
  +actor_rollout_ref.rollout.agent.sticky_sessions=True \
  +actor_rollout_ref.rollout.agent.hash_sticky_sessions=True \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
  actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${VAL_FILE}" \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${PROMPT_LENGTH}" \
  data.max_response_length="${RESPONSE_LENGTH}" \
  data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  data.filter_overlong_prompts=False \
  data.dataloader_num_workers=0 \
  data.seed="${SEED}" \
  "${SAMPLER_ARGS[@]}" \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload="${PARAM_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OPTIMIZER_OFFLOAD}" \
  actor_rollout_ref.actor.checkpoint.save_contents="${CHECKPOINT_SAVE_CONTENTS}" \
  actor_rollout_ref.actor.fsdp_config.seed="${SEED}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_LENGTH}" \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu="${MAX_LENGTH}" \
  +actor_rollout_ref.rollout.plugin.enable_thinking=False \
  +actor_rollout_ref.rollout.plugin.max_turn="${MAX_TURN}" \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=2048 \
  +actor_rollout_ref.rollout.plugin.max_session="${MAX_SESSION}" \
  +actor_rollout_ref.rollout.plugin.val_max_session="${MAX_SESSION}" \
  +actor_rollout_ref.rollout.plugin.search_snippet_tokens="${SEARCH_SNIPPET_TOKENS}" \
  +actor_rollout_ref.rollout.plugin.seen_search_snippet_tokens="${SEEN_SEARCH_SNIPPET_TOKENS}" \
  +actor_rollout_ref.rollout.plugin.open_page_tokens="${OPEN_PAGE_TOKENS}" \
  +actor_rollout_ref.rollout.plugin.tool_observation_tokens="${TOOL_OBSERVATION_TOKENS}" \
  +actor_rollout_ref.rollout.plugin.branch_return_reserve_tokens="${BRANCH_RETURN_RESERVE_TOKENS}" \
  +actor_rollout_ref.rollout.plugin.branch_min_observation_tokens="${BRANCH_MIN_OBSERVATION_TOKENS}" \
  +actor_rollout_ref.rollout.plugin.session_timeout=3600 \
  +actor_rollout_ref.rollout.plugin.branch_len="${BRANCH_LEN}" \
  +actor_rollout_ref.rollout.plugin.must_search=True \
  +actor_rollout_ref.rollout.plugin.val_max_turn="${MAX_TURN}" \
  +actor_rollout_ref.rollout.plugin.val_response_length="${RESPONSE_LENGTH}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.val_only="${VAL_ONLY}" \
  +trainer.save_initial_checkpoint=True \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes=1 \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.test_freq="${VAL_FREQ}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
  +trainer.track_best_checkpoint="${TRACK_BEST_CHECKPOINT}" \
  +trainer.keep_best_and_latest="${KEEP_BEST_AND_LATEST}" \
  "${RESUME_ARGS[@]}" \
  trainer.default_local_dir="${OUTPUT_DIR}/checkpoints" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  trainer.validation_data_dir="${VALIDATION_DIR}" \
  'trainer.logger=[console,file,tensorboard]' \
  trainer.project_name=SCAD \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  "${HYDRA_ARGS[@]}"
