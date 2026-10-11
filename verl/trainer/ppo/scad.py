# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Structured credit assignment and subtask-local distillation."""
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch


def _scad_object_records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def compute_scad_tree_credit(
    scores: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    index: np.ndarray,
    gen_uid: np.ndarray,
    roles: np.ndarray,
    sample_valid: np.ndarray,
    path_nodes: np.ndarray,
    planner_node_spans: np.ndarray,
    prior_count: float = 2.0,
    normalize: bool = True,
    clip: float | None = 5.0,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute Planner sibling-relative MC credit on the frozen trie.

    Planner actions compare subtask clusters (plus STOP) under the same parent.
    Baselines are visit-weighted and shrunk toward the question-level return
    prior for sparse prefixes.  Worker credit is intentionally not inferred
    from free-form branch reports.
    """

    bsz, response_length = response_mask.shape
    device = scores.device
    planner_token_adv = torch.zeros((bsz, response_length), dtype=torch.float32, device=device)

    # One terminal score per sampled rollout, explicitly preferring its
    # Planner row so agent-loop flattening and later batch reordering cannot
    # change the value used by the tree.
    rollout_scores: dict[tuple[str, str], float] = {}
    rollout_planner_rows: dict[tuple[str, str], int] = {}
    for sample_index in range(bsz):
        if not sample_valid[sample_index]:
            continue
        key = str(index[sample_index]), str(gen_uid[sample_index])
        score = float(scores[sample_index].detach().cpu())
        if key not in rollout_scores or str(roles[sample_index]) == "planner":
            rollout_scores[key] = score
        if str(roles[sample_index]) == "planner":
            rollout_planner_rows[key] = sample_index

    question_scores: dict[str, list[float]] = defaultdict(list)
    for (uid, _), score in rollout_scores.items():
        question_scores[uid].append(score)
    question_prior = {
        uid: float(np.mean(values)) if values else 0.0 for uid, values in question_scores.items()
    }
    question_scale = {
        uid: (
            float(np.std(values, ddof=1))
            if len(values) >= 2 and float(np.std(values, ddof=1)) > epsilon
            else 1.0
        )
        for uid, values in question_scores.items()
    }

    # key -> action -> {gen_uid: terminal return}. Dictionaries enforce the
    # first-visit condition even if metadata is duplicated elsewhere.
    planner_groups: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (uid, rollout_uid), planner_row in rollout_planner_rows.items():
        score = rollout_scores[(uid, rollout_uid)]
        for node in _scad_object_records(path_nodes[planner_row]):
            node_id = str(node.get("node_id", ""))
            parent_id = str(node.get("parent_node_id", ""))
            subtask_cluster = str(node.get("subtask_cluster_id", ""))
            kind = str(node.get("kind", "pair"))
            if not node_id or not parent_id or not subtask_cluster:
                continue
            planner_action = "__STOP__" if kind == "stop" else subtask_cluster
            planner_groups[(uid, parent_id)][planner_action][rollout_uid] = score

    def sibling_advantages(groups):
        lookups = {}
        group_count = 0
        multi_action_count = 0
        singleton_count = 0
        for group_key, actions in groups.items():
            uid = group_key[0]
            action_returns = {
                action: list(returns_by_rollout.values())
                for action, returns_by_rollout in actions.items()
                if returns_by_rollout
            }
            if not action_returns:
                continue
            group_count += 1
            multi_action_count += int(len(action_returns) >= 2)
            singleton_count += int(len(action_returns) == 1)
            visits = [value for values in action_returns.values() for value in values]
            local_mean = float(np.mean(visits))
            n_visits = len(visits)
            prior_weight = max(float(prior_count), 0.0)
            baseline = (
                n_visits * local_mean + prior_weight * question_prior.get(uid, local_mean)
            ) / max(n_visits + prior_weight, epsilon)

            q_values = {action: float(np.mean(values)) for action, values in action_returns.items()}
            if normalize:
                weighted_q = [q_values[action] for action, values in action_returns.items() for _ in values]
                scale = float(np.std(weighted_q, ddof=1)) if len(weighted_q) >= 2 else 0.0
                if scale <= epsilon:
                    scale = question_scale.get(uid, 1.0)
            else:
                scale = 1.0
            for action, q_value in q_values.items():
                advantage = (q_value - baseline) / max(scale, epsilon)
                if clip is not None and float(clip) > 0:
                    advantage = float(np.clip(advantage, -float(clip), float(clip)))
                lookups[(*group_key, action)] = advantage
        return lookups, group_count, multi_action_count, singleton_count

    planner_lookup, planner_group_count, planner_multi_count, planner_singleton_count = sibling_advantages(
        planner_groups
    )

    planner_span_tokens = 0.0
    planner_total_tokens = 0.0
    for sample_index in range(bsz):
        if not sample_valid[sample_index]:
            continue
        role = str(roles[sample_index])
        uid = str(index[sample_index])
        if role == "planner":
            planner_total_tokens += float(response_mask[sample_index].sum().detach().cpu())
            for span in _scad_object_records(planner_node_spans[sample_index]):
                parent_id = str(span.get("parent_node_id", ""))
                subtask_cluster = str(span.get("subtask_cluster_id", ""))
                kind = str(span.get("kind", "pair"))
                action = "__STOP__" if kind == "stop" else subtask_cluster
                advantage = planner_lookup.get((uid, parent_id, action), 0.0)
                try:
                    start = min(max(int(span.get("start", 0)), 0), response_length)
                    end = min(max(int(span.get("end", start)), start), response_length)
                except (TypeError, ValueError):
                    continue
                if end <= start:
                    continue
                planner_token_adv[sample_index, start:end] += float(advantage)
                planner_span_tokens += float(
                    response_mask[sample_index, start:end].sum().detach().cpu()
                )

    planner_token_adv *= response_mask
    metrics = {
        "scad/tree_planner_groups": float(planner_group_count),
        "scad/tree_planner_multi_action_fraction": planner_multi_count / max(planner_group_count, 1),
        "scad/tree_planner_singleton_fraction": planner_singleton_count / max(planner_group_count, 1),
        "scad/tree_planner_span_token_fraction": planner_span_tokens / max(planner_total_tokens, 1.0),
    }
    return planner_token_adv, metrics


def compute_scad_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    gen_uid: np.ndarray,
    roles: np.ndarray,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    subtask_indices: Optional[np.ndarray] = None,
    conditioning_modes: Optional[np.ndarray] = None,
    node_ids: Optional[np.ndarray] = None,
    parent_node_ids: Optional[np.ndarray] = None,
    node_paths: Optional[np.ndarray] = None,
    subtask_cluster_ids: Optional[np.ndarray] = None,
    path_nodes: Optional[np.ndarray] = None,
    planner_node_spans: Optional[np.ndarray] = None,
    dummy_gen_uid: Optional[Any] = None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    distillation_coef: float = 1.0,
    rl_coef: float = 1.0,
    distillation_clip: Optional[float] = 5.0,
    tree_credit_enable: bool = False,
    tree_episode_coef: float = 1.0,
    tree_planner_coef: float = 1.0,
    tree_prior_count: float = 2.0,
    tree_normalize: bool = True,
    tree_clip: Optional[float] = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute SCAD's subtask-local distillation and role/sign routed RL signal.

    A trajectory is first assigned a group-relative outcome advantage,
    de-duplicated by ``gen_uid`` because one rollout produces a
    Planner sequence and several Worker sequences.  The final signal is:

    * Planner: signed trajectory advantage plus span-local subtask/STOP credit.
    * Worker: positive trajectory advantage plus token-level on-policy
      distillation ``log p_T - log p_old``.

    Negative trajectory credit is stopped at the Planner/Worker boundary.  If
    tree credit is enabled, signed sibling-relative MC credit is added only to
    Planner subtask/STOP decisions under a shared parent and broadcast only to
    the recorded decision-token span.  Free-form Worker reports do not receive
    answer-node credit.
    """

    if student_log_probs.shape != response_mask.shape:
        raise ValueError(
            f"student_log_probs shape {student_log_probs.shape} != response_mask {response_mask.shape}"
        )
    if teacher_log_probs.shape != response_mask.shape:
        raise ValueError(
            f"teacher_log_probs shape {teacher_log_probs.shape} != response_mask {response_mask.shape}"
        )

    scores = token_level_rewards.sum(dim=-1).float()
    device = scores.device
    bsz, response_length = response_mask.shape
    roles = np.asarray(roles, dtype=object)
    index = np.asarray(index, dtype=object)
    gen_uid = np.asarray(gen_uid, dtype=object)
    if not (len(roles) == len(index) == len(gen_uid) == bsz):
        raise ValueError("SCAD metadata length must equal batch size")
    for name, values in (
        ("node_ids", node_ids),
        ("parent_node_ids", parent_node_ids),
        ("node_paths", node_paths),
        ("subtask_cluster_ids", subtask_cluster_ids),
        ("path_nodes", path_nodes),
        ("planner_node_spans", planner_node_spans),
        ("conditioning_modes", conditioning_modes),
    ):
        if values is not None and len(values) != bsz:
            raise ValueError(f"SCAD {name} length must equal batch size")
    sample_valid = np.ones(bsz, dtype=bool)
    if dummy_gen_uid is not None:
        sample_valid = np.asarray(
            [str(value) != str(dummy_gen_uid) for value in gen_uid],
            dtype=bool,
        )

    def group_relative_scalars(group_keys, *, eligible=None, singleton_passthrough=True):
        eligible = np.ones(bsz, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
        group_scores: dict[Any, list[torch.Tensor]] = defaultdict(list)
        group_seen: dict[Any, set[Any]] = defaultdict(set)
        for sample_idx in range(bsz):
            if not eligible[sample_idx]:
                continue
            key = group_keys[sample_idx]
            rollout_key = gen_uid[sample_idx]
            if rollout_key in group_seen[key]:
                continue
            group_seen[key].add(rollout_key)
            group_scores[key].append(scores[sample_idx])

        means: dict[Any, torch.Tensor] = {}
        stds: dict[Any, torch.Tensor] = {}
        for key, values in group_scores.items():
            if len(values) >= 2:
                values_tensor = torch.stack(values)
                means[key] = values_tensor.mean()
                stds[key] = values_tensor.std()
            elif singleton_passthrough:
                means[key] = torch.zeros((), device=device)
                stds[key] = torch.ones((), device=device)

        result = torch.zeros_like(scores)
        for sample_idx in range(bsz):
            if not eligible[sample_idx]:
                continue
            key = group_keys[sample_idx]
            if key not in means:
                continue
            centered = scores[sample_idx] - means[key]
            result[sample_idx] = (
                centered / (stds[key] + epsilon)
                if norm_adv_by_std_in_grpo
                else centered
            )
        return result

    global_scalars = group_relative_scalars(
        index,
        eligible=sample_valid,
        singleton_passthrough=True,
    )
    planner_samples = np.asarray(
        [sample_valid[i] and str(role) == "planner" for i, role in enumerate(roles)],
        dtype=bool,
    )
    worker_samples = np.asarray(
        [sample_valid[i] and str(role) == "worker" for i, role in enumerate(roles)],
        dtype=bool,
    )

    sample_valid_tensor = torch.as_tensor(sample_valid, device=device).unsqueeze(-1)
    effective_response_mask = response_mask * sample_valid_tensor
    token_mask = effective_response_mask.to(student_log_probs.dtype)
    teacher_delta = (teacher_log_probs.float() - student_log_probs.float()) * token_mask
    if distillation_clip is not None and float(distillation_clip) > 0:
        teacher_delta = teacher_delta.clamp(min=-float(distillation_clip), max=float(distillation_clip))

    planner_mask = torch.as_tensor(planner_samples, device=device).unsqueeze(-1) * effective_response_mask
    worker_mask = torch.as_tensor(worker_samples, device=device).unsqueeze(-1) * effective_response_mask
    global_adv = global_scalars.unsqueeze(-1) * effective_response_mask
    distillation_mask = worker_mask
    distillation_adv = teacher_delta * worker_mask

    if tree_credit_enable:
        required_tree_metadata = {
            "path_nodes": path_nodes,
            "planner_node_spans": planner_node_spans,
        }
        missing_tree_metadata = [name for name, value in required_tree_metadata.items() if value is None]
        if missing_tree_metadata:
            raise RuntimeError(
                "SCAD tree credit requires frozen node metadata: "
                + ", ".join(missing_tree_metadata)
            )
        planner_tree_adv, tree_metrics = compute_scad_tree_credit(
            scores,
            effective_response_mask,
            index=index,
            gen_uid=gen_uid,
            roles=roles,
            sample_valid=sample_valid,
            path_nodes=np.asarray(path_nodes, dtype=object),
            planner_node_spans=np.asarray(planner_node_spans, dtype=object),
            prior_count=tree_prior_count,
            normalize=tree_normalize,
            clip=tree_clip,
            epsilon=epsilon,
        )
    else:
        planner_tree_adv = torch.zeros_like(global_adv)
        tree_metrics = {}

    effective_episode_coef = float(tree_episode_coef) if tree_credit_enable else 1.0

    rl_adv = global_adv * planner_mask + torch.relu(global_adv) * worker_mask
    advantages = float(rl_coef) * effective_episode_coef * rl_adv + float(distillation_coef) * distillation_adv
    if tree_credit_enable:
        advantages = advantages + float(tree_planner_coef) * planner_tree_adv

    advantages = advantages * effective_response_mask

    def masked_stat(values, mask, *, square=False):
        mask = mask.to(values.dtype)
        denom = mask.sum().clamp_min(1)
        selected = values.square() if square else values
        return float((selected * mask).sum().div(denom).detach().cpu())

    worker_count = int(worker_samples.sum())
    valid_count = int(sample_valid.sum())
    worker_global_negative = (global_scalars < 0) & torch.as_tensor(worker_samples, device=device)
    metrics = {
        "scad/planner_sample_fraction": int(planner_samples.sum()) / max(valid_count, 1),
        "scad/worker_sample_fraction": worker_count / max(valid_count, 1),
        "scad/worker_suppressed_negative_fraction": float(
            worker_global_negative.float().sum().div(max(worker_count, 1)).cpu()
        ),
        "scad/distillation_token_fraction": float(
            distillation_mask.float().sum().div(effective_response_mask.sum().clamp_min(1)).cpu()
        ),
        "scad/teacher_gap_mean": masked_stat(teacher_delta, distillation_mask),
        "scad/teacher_abs_gap_mean": masked_stat(teacher_delta.abs(), distillation_mask),
        "scad/teacher_preferred_token_fraction": masked_stat(
            (teacher_delta > 0).float(), distillation_mask
        ),
        "scad/teacher_gap_rms": masked_stat(teacher_delta, distillation_mask, square=True) ** 0.5,
        "scad/final_adv_mean": masked_stat(advantages, effective_response_mask),
        "scad/final_adv_rms": masked_stat(advantages, effective_response_mask, square=True) ** 0.5,
    }
    metrics.update(tree_metrics)
    if tree_credit_enable:
        metrics.update(
            {
                "scad/tree_planner_adv_rms": masked_stat(
                    planner_tree_adv, planner_mask, square=True
                )
                ** 0.5,
            }
        )
    if node_ids is not None:
        node_ids = np.asarray(node_ids, dtype=object)
        valid_node_ids = [
            str(node_ids[i])
            for i in range(bsz)
            if sample_valid[i] and str(node_ids[i]) not in {"", "None", "__dummy__"}
        ]
        metrics["scad/node_id_available_fraction"] = len(valid_node_ids) / max(valid_count, 1)
        metrics["scad/node_id_unique_fraction"] = len(set(valid_node_ids)) / max(len(valid_node_ids), 1)
    if conditioning_modes is not None:
        conditioning_modes = np.asarray(conditioning_modes, dtype=object)
        if len(conditioning_modes) == bsz:
            local_workers = sum(
                bool(worker_samples[i] and str(conditioning_modes[i]) == "local")
                for i in range(bsz)
            )
            metrics["scad/local_conditioning_worker_fraction"] = local_workers / max(worker_count, 1)

    if subtask_indices is not None:
        subtask_indices = np.asarray(subtask_indices)
        if len(subtask_indices) == bsz:
            for bucket_name, predicate in (
                ("0", lambda depth: depth == 0),
                ("1", lambda depth: depth == 1),
                ("2", lambda depth: depth == 2),
                ("3plus", lambda depth: depth >= 3),
            ):
                sample_mask = torch.as_tensor(
                    [worker_samples[i] and predicate(int(subtask_indices[i])) for i in range(bsz)],
                    device=device,
                ).unsqueeze(-1)
                depth_mask = sample_mask * response_mask
                if depth_mask.sum().item() > 0:
                    metrics[f"scad/teacher_gap_depth_{bucket_name}"] = masked_stat(
                        teacher_delta, depth_mask
                    )
                    metrics[f"scad/teacher_abs_gap_depth_{bucket_name}"] = masked_stat(
                        teacher_delta.abs(), depth_mask
                    )
    return advantages, advantages, metrics
