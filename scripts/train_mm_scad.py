"""Multimodal-only dataset and agent loop registration."""
import os

import numpy as np

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, register
from verl.utils.dataset.rl_dataset import RLHFDataset
from agents.scad_agent import process_item
from agents.mm_utils import MMRuntime, load_images
from agents.utils import CallLLM, TaskContext


class MMRLHFDataset(RLHFDataset):
    """Defer visual encoding until the actual Planner/Worker prompt is built.

    Dataset prompts are transport only. Each returned rollout supplies its own
    pixels and expanded tokens, including fresh local Worker contexts.
    """
    def __init__(self, data_files, tokenizer, config, processor=None, max_samples=-1):
        if processor is None:
            raise ValueError("MMRLHFDataset requires a vision-language model")
        super().__init__(data_files, tokenizer, config, processor=None, max_samples=max_samples)

    def _build_messages(self, example):
        example.pop(self.image_key, None)
        return example.pop(self.prompt_key)


def _sample_value(value):
    """Per-sample agent-loop kwargs may arrive as scalars, length-one batches or 0-d object arrays."""
    while isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value.item()
        elif value.shape[0] == 1 and value.dtype == object:
            value = value[0]
        else:
            return value.tolist()
    return value


@register("mm_scad_agent")
class MMSCADAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        pass  # Per-instance state is set by AgentLoopBase; no shared sample state.

    async def run(self, sampling_params, **kwargs):
        item = DataProto.from_dict(non_tensors=kwargs)
        extra = _sample_value(kwargs["extra_info"])
        plugin = self.config.actor_rollout_ref.rollout.plugin
        root = os.getenv("MM_DATASET_ROOT", "../SCAD_DATA/multimodal/dataset")
        paths = [str(path) for path in _sample_value(extra["image_paths"])]
        images = load_images(root, paths, int(plugin.get("mm_max_image_pixels", 524288)))
        runtime = MMRuntime(self.processor, images, paths)
        client = CallLLM(self.server_manager, self.tokenizer, self.config.actor_rollout_ref.rollout,
                         self.loop, sampling_params=sampling_params)
        context = TaskContext(self.config, kwargs.get("global_step", 0),
                              not kwargs.get("validate", False), self.tokenizer, client)
        return await process_item(item, context, runtime=runtime)
