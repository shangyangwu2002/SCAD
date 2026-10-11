import logging
import os
from typing import Any, Union

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl import DataProto
from agents.scad_agent import process_item
from agents.utils import CallLLM, TaskContext

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("scad_agent")
class SCADAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True

        logger.info("Initializing SCADAgentLoop class")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.config = config

    async def run(
        self, sampling_params: dict[str, Any], **kwargs
    ) -> Union[AgentLoopOutput, list[AgentLoopOutput]]:
        item = DataProto.from_dict(non_tensors=kwargs)

        llm_client = CallLLM(
            url=self.server_manager,
            tokenizer=self.tokenizer,
            config=self.config.actor_rollout_ref.rollout,
            loop=self.loop,
            sampling_params=sampling_params,
        )
        context = TaskContext(
            config=self.config,
            global_step=kwargs.get('global_step', 0),
            llm_client=llm_client,
            is_train=not kwargs.get('validate', False),
            tokenizer=self.tokenizer,
        )
        rollout_results = await process_item(item, context)

        return rollout_results


def main():
    """Run VERL's main PPO trainer with the custom agent loop registered."""
    from verl.trainer.main_ppo import main as verl_main
    verl_main()


if __name__ == "__main__":
    main()
