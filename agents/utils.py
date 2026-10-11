import time
import copy
import uuid
import re, unicodedata
from dataclasses import dataclass
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, AutoTokenizer
import asyncio
from envs.local_search import LocalSearch, truncate_to_token_budget


def select_env(ability, config, extra_info=None):
    """Resolve the supported text-search dataset ability."""
    if not isinstance(ability, str):
        raise TypeError(f"ability must be a string, got {type(ability).__name__}")
    normalized = ability.strip().lower()
    if normalized in {"search", "localsearch"} or normalized.startswith("localsearch@"):
        return LocalSearch
    raise ValueError(f"Unsupported search dataset ability: {ability}")


def is_weird(text, repeat_n=128, cjk_limit=128):
    s = unicodedata.normalize('NFKC', text)
    if re.search(rf'(.)\1{{{repeat_n - 1},}}|(.{{2,12}})\2{{{repeat_n - 1},}}', s):
        return True
    CJK = ((0x4E00, 0x9FFF), (0x3040, 0x309F), (0x30A0, 0x30FF), (0xAC00, 0xD7AF))
    cjk_count = sum(any(a <= ord(c) <= b for a, b in CJK) for c in s)
    return cjk_count >= cjk_limit or (len(s) > 0 and cjk_count / len(s) > 0.8)

class LLMClass:
    async def create_completion(self, input_ids, **kwargs):
        raise NotImplementedError

class CallLLM(LLMClass):  # Call LLM in Verl RL env
    def __init__(self, url, tokenizer, config, loop, **kwargs):
        self.server_manager = url
        self.tokenizer = tokenizer
        self.config = config
        self.loop = loop
        self.sampling_params = dict(kwargs.get("sampling_params") or {})

    async def _create_completion(self, input_ids, **kwargs):
        from uuid import uuid4

        max_len = kwargs.pop('max_len', None) or self.config.prompt_length + self.config.response_length
        max_len = min(max_len, self.config.prompt_length + self.config.response_length)
        max_new_tokens = max_len - len(input_ids)
        # This is used to avoid repetitive generation.
        if hasattr(self.config, 'plugin') and getattr(self.config.plugin, 'turn_max_new_tokens', -1) > 0:
            max_new_tokens = min(max_new_tokens, self.config.plugin.turn_max_new_tokens)
        if 'max_new_tokens' in kwargs:
            max_new_tokens = min(max_new_tokens, kwargs['max_new_tokens'])

        if max_new_tokens < 10:
            print(f"[DEBUG] max_new_tokens {max_new_tokens}, skip rollout")
            return None

        uid = kwargs.pop('uid', None) or uuid4().hex

        sampling_params = kwargs.pop('sampling_params', None) or self.sampling_params
        sampling_params = {
            'temperature': sampling_params.get('temperature', 1.0),
            'top_p': sampling_params.get('top_p', 1.0),
            'repetition_penalty': sampling_params.get('repetition_penalty', 1.0),
            'logprobs': sampling_params.get('logprobs', False),
            'max_tokens': max_new_tokens,
        }

        output = await self.server_manager.generate(
            request_id=uid,
            prompt_ids=input_ids,
            sampling_params=sampling_params,
            image_data=kwargs.pop('image_data', None),
        )

        if output is None or len(output.token_ids) == 0:
            return None

        response_text = await self.loop.run_in_executor( None, lambda: self.tokenizer.decode(output.token_ids, skip_special_tokens=True))

        return {
            "choices": [{
                "message": {
                    "content": response_text,
                    "raw_output_ids": output.token_ids,
                    "response_log_probs": output.log_probs
                    if output.log_probs is not None
                    else [0.0] * len(output.token_ids),
                    "extra_data": {"input_ids": input_ids},
                    "metrics": {}
                }
            }]
        }

    async def create_completion(self, input_ids, **kwargs):
        completion = await self._create_completion(input_ids, **kwargs)
        return completion


def truncate_prompt(chat, prompt_length, tokenizer, prompt_turn, apply_chat_template_kwargs=None):
    apply_chat_template_kwargs = apply_chat_template_kwargs or {}
    exceed_len = len(tokenizer.apply_chat_template(
        chat[:prompt_turn], **apply_chat_template_kwargs
    )) + 8 - prompt_length
    _cut_idx = 0
    while exceed_len > 0:  # truncate long user prompt
        print('[PROMPT] now exceed', exceed_len, 'work on cut turn', _cut_idx)
        chat[_cut_idx]['content'] = tokenizer.decode(
            tokenizer.encode(chat[_cut_idx]['content'], add_special_tokens=False)[
                exceed_len + 4:], add_special_tokens=False)
        exceed_len = len(tokenizer.apply_chat_template(
            chat[:prompt_turn], **apply_chat_template_kwargs
        )) + 8 - prompt_length
        _cut_idx = _cut_idx + 1
        if _cut_idx >= prompt_turn:
            break
    return chat


class AgentContext:
    def __init__(self, chat, tokenizer, config, prompt_turn=2):
        self.tokenizer = tokenizer
        self.config = config
        self.prompt_turn = prompt_turn

        self.prompt_length = config.prompt_length
        self.response_length = config.response_length

        # Qwen3 must use the same non-thinking prefix for every turn.
        self.enable_thinking = bool(getattr(config.plugin, "enable_thinking", False))
        self.apply_chat_template_kwargs = {"enable_thinking": self.enable_thinking}

        self.context_uid = str(uuid.uuid4())

        self.chat = copy.deepcopy([turn for turn in chat])
        self.chat = truncate_prompt(
            self.chat,
            config.prompt_length,
            tokenizer,
            prompt_turn,
            apply_chat_template_kwargs=self.apply_chat_template_kwargs,
        )
        self.chat_ids = [self.get_turn_context(i) for i in range(len(self.chat))]
        self.log_probs = [[0.0] * len(turn) for turn in self.chat_ids]
        self.token_mask = [[False] * len(turn) for turn in self.chat_ids]
        self.generation_prompt = None
        self.prompt_ids_len = len(sum(self.chat_ids[:prompt_turn], []))

    def get_turn_context(self, i):
        # Render each turn independently: Qwen3 can rewrite historical assistant text.
        return self.tokenizer.apply_chat_template(
            [self.chat[i]],
            add_generation_prompt=False,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )

    def get_generation_prompt(self):
        if self.generation_prompt is None:
            tokens = self.tokenizer.apply_chat_template(
                self.chat,
                add_generation_prompt=False,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            )
            add_tokens = self.tokenizer.apply_chat_template(
                self.chat,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            )
            self.generation_prompt = add_tokens[len(tokens):]
        return self.generation_prompt

    def messages(self):
        return self.chat


    def context(self, turn_cut: int=None):
        if turn_cut is not None:
            return sum(self.chat_ids[:turn_cut], []) + self.get_generation_prompt()
        return sum(self.chat_ids, []) + self.get_generation_prompt()

    def append(self, turn, completion=None):
        self.chat.append(turn)
        if completion is None:
            self.chat_ids.append(self.get_turn_context(len(self.chat) - 1))
            self.log_probs.append([0.0] * len(self.chat_ids[-1]))
            self.token_mask.append([False] * len(self.chat_ids[-1]))
        else:
            completion_tokens = completion["choices"][0]["message"]["raw_output_ids"]
            completion_log_probs = completion["choices"][0]["message"]["response_log_probs"]
            self.chat_ids.append(self.get_generation_prompt() + completion_tokens)
            self.log_probs.append([0.0] * len(self.get_generation_prompt()) + completion_log_probs)
            self.token_mask.append([False] * len(self.get_generation_prompt()) + [True] * len(completion_tokens))
            if len(completion_tokens) == 0 or completion_tokens[-1] != self.tokenizer.eos_token_id:
                self.chat_ids[-1].append(self.tokenizer.eos_token_id)
                self.log_probs[-1].append(0.0)
                self.token_mask[-1].append(False)


    async def get_data(self):
        prompt_turn = self.prompt_turn
        prompt_length = self.prompt_length
        response_length = self.response_length

        prompt_ids = sum(self.chat_ids[:prompt_turn], [])
        if len(prompt_ids) > prompt_length:
            print('[PROMPT] prompt truncated')
            prompt_ids = prompt_ids[-prompt_length:]

        response_ids = sum(self.chat_ids[prompt_turn:], [])[:response_length]
        response_logprobs = sum(self.log_probs[prompt_turn:], [])[:response_length]
        response_mask = [1 if m else 0 for turn in self.token_mask[self.prompt_turn:] for m in turn][:response_length]
        return {
            'prompt_ids': prompt_ids,
            'response_ids': response_ids,
            'response_logprobs': response_logprobs,
            'response_mask': response_mask,
            'num_turns': len(self.chat_ids),
            'messages': self.chat,
        }


class Agent(AgentContext):
    def __init__(self, llm_client, conversations, tokenizer, config, prompt_turn=2):
        super().__init__(conversations, tokenizer, config, prompt_turn=prompt_turn)
        self.llm_client = llm_client

    async def step(self, max_new_tokens=None):
        prompt = self.context()
        max_len = self.prompt_ids_len + self.config.response_length
        if max_new_tokens is not None:
            max_len = min(len(prompt) + max_new_tokens, 131072)
        completion = await self.llm_client.create_completion(
            prompt, uid=self.context_uid, max_len=max_len, messages=self.chat)
        if completion is None:
            return None
        response = completion["choices"][0]["message"]["content"]
        self.append({'role': 'assistant', 'content': response}, completion)
        return response

    async def react(self, run_action, max_turn=64, max_tokens=None, session_timeout=60 * 60,
                    should_continue=None, summary_prompt=None, safe_finish=None, observation_prompt=None):
        if should_continue is None:
            should_continue = lambda st: True
        session_start_time = time.time()
        iteration = 0
        if max_tokens is not None:
            max_tokens = max_tokens - 512
        else:
            max_tokens = self.config.response_length - 512
        return_reserve = int(
            getattr(self.config.plugin, "branch_return_reserve_tokens", 4096)
        )
        return_reserve = min(max(return_reserve, 512), max(max_tokens // 2, 512))
        turn_reserve = max(
            int(getattr(self.config.plugin, "turn_max_new_tokens", 2048)),
            10,
        )
        minimum_observation_tokens = int(
            getattr(self.config.plugin, "branch_min_observation_tokens", 1024)
        )
        minimum_observation_tokens = max(minimum_observation_tokens, 64)
        # Reserve room for the next action, its observation, and a final Worker report.
        observation_limit = max_tokens - return_reserve
        action_start_limit = max(
            observation_limit - turn_reserve - minimum_observation_tokens,
            0,
        )

        last_response = None
        response = None
        init_len = len(self.context(turn_cut=self.prompt_turn))
        budget_return = False
        observation_truncations = 0
        while iteration < max_turn:
            if time.time() - session_start_time > session_timeout:
                print('[SESSION] Session Timeout')
                break
            if len(self.context()) - init_len >= action_start_limit:
                budget_return = True
                break

            iteration += 1
            response = await self.step()
            if response is None:
                break

            if not should_continue(response):
                last_response = response
                break
            if safe_finish is not None and safe_finish(response) is not None:
                observation = safe_finish(response)
            else:
                observation = await run_action(response)
            if observation is None:
                break
            suffix = '\n' + observation_prompt if observation_prompt else ''
            used_tokens = len(self.context()) - init_len
            suffix_tokens = len(
                self.tokenizer.encode(suffix, add_special_tokens=False)
            ) if suffix else 0
            available_observation_tokens = max(
                observation_limit - used_tokens - suffix_tokens - 16,
                minimum_observation_tokens,
            )
            observation, was_truncated, _ = truncate_to_token_budget(
                observation,
                self.tokenizer,
                available_observation_tokens,
                marker="\n[Observation truncated; return with the strongest evidence if the budget is low.]",
            )
            observation_truncations += int(was_truncated)
            observation += suffix
            self.append({'role': 'user', 'content': observation, })

        if last_response is None and summary_prompt is not None:
            if self.chat[-1]['role'] == 'user':
                self.append({'role': 'assistant', 'content': "", })
            self.append({'role': 'user', 'content': summary_prompt, })
            last_response = await self.step(max_new_tokens=return_reserve)
        elif last_response is None:
            last_response = str(response or "")

        return {
            'last_response': last_response or "",
            'iteration': iteration,
            'budget_return': budget_return,
            'observation_truncations': observation_truncations,
        }


@dataclass
class TaskContext:
    config: DictConfig
    global_step: int
    is_train: bool
    tokenizer: PreTrainedTokenizer | AutoTokenizer | None = None
    llm_client: LLMClass = None


async def run_action(env, response):
    try:
        try:
            act = time.time()
            env_return = await asyncio.wait_for(env.run_action(response), timeout=120.0)
            if time.time() - act > 10:
                print('Action Cost', time.time() - act)
        except asyncio.TimeoutError:
            print('[ACTION] Action timed out after 120 seconds')
            env_return = {'observation': 'Action timed out after 120 seconds'}
        if 'action' in env_return:
            action = env_return['action']
            if action == 'finish':
                return None
        elif env_return.get('observation', None) == 'finish':
            return None
        observation = env_return.pop('observation', 'Empty')
    except Exception as e:
        observation = f"Error: {e}"
    return observation
