import os
import re
import time
import copy
import asyncio
from functools import partial
import numpy as np
from uuid import uuid4
from typing import Union

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from envs.local_search import extract_json_tool

from .utils import Agent, select_env, is_weird, TaskContext, run_action
from .prompts import create_chat, create_worker_chat


def _unwrap_singleton(value):
    """Unwrap a single agent-loop sample from its batch container."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.shape[0] == 1:
            return value[0]
    return value


def print_chat(chat):
    chat_str = ""
    for turn in chat:
        content = turn['content']
        if isinstance(content, list):
            content = '\n'.join(part.get('text', '[image]') for part in content)
        if is_weird(str(turn)):
            chat_str += '# ' + turn['role'] + ' **CJK**\n\n' + content + "\n\n---\n\n"
        else:
            chat_str += '# ' + turn['role'] + '\n\n' + content + "\n\n---\n\n"
    return chat_str

def extract_fn_call(text):
    if text is None:
        return None
    if '<tool_call>' in text or '<answer>' in text:
        calls = extract_json_tool(text)
        return calls[-1] if calls else None
    func_matches = re.findall(r'<function=([^>]+)>', text)
    if not func_matches:
        return None
    last_function = func_matches[-1]
    last_func_pos = text.rfind(f'<function={last_function}>')
    text_after_last_func = text[last_func_pos:]
    params = dict(re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', text_after_last_func, re.DOTALL))
    return {'function': last_function, 'arguments': params}


def is_fn_call(text, *names):
    call = extract_fn_call(text)
    return call is not None and call['function'] in names


def clean_response(response):
    if '<function=return>' in response:
        response = response.split('<function=return>')[-1]
    else:
        response = re.split(r'<\[[^\]]+\]>', response)[-1]
    return response


def _generated_turn_span(agent, turn_index, response_length):
    """Return the generated-token half-open span in Agent.get_data response coordinates."""

    if turn_index < agent.prompt_turn or turn_index >= len(agent.token_mask):
        return None
    turn_offset = sum(len(tokens) for tokens in agent.chat_ids[agent.prompt_turn:turn_index])
    generated_positions = [
        position for position, is_generated in enumerate(agent.token_mask[turn_index]) if is_generated
    ]
    if not generated_positions:
        return None
    start = turn_offset + generated_positions[0]
    end = turn_offset + generated_positions[-1] + 1
    start = min(max(start, 0), response_length)
    end = min(max(end, start), response_length)
    return (start, end) if end > start else None


async def get_terminal_reward(env, item, messages, context):
    """Close the environment even when strict reward evaluation aborts a batch."""
    try:
        score_msg, reward, reward_dict = await asyncio.wait_for(
            env.get_reward(item, messages, context), timeout=60 * 10,
        )
        score = (score_msg, reward)
        print(score)
        return score, reward_dict
    except Exception as error:
        print(f"[Error] Getting reward: {error}")
        if os.getenv("JUDGE_STRICT", "0").lower() in {"1", "true"}:
            raise
        return ("", 0), {"ans_reward": 0.0, "format_reward": 0.0, "ref_reward": 0.0}
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            await close()


async def process_item(
        item: DataProto,
        context: TaskContext,
        runtime=None,
) -> Union[AgentLoopOutput, list[AgentLoopOutput]]:
    for proxy_key in ("NO_PROXY", "no_proxy"):
        exclusions = {value for value in os.environ.get(proxy_key, "").split(",") if value}
        exclusions.update({"127.0.0.1", "localhost"})
        os.environ[proxy_key] = ",".join(sorted(exclusions))
    tokenizer = context.tokenizer
    agent_factory = runtime.agent if runtime is not None else Agent
    env_selector = runtime.select_env if runtime is not None else select_env
    prompt_builder = runtime.create_chat if runtime is not None else create_chat
    worker_prompt_builder = runtime.create_worker_chat if runtime is not None else create_worker_chat

    config = context.config.actor_rollout_ref.rollout
    is_train = context.is_train

    if not is_train:
        if getattr(config.plugin, "val_response_length", None):
            config.response_length = getattr(config.plugin, "val_response_length", None)

    ability = _unwrap_singleton(item.non_tensor_batch['ability'])

    uid = _unwrap_singleton(item.non_tensor_batch.get('uid', uuid4().hex))
    gen_uid = _unwrap_singleton(item.non_tensor_batch.get('gen_uid', None))

    EnvClass = env_selector(ability, config)
    env = EnvClass(config, tokenizer, ability)

    try:
        await env.init_env(item)
    except Exception as e:
        print(f"[Error] during environment init: {str(e)}")
        raise

    user_prompt = prompt_builder(env.instance_info['problem_statement'])

    max_turn = getattr(config.plugin, 'max_turn', 64) if config.plugin else 64
    max_session = getattr(config.plugin, "max_session", 5)
    if not is_train:
        max_turn = getattr(config.plugin, "val_max_turn", max_turn)
        max_session = getattr(config.plugin, "val_max_session", max_session)
    session_timeout = getattr(config.plugin, "session_timeout", 90 * 60)
    llm_client = context.llm_client

    prompt_turn = len(user_prompt)
    agent = dict()
    agent['main'] = agent_factory(llm_client, user_prompt, tokenizer, config, prompt_turn=prompt_turn)
    branches = []
    branch_tasks = {}
    branch_return = {}
    branch_reports = {}
    planner_decisions = []
    init_len = len(agent['main'].context())
    session_start_time = time.time()
    iteration = 0
    mask_rollout = True  # If True then no grad update on this traj
    session_message = []
    while iteration < max_turn:
        if time.time() - session_start_time > session_timeout:
            print('[SESSION] Session Timeout')
            break

        iteration += 1

        response = await agent['main'].step()

        if response is None:
            break

        session_message.append({'role': 'assistant', 'content': response})
        planner_turn_index = len(agent['main'].chat) - 1
        fn_call = extract_fn_call(response)
        if fn_call is not None and fn_call['function'] == 'branch':
            if len(branches) + 1 > max_session:
                observation = f"You've already reached the limit of {len(branches)} branch calls. Continue working independently."
            else:
                description = fn_call['arguments'].get('description', 'Agent')
                message_to_branch = fn_call['arguments'].get('prompt', 'Empty prompt')
                print('[BRANCH]', description, len(agent['main'].context()))
                agent_name = f"#{len(branches)}-" + description.replace(' ', '_')
                branches.append(agent_name)
                planner_decisions.append(
                    {
                        'kind': 'pair',
                        'index': len(branches) - 1,
                        'turn_index': planner_turn_index,
                    }
                )
                branch_tasks[agent_name] = message_to_branch
                history = worker_prompt_builder(
                    env.instance_info['problem_statement'], message_to_branch,
                )
                agent[agent_name] = agent_factory(llm_client, history, tokenizer, config, prompt_turn=prompt_turn)
                agent_return = await agent[agent_name].react(
                    partial(run_action, env),
                    max_turn=max_turn,
                    max_tokens=getattr(config.plugin, "branch_len", None),
                    session_timeout=session_timeout - time.time() + session_start_time,
                    should_continue=lambda resp: not is_fn_call(resp, 'return'),
                    safe_finish=lambda
                        x: "You are in branch mode and cannot branch task or finish the task. Use the `return` tool to go back to the main agent." if is_fn_call(x, 'finish', 'branch') else None,
                    summary_prompt="The branch is approaching its token budget. Preserve the strongest evidence already collected, clearly state the progress and pending work, and use the return tool now.",
                    observation_prompt=f"* You are now in branch mode: {description}. Conduct the sub task based on instruction, and when you complete the assigned sub task, use return tool to return, do not perform action beyond the assigned sub task.",
                )
                env.stats['branch_budget_return'] += int(agent_return.get('budget_return', False))
                env.stats['branch_observation_truncated'] += int(
                    agent_return.get('observation_truncations', 0)
                )
                iteration += agent_return['iteration']
                last_response = agent_return['last_response']
                session_message.extend(agent[agent_name].messages()[len(history):])
                fn_call = extract_fn_call(last_response)
                branch_message = None
                unwrapped_report = None
                if fn_call is not None and fn_call['function'] == 'return':
                    if 'message' in fn_call['arguments']:
                        report = fn_call['arguments'].get('message', 'Empty message')
                        unwrapped_report = report
                        branch_message = f'Branch has finished its task, the returned message is:\n\n{report}'
                elif fn_call is not None and fn_call['function'] == 'finish':
                    if 'message' in fn_call['arguments']:
                        report = fn_call['arguments'].get('message', 'Empty message')
                        unwrapped_report = report
                        branch_message = f'Branch has finished its task, the returned message is:\n\n{report}'
                if branch_message is None:
                    unwrapped_report = clean_response(last_response)
                    branch_message = f'Branch has finished its task. The last message was:\n\n{unwrapped_report}'
                observation = branch_message
                branch_return[agent_name] = observation
                branch_reports[agent_name] = unwrapped_report or ""
        elif is_fn_call(
            response, *(runtime.planner_research_tools if runtime is not None else ("search", "open_page", "return"))
        ):
            observation = (
                "Planner role violation: direct research tools are disabled. "
                "Delegate a self-contained research task with branch."
            )
        else:
            if fn_call is not None and fn_call['function'] == 'finish':
                planner_decisions.append(
                    {'kind': 'stop', 'index': -1, 'turn_index': planner_turn_index}
                )
            observation = await run_action(env, response)
            if observation is None:
                mask_rollout = False
                break

        if agent['main'].chat[-1]['role'] == 'user':
            print('[ROLE ERROR]')
            print(agent['main'].chat[-1])
            agent['main'].append({'role': 'assistant', 'content': str(response)})

        agent['main'].append({'role': 'user', 'content': observation})
        session_message.append({'role': 'user', 'content': observation})

    env.stats['session_time'] = time.time() - session_start_time

    print('[TASK] Task Finish, Start Reward')
    score, reward_dict = await get_terminal_reward(env, item, agent['main'].messages(), context)

    outs = []
    env.stats['get_final_score'] = score[1]
    env.stats['traj_num'] = len(agent)
    env.stats['main_len'] = min(len(agent['main'].context()) - init_len, config.response_length)
    env.stats['total_token'] = len(tokenizer.encode(print_chat(user_prompt + session_message)))
    env.stats['main_turn'] = len(agent['main'].messages())
    env.stats['is_branch'] = int(len(agent) > 1)
    env.stats['branch_success'] = int(int(len(agent) > 1) * score[1])
    env.stats['use_all_branch'] = int(len(branches) + 1 > max_session)

    if getattr(env, 'is_finish', False) or getattr(env, 'finish', False):
        mask_rollout = False
    if score[1] > 0:
        mask_rollout = False

    is_finish = getattr(env, 'is_finish', False) or getattr(env, 'finish', False)
    subtask_trace = tuple(
        {
            'index': index,
            'subtask': branch_tasks.get(branch_name, ""),
            # Keep the original unwrapped Worker report for post-hoc canonicalization.
            'answer': branch_reports.get(branch_name, ""),
        }
        for index, branch_name in enumerate(branches)
    )

    for name in agent if is_train else ['main']:  # in eval only return main agent
        out = await agent[name].get_data()
        is_worker = name != 'main'
        subtask_index = branches.index(name) if is_worker and name in branches else -1
        subtask_text = branch_tasks.get(name, "")
        agent_reward = score[1]
        planner_decision_spans = []
        if not is_worker:
            for decision in planner_decisions:
                span = _generated_turn_span(
                    agent['main'],
                    decision['turn_index'],
                    len(out['response_ids']),
                )
                if span is not None:
                    planner_decision_spans.append(
                        {
                            'kind': decision['kind'],
                            'index': decision['index'],
                            'start': span[0],
                            'end': span[1],
                        }
                    )
        out = AgentLoopOutput(
            prompt_ids=out['prompt_ids'],
            response_ids=out['response_ids'],
            response_mask=out['response_mask'],
            response_logprobs=out['response_logprobs'],
            multi_modal_data=out.get('multi_modal_data', {}),
            reward_score=agent_reward,
            num_turns=out['num_turns'],
            metrics=AgentLoopMetrics(),
            extra_fields={
                **({'scad_multimodal': True} if runtime is not None else {}),
                'messages': out['messages'],
                'env_stats': copy.deepcopy(env.stats) if hasattr(env, 'stats') else {},
                'num_branches': len(branches),
                'branch_names': branches,
                'mask_rollout': mask_rollout,
                'is_finish': is_finish,
                'agent_name': name,
                'scad_role': 'worker' if is_worker else 'planner',
                'subtask_index': subtask_index,
                'subtask_text': subtask_text,
                'subtask_return': branch_return.get(name, ""),
                'subtask_trace': subtask_trace,
                'planner_decision_spans': tuple(planner_decision_spans),
                'conditioning_mode': 'local' if is_worker else 'inherited',
                'message_str': print_chat(session_message),
                'meta_info': f"N: {len(agent)} | {name}",
                'uid': uid,
                'gen_uid': gen_uid,
            }
        )

        outs.append(copy.deepcopy(out))

    return outs
