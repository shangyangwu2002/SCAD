import os
import copy
import collections
import ast
import numpy as np
import re, unicodedata
from collections import Counter
import asyncio, json, httpx
import logging
import hashlib
from datetime import datetime, timezone

from envs.serpapi_search import CachedSerpApiSearchClient, SerpApiSearchClient  # noqa: F401 (re-export)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def webwalker_uses_serpapi(extra_info: dict) -> bool:
    """Use live search for explicit live rows and uncovered WebWalkerQA test items."""
    retrieval_mode = str(extra_info.get("retrieval_mode", "local_or_serpapi")).lower()
    if retrieval_mode == "serpapi_live":
        return True
    if str(extra_info.get("dataset", "")).lower() != "webwalker_qa":
        return False
    if str(extra_info.get("eval_split", "")).lower() != "test":
        return False
    coverage = extra_info.get("local_corpus_coverage")
    if coverage is None:
        try:
            coverage = json.loads(extra_info.get("strata_json") or "{}").get("coverage")
        except (TypeError, json.JSONDecodeError):
            coverage = None
    return str(coverage).lower() == "none" and retrieval_mode in {"local_or_serpapi", "serpapi"}


def _unwrap_singleton(value):
    """Accept both legacy length-one batches and scalar agent-loop samples."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.shape[0] == 1:
            return value[0]
    return value


def _answer_list(value):
    """Normalize answers from supported parquet schemas."""
    if value is None:
        return []
    if isinstance(value, dict):
        for key in ("target", "answers", "answer", "ground_truth"):
            if key in value:
                answers = _answer_list(value[key])
                if answers:
                    return answers
        return []
    if isinstance(value, np.ndarray):
        return _answer_list(value.tolist())
    if isinstance(value, (list, tuple, set)):
        answers = []
        for entry in value:
            answers.extend(_answer_list(entry))
        return answers
    answer = str(value).strip()
    return [answer] if answer else []

GRADER_TEMPLATE = r"""
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, contains all the essential information from [correct_answer], is equivalent despite minor wording/order differences (such as name order, inclusion or omission of middle names/initials, common honorifics, standard shortenings of first names, inclusion/omission of non-contradictory date parts like year, minor articles like "a"/"the", extra descriptive context, non-essential descriptive prefixes/suffixes such as "Restaurant", "Inc.", "Ltd.", or sports suffixes like "FC", "CF", "SC", inclusion/omission of subtitles in titles, minor spacing/punctuation differences — including presence/absence of quotation marks, interchangeable punctuation such as ":" / "-" / "–", case-only differences, or presence/absence of diacritics), or is within a small margin of error for numerical problems. Answer 'no' only if the extracted answer is factually incorrect, missing essential identifying information, or contradicts the [correct_answer].

confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


def parse_judge_response(judge_response: str) -> dict:
    result = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
        "parse_error": False
    }

    if not judge_response:
        result["parse_error"] = True
        return result

    # Extract extracted_final_answer (try bold formats first, then regular)
    answer_match = re.search(r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)", judge_response,
                             re.IGNORECASE | re.DOTALL)
    if not answer_match:
        answer_match = re.search(r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)", judge_response,
                                 re.IGNORECASE | re.DOTALL)
    if not answer_match:
        answer_match = re.search(r"extracted_final_answer:\s*(.*?)(?=\n|$)", judge_response, re.IGNORECASE | re.DOTALL)
    if answer_match:
        result["extracted_final_answer"] = answer_match.group(1).strip()

    # Extract reasoning/explanation
    reasoning_match = re.search(r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
                                judge_response, re.IGNORECASE | re.DOTALL)
    if not reasoning_match:
        reasoning_match = re.search(r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
                                    judge_response, re.IGNORECASE | re.DOTALL)
    if not reasoning_match:
        reasoning_match = re.search(r"reasoning:\s*(.*?)(?=\ncorrect:|$)", judge_response, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    # Accept presentation changes, but only read a standalone verdict field.
    # The old unanchored regex also matched "incorrect: yes" and could not
    # read "correct: **yes**" or JSON. Never infer a grade from prose.
    json_text = re.sub(r"\A```(?:json)?\s*|\s*```\Z", "", judge_response.strip(), flags=re.IGNORECASE)
    try:
        fields = json.loads(json_text)
    except (ValueError, TypeError):
        fields = None
    if isinstance(fields, dict):
        verdict = fields.get("correct")
        if isinstance(verdict, bool):
            result["correct"] = verdict
        elif isinstance(verdict, str) and verdict.strip().lower() in {"yes", "no"}:
            result["correct"] = verdict.strip().lower() == "yes"
        for key in ("extracted_final_answer", "reasoning"):
            if isinstance(fields.get(key), str):
                result[key] = fields[key]
    else:
        verdicts = []
        for line in judge_response.splitlines():
            plain_line = line.strip().lstrip("# ")
            plain_line = re.sub(r"^[-+]\s+", "", plain_line)
            plain_line = re.sub(r"[*`_\"']", "", plain_line)
            verdict_match = re.fullmatch(r"correct\s*:\s*(yes|no)\s*[.!]?", plain_line, re.IGNORECASE)
            if verdict_match:
                verdicts.append(verdict_match.group(1).lower() == "yes")
        if verdicts and len(set(verdicts)) == 1:
            result["correct"] = verdicts[0]

    # Extract confidence (percentage)
    confidence_match = re.search(r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE)
    if not confidence_match:
        confidence_match = re.search(r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE)
    if not confidence_match:
        confidence_match = re.search(r"confidence:\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE)
    if confidence_match:
        result["confidence"] = float(confidence_match.group(1))
        if result["confidence"] > 100:
            result["confidence"] = 100

    # Check if we got the essential fields
    if result["correct"] is None:
        result["parse_error"] = True

    return result


def em_score(label: str, pred: str) -> bool:
    ign = {'a', 'an', 'the', 'of', 'on', 'in', 'and', '&', 'for', 'to', 'by', 'with'}
    deacc = lambda s: ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    def norm(s: str) -> str:
        s = deacc(s).lower()
        s = re.sub(r'\s*\([^)]*\)\s*', ' ', s)  # drop parenthetical qualifiers: (Egypt), (US), etc.
        s = re.sub(r'[“”"\'`]+', '', s)  # drop quotes
        s = re.sub(r'[:–—\-_/.,;!()?]+', ' ', s)  # unify punctuation to spaces
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    strip = lambda s: re.sub(r'\s+', '', norm(s))
    toks = lambda s: [t for t in norm(s).split() if t not in ign and not re.fullmatch(r'\d{4}', t)]
    if strip(label) == strip(pred): return True
    lt, pt = toks(label), toks(pred)
    if not lt or not pt: return False
    if Counter(lt) == Counter(pt): return True
    if len(lt) >= 2 and len(pt) >= 2 and lt[-1] == pt[-1]:
        f1, f2 = lt[0], pt[0]
        if f1 == f2 or (min(len(f1), len(f2)) >= 4 and (f1.startswith(f2) or f2.startswith(f1))): return True
    head = lambda s: strip(re.split(r'[:–—-]', norm(s), 1)[0])
    if head(label) == head(pred): return True
    return False


_judge_semaphore = None


def _get_judge_semaphore():
    global _judge_semaphore
    if _judge_semaphore is None:
        _judge_semaphore = asyncio.Semaphore(int(os.getenv("JUDGE_MAX_CONCURRENCY", "4")))
    return _judge_semaphore


async def call_openai_raw(messages, model=None, max_retries=3):
    from openai import AsyncOpenAI
    if isinstance(messages, str):
        messages = [{'role': 'user', 'content': messages}]
    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("JUDGE_API_BASE") or os.getenv("OPENAI_BASE_URL")
    model = os.getenv("JUDGE_MODEL") or model or "gpt-4o-mini"
    if not api_key:
        raise RuntimeError("JUDGE_API_KEY (or OPENAI_API_KEY) is required for llm_judge")
    # Keep judge requests independent of temporary shell proxy settings.
    async with httpx.AsyncClient(timeout=300.0, trust_env=False) as http_client:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=300.0,
            max_retries=0,
            http_client=http_client,
        )
        async with _get_judge_semaphore():
            for attempt in range(max_retries):
                try:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0,
                    )
                    choice = resp.choices[0]
                    content = choice.message.content or ""
                    if not content.strip():
                        print("[LLM JUDGE] Empty response: " + json.dumps({
                            "model": model,
                            "finish_reason": choice.finish_reason,
                            "refusal": choice.message.refusal,
                            "response_id": resp.id,
                        }, ensure_ascii=False), flush=True)
                    return content
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"[LLM JUDGE] Error after {max_retries} attempts: {e}")
                        return f"Error: {e}"
                    await asyncio.sleep(2 ** attempt)
    return ""

async def judge(question, correct_answers, predicted_answer, *, strict=False):
    strict = strict or os.getenv("JUDGE_STRICT", "0").lower() in {"1", "true"}
    if isinstance(correct_answers, str):
        correct_answers = [correct_answers]
    correct_answers = [str(answer) for answer in correct_answers if str(answer).strip()]
    correct_answers = [
        "ttellomS saiboT"[::-1] if "tellomS saiboT"[::-1] in answer else answer
        for answer in correct_answers
    ]
    correct_answers = [
        "yayhdapottahC najnarawsiB"[::-1] if "yayhdapattahC najnarawsiB"[::-1] in answer else answer
        for answer in correct_answers
    ]
    predicted_answer = "yrtnuoC a fo htaP ehT :sedirelC sokfalG"[::-1] if "yrtnuoC a fo htaP ehT :sedirelC socfalG"[::-1] in predicted_answer else predicted_answer
    if any(em_score(answer, predicted_answer) for answer in correct_answers):
        score = 1
    elif not predicted_answer.strip() or not correct_answers:
        score = 0
    else:
        score = 0
        judge_prompt = GRADER_TEMPLATE.format(
            question=question,
            response=predicted_answer,
            correct_answer=" | ".join(correct_answers),
        )
        messages = [{'role': 'user', 'content': judge_prompt}]
        for attempt in range(3):
            response = await call_openai_raw(messages)
            grade_report = parse_judge_response(response)
            audit = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model": os.getenv("JUDGE_MODEL"),
                "question_id": hashlib.sha256(str(question).encode()).hexdigest(),
                "question": question,
                "correct_answers": correct_answers,
                "predicted_answer": predicted_answer,
                "attempt": attempt + 1,
                "response": response,
                "grade_report": grade_report,
            }
            audit_path = os.getenv("JUDGE_AUDIT_FILE")
            if audit_path:
                # A single append write keeps concurrently completed rewards
                # available even if the surrounding rollout later fails.
                record = (json.dumps(audit, ensure_ascii=False) + "\n").encode()
                fd = os.open(audit_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    written = os.write(fd, record)
                    if written != len(record):
                        raise OSError("Incomplete judge audit write")
                finally:
                    os.close(fd)
            if grade_report['parse_error']:
                print("[LLM JUDGE] Invalid grade: " + json.dumps({
                    "question_id": audit["question_id"],
                    "attempt": attempt + 1,
                    "response_preview": response[:2000],
                }, ensure_ascii=False), flush=True)
                # Keep the same rubric, model and answer; only clarify the
                # requested format after a malformed response.
                if response.strip() and not response.startswith("Error:"):
                    messages = [{'role': 'user', 'content': judge_prompt +
                        "\n\nFormatting reminder: Return the four requested fields as plain text, "
                        "one field per line. The verdict line must be exactly 'correct: yes' "
                        "or 'correct: no'. Do not omit the verdict field."}]
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                continue
            score = int(grade_report['correct'])
            break
        else:
            if strict:
                raise RuntimeError("Text final-answer judge returned no valid grade after three attempts")

    print(
        f"[Judged] score={score}\nLabels: "
        + " | ".join(correct_answers)
        + '\nModel: '
        + predicted_answer.split('\n')[0]
    )
    return score


TOKEN_TRUNCATION_MARKER = "\n[Content truncated to fit the token budget.]"


def truncate_to_token_budget(
    text: str,
    tokenizer,
    max_tokens: int,
    *,
    marker: str = TOKEN_TRUNCATION_MARKER,
) -> tuple[str, bool, int]:
    """Truncate text with the policy tokenizer, independent of language.

    ``str.split`` and ``\\S+`` do not provide a meaningful bound for pages
    written without whitespace (notably Chinese).  Tokenize only up to one
    token beyond the requested limit so pathological pages cannot also create
    a large temporary token list merely to discover that they are over budget.

    Returns ``(text, was_truncated, kept_tokens)``.  The marker is included in
    ``max_tokens`` whenever it fits.
    """
    text = str(text or "")
    max_tokens = max(int(max_tokens), 0)
    if not text:
        return "", False, 0
    if tokenizer is None:
        raise ValueError("tokenizer is required for token-budgeted observations")
    if max_tokens == 0:
        return "", True, 0

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens + 1,
    )
    if len(token_ids) <= max_tokens:
        return text, False, len(token_ids)

    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    if len(marker_ids) >= max_tokens:
        kept_ids = token_ids[:max_tokens]
        kept_tokens = len(kept_ids)
        suffix = ""
    else:
        kept_ids = token_ids[: max_tokens - len(marker_ids)]
        kept_tokens = len(kept_ids) + len(marker_ids)
        suffix = marker
    try:
        kept_text = tokenizer.decode(
            kept_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        kept_text = tokenizer.decode(kept_ids, skip_special_tokens=True)
    return kept_text.rstrip() + suffix, True, kept_tokens


class AsyncSearchClient:
    def __init__(
        self,
        base_url: str,
        source: str | None = None,
        timeout: float = 300.0,
        retries: int = 3,
        backoff: float = 0.5,
    ):
        if not base_url:
            raise ValueError("LOCAL_SEARCH_URL must be set")
        self.base_url = base_url.rstrip("/")
        self.source = source
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._client = httpx.AsyncClient(base_url=self.base_url)

    async def close(self):
        await self._client.aclose()

    async def _post(self, path: str, payload: dict):
        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                r = await self._client.post(path, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                return data.get("results", data)  # convenience: unwrap "results" if present
            except httpx.HTTPError as e:
                last_exc = e
                if attempt == self.retries:
                    raise
                await asyncio.sleep(self.backoff * attempt)
        raise last_exc  # should not reach

    async def search(self, query: str, k: int = 10):
        return await self._post("/search", {"query": query, "k": k, "source": self.source})

    async def open(self, url: str | None = None, docid: str | None = None):
        return await self._post(
            "/open",
            {"url": url, "docid": docid, "source": self.source},
        )


def extract_json_tool(text: str):
    """Parse supported JSON tool calls and normalize them to agent actions."""
    calls = []

    def as_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return list(value)
        return []

    def parse_obj(s):
        for p in (json.loads, ast.literal_eval):
            try:
                return p(s)
            except Exception:
                pass
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            frag = m.group(0)
            for p in (json.loads, ast.literal_eval):
                try:
                    return p(frag)
                except Exception:
                    pass
        return None

    for kind, body in re.findall(r"<(tool_call|answer)>\s*(.*?)\s*</\1>", text, flags=re.S):
        body = body.strip()
        if kind == "tool_call":
            if body.startswith("```") and body.endswith("```"):
                body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body, flags=re.S).strip()
            obj = parse_obj(body)
            if isinstance(obj, dict) and "name" in obj:
                args = obj.get("arguments", {})
                calls.append({"function": obj["name"], "arguments": args if isinstance(args, dict) else {}})
        elif kind == "answer":
            calls.append({"function": "finish", "arguments": {"answer": body}})
    implicit_query_count = sum(
        len(as_list(fn['arguments'].get('querys', fn['arguments'].get('query'))))
        for fn in calls
        if fn['function'] in {'search', 'web_search'} and fn['arguments'].get('topk') is None
    )
    aligned_calls = []
    for fn in calls:
        if fn['function'] in {"search", "web_search"}:
            arguments = fn['arguments']
            queries = as_list(arguments.get('querys', arguments.get('query')))
            explicit_topk = arguments.get('topk')
            try:
                topk = int(explicit_topk) if explicit_topk is not None else max(10 // max(implicit_query_count, 1), 2)
            except (TypeError, ValueError):
                topk = max(10 // max(implicit_query_count, 1), 2)
            if not queries:
                aligned_calls.append({"function": "search", "arguments": {"query": "", "topk": topk}})
            for q in queries:
                aligned_calls.append({"function": "search", "arguments": {"query": q, "topk": topk}})
        elif fn['function'] in {"visit", "web_browse"}:
            arguments = fn['arguments']
            urls = as_list(arguments.get('urls', arguments.get('url')))
            if not urls:
                aligned_calls.append({"function": "open_page", "arguments": {}})
            for url in urls:
                target = {"docid": url} if re.fullmatch(r"\d+", str(url)) else {"url": url}
                aligned_calls.append({"function": "open_page", "arguments": target})
        else:
            aligned_calls.append(fn)
    return aligned_calls


def extract_fn_call(text):
    if not text:
        return None
    if '<tool_call>' in text or '<answer>' in text:
        json_tool = extract_json_tool(text)
        if len(json_tool) > 0:
            return json_tool
    text = re.split(r'<\[[^\]]+\]>', text)[-1].strip()
    matches = list(re.finditer(r'(?m)^[ \t]*<function=([^>]+)>\s*(.*?)\s*</function>',
                               text, re.DOTALL))
    if not matches:
        return None
    groups = [[matches[0]]]
    for m in matches[1:]:
        prev = groups[-1][-1]
        line_gap = text.count('\n', prev.end(), m.start())
        groups[-1].append(m) if line_gap < 4 else groups.append([m])
    last = groups[-1]
    return [
        {
            'function': m.group(1),
            'arguments': dict(re.findall(r'<parameter=([^>]+)>(.*?)</parameter>',
                                         m.group(2), re.DOTALL))
        }
        for m in last
    ]


class LocalSearch:
    def __init__(self, config, tokenizer, ability):
        self.config = config
        self.tokenizer = tokenizer
        self.ability = ability
        self.stats = collections.Counter()
        self.stats['finish'] = 0
        self.stats['search'] = 0
        self.stats['open_page'] = 0
        self.stats['change_answer'] = 0
        self.stats['is_search'] = 0
        self.stats['is_open'] = 0
        self.stats['is_finish'] = 0
        self.stats['visit_pages'] = 0
        self.env_fail = False

        base_url = os.getenv("LOCAL_SEARCH_URL")

        self.client = AsyncSearchClient(base_url=base_url)
        self.question = None
        self.label_answer = None
        self.label_answers = []
        self.predicted_answer = None
        self.must_search = getattr(self.config.plugin, "must_search", True)
        # These are token limits, not whitespace-delimited word limits.  The
        # aggregate cap prevents a top-k search from multiplying the per-page
        # allowance into an unbounded tool response.
        self.search_snippet_tokens = int(
            getattr(self.config.plugin, "search_snippet_tokens", 512)
        )
        self.seen_search_snippet_tokens = int(
            getattr(self.config.plugin, "seen_search_snippet_tokens", 128)
        )
        self.open_page_tokens = int(
            getattr(self.config.plugin, "open_page_tokens", 2500)
        )
        self.tool_observation_tokens = int(
            getattr(self.config.plugin, "tool_observation_tokens", 6144)
        )
        for name in (
            "search_snippet_tokens",
            "seen_search_snippet_tokens",
            "open_page_tokens",
            "tool_observation_tokens",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"rollout.plugin.{name} must be positive")
        self.visited_pages = set()
        self.is_finish = False

    def _truncate_tool_text(self, text: str, max_tokens: int, stat: str) -> str:
        text, truncated, kept_tokens = truncate_to_token_budget(
            text,
            self.tokenizer,
            max_tokens,
        )
        self.stats[f'{stat}_kept_tokens'] += kept_tokens
        if truncated:
            self.stats[f'{stat}_truncated'] += 1
        return text

    async def init_env(self, item):
        extra_info = copy.deepcopy(_unwrap_singleton(item.non_tensor_batch['extra_info']))
        if not isinstance(extra_info, dict):
            raise TypeError(f"extra_info must be a mapping, got {type(extra_info).__name__}")

        # Accept ``question``/``env_kwargs`` and ``query``/``answers`` fields
        # in extra_info without rewriting the source dataset.
        env_kwargs = _unwrap_singleton(item.non_tensor_batch.get('env_kwargs', {}))
        if not isinstance(env_kwargs, dict):
            env_kwargs = {}
        reward_model = _unwrap_singleton(item.non_tensor_batch.get('reward_model', {}))
        if not isinstance(reward_model, dict):
            reward_model = {}

        metadata = copy.deepcopy(extra_info)
        for key in (
            'question',
            'data_source',
            'retrieval_source',
            'retrieval_mode',
            'local_corpus_coverage',
            'eval_split',
            'instance_id',
            'stable_hash',
        ):
            if metadata.get(key) in (None, '') and env_kwargs.get(key) not in (None, ''):
                metadata[key] = env_kwargs[key]
        metadata.setdefault('dataset', metadata.get('data_source', ''))
        metadata.setdefault('eval_split', metadata.get('split', ''))

        self.question = metadata.get('query') or metadata.get('question')
        if not self.question:
            raise ValueError("search sample is missing query/question")
        metadata['query'] = self.question

        self.label_answers = _answer_list(metadata.get('answers'))
        if not self.label_answers:
            self.label_answers = _answer_list(metadata.get('answer'))
        if not self.label_answers:
            self.label_answers = _answer_list(env_kwargs.get('ground_truth'))
        if not self.label_answers:
            self.label_answers = _answer_list(reward_model.get('ground_truth'))
        if not self.label_answers:
            raise ValueError("search sample is missing answer/ground_truth")
        self.label_answer = self.label_answers[0]
        metadata['answer'] = self.label_answer
        metadata['answers'] = self.label_answers

        self.client.source = metadata.get('retrieval_source')
        if webwalker_uses_serpapi(metadata):
            await self.client.close()
            # Disk-cached (SERPAPI_CACHE_DIR, "" disables) so re-runs replay searches and pages for free.
            self.client = CachedSerpApiSearchClient.from_env()
            self.stats['serpapi_backend'] = 1
        else:
            self.stats['local_backend'] = 1
        self.predicted_answer = None
        self.instance_info = metadata
        self.instance_info['problem_statement'] = self.question

    async def close(self):
        await self.client.close()

    async def run_action(self, response):
        self.stats['action'] += 1
        fn_call = extract_fn_call(response)
        if fn_call is None or len(fn_call) == 0:
            return {'observation': 'No function call was detected in the model response.'}
        else:
            observation = ''
            has_retrieval_observation = False
            for fn in fn_call:
                name = fn['function']
                if name == 'search':
                    has_retrieval_observation = True
                    self.stats['search'] += 1
                    self.stats['is_search'] = 1
                    query = fn['arguments'].get('query', '')
                    topk = (lambda v: int(v) if str(v).isdigit() else 10)(fn['arguments'].get('topk', 10))
                    if not query:
                        observation += '[Error] The "search" function requires a "query" argument.'
                    else:
                        observation += f'[Search Results for "{query}"]\n'
                        try:
                            serp = await self.client.search(query, 50)
                        except Exception as error:
                            self.stats['search_error'] += 1
                            observation += (
                                f"[Search backend error: {type(error).__name__}: {error}]\n"
                                "Try a different query or continue with the evidence already collected.\n"
                            )
                            serp = []
                        show_topk = 0
                        for i, page in enumerate(serp, 1):
                            if page['docid'] in self.visited_pages:
                                page['text'] = (
                                    "(This page was already seen in a previous search. Here, a shorter "
                                    "snippet is shown. If you find this page relevant, please use the "
                                    "open_page tool to inspect the full content) "
                                    + self._truncate_tool_text(
                                        page['text'],
                                        self.seen_search_snippet_tokens,
                                        'seen_search_snippet',
                                    )
                                )
                                show_topk += 0.25
                            else:
                                self.visited_pages.add(page['docid'])
                                self.stats['visit_pages'] = len(self.visited_pages)
                                page['text'] = self._truncate_tool_text(
                                    page['text'],
                                    self.search_snippet_tokens,
                                    'search_snippet',
                                )
                                show_topk += 1
                            observation += (
                                f"\n--- #{i}: {page['docid']}---\n"
                                f"docid: {page['docid']}\n"
                                f"url: {page['url']}\n"
                                f"content: {page['text']}\n"
                            )
                            if show_topk >= topk:
                                break
                        observation += "\n"

                elif name == 'open_page':
                    has_retrieval_observation = True
                    self.stats['open_page'] += 1
                    self.stats['is_open'] = 1
                    url = fn['arguments'].get('url', None)
                    docid = fn['arguments'].get('docid', None)
                    if not docid and not url:
                        observation += '[Error] The "open_page" function requires either a "docid" or a "url".'
                    else:
                        open_pages = await self.client.open(url, docid)
                        for page in open_pages:
                            page['text'] = self._truncate_tool_text(
                                page['text'],
                                self.open_page_tokens,
                                'open_page',
                            )
                            observation += (
                                f"[Opened Page Content]\n"
                                f"docid: {page['docid']}\n"
                                f"url: {page['url']}\n"
                                f"content: {page['text']}\n"
                            )
                        observation += "\n"
                elif name == 'finish':
                    self.stats['is_finish'] = 1
                    self.is_finish = True
                    answer = fn['arguments'].get('answer', "")
                    explanation = fn['arguments'].get('explanation', None)
                    confidence = fn['arguments'].get('confidence', None)
                    if len(answer.strip()) == 0:
                        print(response)
                        observation = ("Fail to parse answer. Please resubmit with the correct tool call format, eg\n"
                                       "<function=finish>\n" "<parameter=answer>YOUR ANSWER</parameter>\n"
                                       "<parameter=explanation>YOUR EXPLANATION</parameter>\n"
                                       "<parameter=confidence>YOUR CONFIDENCE</parameter>\n" "</function>\n")
                        return {'observation': observation.strip()}
                    if self.predicted_answer is not None:
                        if self.predicted_answer[0] != answer:
                            self.stats['change_answer'] += 1

                    if self.stats['search'] == 0:
                        if any(em_score(label, answer) for label in self.label_answers) and self.must_search:
                            observation = "Answer submission failed. You MUST use the search tool to verify the answer and all the evidence, and cite the correct source document in your explanation to support your claim."
                            self.must_search = False
                            return {'observation': observation.strip()}
                        answer = ""  # No search no reward
                    self.predicted_answer = (answer, explanation, confidence)

                    return {'action': 'finish'}
                else:
                    observation = f'[Error] The function "{name}" is not supported.'
            if has_retrieval_observation:
                observation, truncated, kept_tokens = truncate_to_token_budget(
                    observation,
                    self.tokenizer,
                    self.tool_observation_tokens,
                    marker="\n[Tool observation truncated to fit the token budget.]",
                )
                self.stats['tool_observation_kept_tokens'] += kept_tokens
                if truncated:
                    self.stats['tool_observation_truncated'] += 1
            observation += "\n\n* Please reflect on the information we have obtained, and keep searching for additional information if we still can not answer the question. Do not give the answer if the information is still not enough."

        return {'observation': observation.strip()}

    async def get_reward(self, item, messages, context):
        if self.env_fail:  # If env fail, direct return 0 reward
            return "", 0, {}
        if self.predicted_answer is None:
            return "", 0, {}
        reward = await judge(
            self.question,
            self.label_answers,
            self.predicted_answer[0],
        )
        return "", reward, {"exact_match": float(reward)}
