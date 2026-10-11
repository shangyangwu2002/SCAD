"""Multimodal search environment with strict frozen/live retrieval routing."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from envs import local_search
from envs.local_search import AsyncSearchClient, LocalSearch, extract_fn_call
from envs.serpapi_search import (
    SERPAPI_SEARCH_URL,
    SerpApiDiskCache,
    SerpApiSearchClient,
    _numeric_docid,
    _normalize_url,
    ordered_key_indices,
    parse_serpapi_keys,
)

DEFAULT_IMAGE_URL_MAP = Path("runtime/multimodal/dataset/image_url_map.json")


async def judge_final_answer(question, answers, prediction):
    """Score against the supplied answer alternatives without text-task heuristics."""
    if not str(prediction).strip() or not answers:
        return 0
    prompt = local_search.GRADER_TEMPLATE.format(
        question=question, response=prediction,
        correct_answer=json.dumps(list(answers), ensure_ascii=False),
    )
    prompt += "\nThe correct_answer JSON array lists alternative acceptable answers; matching any one is sufficient."
    for _ in range(3):
        response = await local_search.call_openai_raw([{"role": "user", "content": prompt}])
        report = local_search.parse_judge_response(response)
        if not report["parse_error"]:
            return int(report["correct"])
    raise RuntimeError("Multimodal final-answer judge returned no valid grade after three attempts")


def _load_mapping(path: str | Path | None) -> dict:
    if not path:
        return {}
    source = Path(path)
    if not source.is_file():
        return {}
    if source.suffix == ".jsonl":
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {str(row["image"]): str(row.get("url") or row.get("public_url")) for row in rows}
    value = json.loads(source.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, dict) else {}


def _normalize_match(match: dict) -> dict | None:
    url = _normalize_url(str(match.get("url") or match.get("link") or ""))
    if urlsplit(url).scheme not in {"http", "https"}:
        return None
    title = str(match.get("title") or "")
    source = str(match.get("source") or "")
    text = "\n".join(piece for piece in (title, source) if piece)
    return {
        "title": title,
        "url": url,
        "docid": str(match.get("docid") or _numeric_docid(url)),
        "text": text,
    }


class DirectAsyncSearchClient(AsyncSearchClient):
    """Local retrieval client isolated from shell proxy configuration."""

    def __init__(self, base_url: str, source=None, timeout=300.0, retries=3, backoff=0.5) -> None:
        if not base_url:
            raise ValueError("LOCAL_SEARCH_URL must be set")
        # Initialize the small transport state directly: the parent creates an
        # HTTP client whose transport cannot be replaced without leaking it.
        self.base_url = base_url.rstrip("/")
        self.source, self.timeout = source, timeout
        self.retries, self.backoff = retries, backoff
        self._client = httpx.AsyncClient(base_url=self.base_url, trust_env=False)


class FrozenImageSearchClient:
    """Read the immutable image-search cache; cache misses stay empty."""

    def __init__(self, cache_path: str | Path | None) -> None:
        self.cache = _load_mapping(cache_path)

    async def image_search(self, image_path: str, k: int = 20) -> list[dict]:
        entry = self.cache.get(str(image_path), {})
        matches = entry.get("visual_matches", []) if isinstance(entry, dict) else []
        # Frozen matches must refer to an existing parent document; URL hashes
        # are only resolvable by the live client that registers those hashes.
        rows = [row for match in matches if match.get("docid")
                and (row := _normalize_match(match)) is not None]
        return rows[: min(max(int(k), 1), 20)]


DEFAULT_SERPAPI_CACHE_DIR = "runtime/cache/multimodal_search"


class LensSerpApiSearchClient(SerpApiSearchClient):
    """SerpAPI client that permits Lens only for explicitly mapped public URLs."""

    def __init__(self, api_keys: tuple[str, ...], *, image_url_map: dict[str, str], **kwargs) -> None:
        cache_dir = kwargs.pop("cache_dir", None)
        super().__init__(api_keys, **kwargs)
        self.image_url_map = {str(key): str(value) for key, value in image_url_map.items() if value}
        self.disk_cache = SerpApiDiskCache(cache_dir)

    def _register(self, rows: list[dict]) -> None:
        for row in rows:
            if row.get("docid") and row.get("url"):
                self._by_docid[str(row["docid"])] = dict(row)
                self._by_url[str(row["url"])] = dict(row)

    async def search(self, query: str, k: int = 10) -> list[dict]:
        query = str(query).strip()
        limit = min(max(1, int(k)), self.max_results)
        params = {"engine": "google", "query": query, "limit": limit}
        cached = self.disk_cache.get("search", params)
        if cached is not None:
            self._register(cached)
            self._search_cache[(query, limit)] = [dict(row) for row in cached]
            return [dict(row) for row in cached]
        rows = await super().search(query, k)
        self.disk_cache.put("search", params, rows)
        return rows

    async def open(self, url: str | None = None, docid: str | None = None) -> list[dict]:
        seed = self._by_docid.get(str(docid)) if docid else None
        if seed is None and url:
            seed = self._by_url.get(_normalize_url(url))
        params = {"engine": "page", "url": seed["url"]} if seed else None
        if params is not None:
            cached = self.disk_cache.get("page", params)
            if cached is not None:
                self._page_cache[seed["url"]] = [dict(row) for row in cached]
                return [dict(row) for row in cached]
        rows = await super().open(url, docid)
        if params is not None and rows and not str(rows[0].get("text", "")).startswith("[Online page unavailable"):
            self.disk_cache.put("page", params, rows)
        return rows

    @classmethod
    def from_env(cls) -> "LensSerpApiSearchClient":
        keys = parse_serpapi_keys(os.getenv("SERPAPI_API_KEYS") or os.getenv("SERPAPI_API_KEY"))
        if not keys:
            raise RuntimeError("SERPAPI_API_KEYS is required for serpapi_live retrieval")
        return cls(
            keys,
            image_url_map=_load_mapping(os.getenv("MM_IMAGE_URL_MAP") or DEFAULT_IMAGE_URL_MAP),
            search_url=os.getenv("SERPAPI_SEARCH_URL", SERPAPI_SEARCH_URL),
            timeout=float(os.getenv("SERPAPI_TIMEOUT", "45")),
            max_results=int(os.getenv("SERPAPI_MAX_RESULTS", "20")),
            cache_dir=os.getenv("MM_SERPAPI_CACHE_DIR", DEFAULT_SERPAPI_CACHE_DIR) or None,
            proxy_url=os.getenv("MM_OPEN_PAGE_PROXY"),
        )

    async def image_search(self, image_path: str, k: int = 20) -> list[dict]:
        limit = min(max(int(k), 1), 20)
        params = {"engine": "google_lens", "image_path": str(image_path), "limit": limit}
        cached = self.disk_cache.get("lens", params)
        if cached is not None:
            self._register(cached)
            return [dict(row) for row in cached]
        rows = await self._image_search_live(image_path, k)
        self.disk_cache.put("lens", params, rows)
        return [dict(row) for row in rows]

    async def _image_search_live(self, image_path: str, k: int = 20) -> list[dict]:
        public_url = self.image_url_map.get(str(image_path))
        if not public_url:
            raise ValueError(f"No explicit public URL mapping for image path: {image_path}")
        parsed = urlsplit(public_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.port == 17897:
            raise ValueError(f"Invalid public URL mapping for image path: {image_path}")
        limit = min(max(int(k), 1), 20)
        failures = []
        for key_index in ordered_key_indices(str(image_path), len(self.api_keys)):
            self.key_slot_counts[key_index] += 1
            try:
                response = await self._client.get(self.search_url, params={
                    "engine": "google_lens", "url": public_url, "hl": "en", "country": "us",
                    "api_key": self.api_keys[key_index],
                })
            except httpx.HTTPError as error:
                failures.append(f"slot {key_index}: {type(error).__name__}")
                continue
            if response.status_code != 200:
                failures.append(f"slot {key_index}: HTTP {response.status_code}")
                continue
            try:
                payload = response.json()
            except ValueError:
                failures.append(f"slot {key_index}: invalid JSON")
                continue
            if payload.get("error"):
                failures.append(f"slot {key_index}: API error")
                continue
            rows = [row for match in payload.get("visual_matches", []) if (row := _normalize_match(match))]
            for row in rows[:limit]:
                self._by_docid[row["docid"]] = row
                self._by_url[row["url"]] = row
            return rows[:limit]
        raise RuntimeError("SerpAPI Lens failed for all configured key slots (" + "; ".join(failures) + ")")


class _RoutedClient:
    def __init__(self, local_client, mode: str, frozen_images: FrozenImageSearchClient, live_factory) -> None:
        self.local_client, self.mode = local_client, mode
        self.frozen_images, self.live_factory = frozen_images, live_factory
        self.live_client = None
        self.source = getattr(local_client, "source", None)

    def _live(self):
        if self.live_client is None:
            self.live_client = self.live_factory()
        return self.live_client

    async def search(self, query: str, k: int = 10):
        return await (self._live() if self.mode == "serpapi_live" else self.local_client).search(query, k)

    async def open(self, url=None, docid=None):
        return await (self._live() if self.mode == "serpapi_live" else self.local_client).open(url, docid)

    async def image_search(self, path: str, k: int = 20):
        client = self._live() if self.mode == "serpapi_live" else self.frozen_images
        return await client.image_search(path, k)

    async def close(self):
        await self.local_client.close()
        if self.live_client is not None:
            await self.live_client.close()


NO_CALL_REMINDER = (
    "No function call was detected in the model response. Plain text is not accepted here: reply with exactly one "
    "tool call in this format (reasoning may precede it, nothing may follow it):\n"
    "<function=search>\n<parameter=query>your query</parameter>\n</function>\n"
    "Other tools: image_search(image_path), open_page(docid or url), branch(description, prompt), "
    "return(message) inside a branch, or finish(answer, explanation, confidence) to submit the final answer."
)


class MMSearch(LocalSearch):
    # class-level defaults; the instance may override max_consecutive_no_call from plugin config
    max_consecutive_no_call = 3
    consecutive_no_call = 0
    no_call_aborted = False
    async def init_env(self, item):
        await super().init_env(item)
        mode = str(self.instance_info.get("retrieval_mode", ""))
        split = str(self.instance_info.get("eval_split", ""))
        if mode not in {"frozen_corpus", "serpapi_live"}:
            raise ValueError(f"Unsupported multimodal retrieval mode: {mode!r}")
        if split in {"train", "val"} and mode != "frozen_corpus":
            raise ValueError(f"{split} samples must use frozen_corpus retrieval")
        cache_path = os.getenv("MM_IMAGE_SEARCH_CACHE", "../SCAD_DATA/multimodal/corpus/image_search_results.json")
        await self.client.close()
        self.client = DirectAsyncSearchClient(
            os.getenv("LOCAL_SEARCH_URL"), source=self.instance_info.get("retrieval_source"))
        self.client = _RoutedClient(self.client, mode, FrozenImageSearchClient(cache_path), LensSerpApiSearchClient.from_env)
        self.stats["local_backend"] = int(mode == "frozen_corpus")
        self.stats["serpapi_backend"] = int(mode == "serpapi_live")
        # Small students sometimes answer in plain text forever; after N consecutive turns without any parsable
        # call the current loop (branch or main) is ended, the episode scores 0 and stays trainable (unmasked).
        self.max_consecutive_no_call = int(getattr(self.config.plugin, "max_consecutive_no_call", 3))

    async def image_search(self, image_path: str, k: int = 20) -> list[dict]:
        image_path = str(image_path).strip()
        allowed = {str(path) for path in self.instance_info.get("image_paths", [])}
        if image_path not in allowed:
            raise ValueError(f"Image path is not attached to this sample: {image_path}")
        return await self.client.image_search(image_path, min(max(int(k), 1), 20))

    async def run_action(self, response):
        """Execute parsed calls in order, with no label-dependent tool feedback."""
        self.stats["action"] += 1
        calls = extract_fn_call(response) or []
        observations = []
        if not calls:
            self.consecutive_no_call += 1
            self.stats["no_call"] += 1
            limit = int(getattr(getattr(self.config, "plugin", None), "max_consecutive_no_call", self.max_consecutive_no_call))
            if limit > 0 and self.consecutive_no_call >= limit:
                self.consecutive_no_call = 0
                self.no_call_aborted = True
                self.stats["no_call_abort"] += 1
                return {"action": "finish"}  # ends this loop without recording an answer
            observations.append(NO_CALL_REMINDER)
        else:
            self.consecutive_no_call = 0
        for call in calls:
            name, arguments = call["function"], call["arguments"]
            # XML formatting whitespace is not part of a document/image ID.
            # Preserve free-form query/report/answer content and the raw reply.
            arguments = {key: value.strip() if key in {"image_path", "path", "docid", "url"}
                         and isinstance(value, str) else value for key, value in arguments.items()}
            if name == "finish":
                answer = arguments.get("answer", "")
                if not isinstance(answer, str) or not answer.strip():
                    observations.append('[Error] The "finish" function requires a non-empty "answer" argument.')
                    # Invalid final submissions never end the episode or reveal labels.
                    continue
                if self.predicted_answer is not None and self.predicted_answer[0] != answer:
                    self.stats["change_answer"] += 1
                self.predicted_answer = (answer, arguments.get("explanation"), arguments.get("confidence"))
                self.is_finish = True
                self.stats["is_finish"] = 1
                self.stats["finish"] += 1
                return {"action": "finish"}
            if name not in {"image_search", "search", "open_page"}:
                observations.append(f'[Error] The function "{name}" is not supported.')
                continue
            self.stats[name] += 1
            self.stats["is_open" if name == "open_page" else "is_search"] = 1
            try:
                if name == "image_search":
                    path = arguments.get("image_path") or arguments.get("path") or ""
                    if not path:
                        raise ValueError('The "image_search" function requires an "image_path" argument.')
                    rows = await self.image_search(path, arguments.get("topk", 20))
                    observations.append(f'[Image Search Results for "{path}"]')
                elif name == "search":
                    query = arguments.get("query", "")
                    if not query:
                        raise ValueError('The "search" function requires a "query" argument.')
                    rows = await self.client.search(query, 50)
                    observations.append(f'[Search Results for "{query}"]')
                else:
                    url, docid = arguments.get("url"), arguments.get("docid")
                    if not url and not docid:
                        raise ValueError('The "open_page" function requires either a "docid" or a "url".')
                    rows = await self.client.open(url, docid)
                    observations.append("[Opened Page Content]")
            except Exception as error:
                self.stats[f"{name}_error"] += 1
                observations.append(f'[{name} error: {type(error).__name__}: {error}]')
                continue
            try:
                topk = max(1, int(arguments.get("topk", 10)))
            except (TypeError, ValueError):
                topk = 10
            shown = 0
            for row in rows:
                docid = row["docid"]
                already_seen = docid in self.visited_pages
                if name == "open_page":
                    limit, stat = self.open_page_tokens, "open_page"
                elif name == "image_search":
                    limit, stat = self.search_snippet_tokens, "image_search_snippet"
                else:
                    limit = self.seen_search_snippet_tokens if already_seen else self.search_snippet_tokens
                    stat = "seen_search_snippet" if already_seen else "search_snippet"
                content = self._truncate_tool_text(row["text"], limit, stat)
                observations.append(f'docid: {docid}\nurl: {row["url"]}\ncontent: {content}')
                self.visited_pages.add(docid)
                self.stats["visit_pages"] = len(self.visited_pages)
                if name == "search":
                    shown += 0.25 if already_seen else 1
                    if shown >= topk:
                        break
        observation = self._truncate_tool_text(
            "\n\n".join(observations), self.tool_observation_tokens, "tool_observation")
        return {"observation": observation.strip()}

    async def get_reward(self, item, messages, context):
        """Judge only the final answer, using every accepted answer variant."""
        self.stats.update(self.tool_usage())  # exported with env_stats -> reward_extra_info tool_* metrics
        if self.env_fail or self.predicted_answer is None:
            return "", 0, {}
        reward = await judge_final_answer(
            self.question, self.label_answers, self.predicted_answer[0])
        return "", reward, {"final_answer_score": float(reward)}

    def tool_usage(self) -> dict:
        """Episode-level tool counters (Planner and every Worker share this environment) for validation metrics."""
        counters = {"search", "image_search", "open_page", "action", "finish", "visit_pages", "no_call", "no_call_abort",
                    "search_error", "image_search_error", "open_page_error", "image_search_empty"}
        usage = {f"tool_{name}": float(self.stats.get(name, 0)) for name in counters}
        usage["tool_any_search"] = float(bool(self.stats.get("search", 0) or self.stats.get("image_search", 0)))
        usage["tool_used_image_search"] = float(bool(self.stats.get("image_search", 0)))
        # Observation-budget counters are incremented lazily by LocalSearch; export them for every episode so the
        # validation batches all carry the same tool_* columns (a missing column in one batch breaks aggregation).
        for name in ("tool_observation_truncated", "tool_observation_kept_tokens"):
            usage[name] = float(self.stats.get(name, 0))
        return usage
