"""SerpAPI-backed search used only for uncovered WebWalkerQA test items."""

from __future__ import annotations

import asyncio
import hashlib
import html
import ipaddress
import json
import os
import re
import socket
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit

import httpx


SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"


def parse_serpapi_keys(raw: str | None) -> tuple[str, ...]:
    """Parse and de-duplicate a comma/whitespace separated key list."""
    keys = []
    for value in re.split(r"[,;\s]+", raw or ""):
        value = value.strip()
        if value and value not in keys:
            keys.append(value)
    return tuple(keys)


def ordered_key_indices(query: str, num_keys: int) -> tuple[int, ...]:
    """Distribute queries stably across keys and retain every key as failover."""
    if num_keys <= 0:
        return ()
    digest = hashlib.sha256(query.encode("utf-8")).digest()
    start = int.from_bytes(digest[:8], "big") % num_keys
    return tuple((start + offset) % num_keys for offset in range(num_keys))


def _normalize_url(url: str) -> str:
    url = html.unescape((url or "").strip())
    url, _fragment = urldefrag(url)
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return url
    port = f":{parsed.port}" if parsed.port else ""
    netloc = host + port
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", parsed.query, ""))


def _numeric_docid(url: str) -> str:
    value = int.from_bytes(hashlib.sha256(url.encode("utf-8")).digest()[:8], "big") % 10**15
    return f"9{value:015d}"


class _VisibleTextParser(HTMLParser):
    _SKIP_TAGS = {"script", "style", "noscript", "svg", "template", "form", "head"}
    _BLOCK_TAGS = {
        "article", "aside", "blockquote", "br", "div", "dl", "dt", "dd", "footer",
        "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav",
        "ol", "p", "pre", "section", "table", "td", "th", "tr", "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif not self._skip_depth and tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif not self._skip_depth and tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self._parts.append(data)

    def text(self) -> str:
        value = " ".join(self._parts)
        value = re.sub(r"[ \t\f\v]+", " ", value)
        value = re.sub(r" *\n *", "\n", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()


def html_to_text(value: str) -> str:
    parser = _VisibleTextParser()
    try:
        parser.feed(value)
        parser.close()
        return parser.text()
    except Exception:
        # Malformed pages should still yield a useful best-effort observation.
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()


def _decode_body(body: bytes, content_type: str) -> str:
    charset_match = re.search(r"charset\s*=\s*['\"]?([^;\s'\"]+)", content_type, re.I)
    encodings = [charset_match.group(1)] if charset_match else []
    encodings.extend(["utf-8", "gb18030"])
    best = ""
    for encoding in dict.fromkeys(encodings):
        try:
            decoded = body.decode(encoding, errors="replace")
        except LookupError:
            continue
        if not best or decoded.count("\ufffd") < best.count("\ufffd"):
            best = decoded
        if "\ufffd" not in decoded:
            break
    return best


async def _is_public_http_url(url: str, *, resolve: bool = True) -> bool:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    try:
        literal = ipaddress.ip_address(parsed.hostname)
        return literal.is_global
    except ValueError:
        pass
    if not resolve:
        # A forward proxy resolves the host itself; local DNS may be unavailable or polluted.
        return True

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addresses = await asyncio.to_thread(socket.getaddrinfo, parsed.hostname, port, type=socket.SOCK_STREAM)
    except OSError:
        return False
    resolved = {entry[4][0] for entry in addresses}
    return bool(resolved) and all(ipaddress.ip_address(address).is_global for address in resolved)


class SerpApiSearchClient:
    """Search SerpAPI and open only URLs returned by this client's searches."""

    def __init__(
        self,
        api_keys: tuple[str, ...],
        *,
        search_url: str = SERPAPI_SEARCH_URL,
        timeout: float = 45.0,
        max_results: int = 10,
        max_page_bytes: int = 5_000_000,
        transport=None,
        proxy_url: str | None = None,
    ) -> None:
        if not api_keys:
            raise ValueError("At least one SerpAPI key is required")
        self.api_keys = tuple(api_keys)
        self.search_url = search_url
        self.proxy_url = (proxy_url or "").strip() or None
        self.timeout = timeout
        self.max_results = max(1, min(int(max_results), 20))
        self.max_page_bytes = max_page_bytes
        self.source = "webwalker"
        self.key_slot_counts = [0] * len(self.api_keys)
        self._search_cache: dict[tuple[str, int], list[dict]] = {}
        self._page_cache: dict[str, list[dict]] = {}
        self._by_docid: dict[str, dict] = {}
        self._by_url: dict[str, dict] = {}
        mounts = None
        if self.proxy_url and transport is None:
            # open_page fetches go through the explicit forward proxy (MM_OPEN_PAGE_PROXY);
            # the SerpAPI host itself stays direct so quota calls never depend on the proxy.
            mounts = {"all://": httpx.AsyncHTTPTransport(proxy=self.proxy_url, trust_env=False)}
            api_host = urlsplit(self.search_url).hostname
            if api_host:
                mounts[f"all://{api_host}"] = httpx.AsyncHTTPTransport(trust_env=False)
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            transport=transport,
            mounts=mounts,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.8,zh-CN;q=0.7,zh;q=0.6",
            },
        )

    @classmethod
    def from_env(cls) -> "SerpApiSearchClient":
        keys = parse_serpapi_keys(os.getenv("SERPAPI_API_KEYS") or os.getenv("SERPAPI_API_KEY"))
        if not keys:
            raise RuntimeError(
                "SERPAPI_API_KEYS is required for WebWalkerQA test items without local corpus coverage"
            )
        return cls(
            keys,
            search_url=os.getenv("SERPAPI_SEARCH_URL", SERPAPI_SEARCH_URL),
            timeout=float(os.getenv("SERPAPI_TIMEOUT", "45")),
            max_results=int(os.getenv("SERPAPI_MAX_RESULTS", "10")),
            max_page_bytes=int(os.getenv("SERPAPI_MAX_PAGE_BYTES", "5000000")),
            proxy_url=os.getenv("MM_OPEN_PAGE_PROXY"),
        )

    async def close(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _result_text(result: dict) -> str:
        pieces = [result.get("title", ""), result.get("date", ""), result.get("snippet", "")]
        return "\n".join(str(piece).strip() for piece in pieces if str(piece).strip())

    async def search(self, query: str, k: int = 10) -> list[dict]:
        query = str(query).strip()
        limit = min(max(1, int(k)), self.max_results)
        cache_key = (query, limit)
        if cache_key in self._search_cache:
            return [dict(row) for row in self._search_cache[cache_key]]

        failures = []
        for key_index in ordered_key_indices(query, len(self.api_keys)):
            self.key_slot_counts[key_index] += 1
            try:
                response = await self._client.get(
                    self.search_url,
                    params={
                        "engine": "google",
                        "q": query,
                        "num": limit,
                        "safe": "active",
                        "api_key": self.api_keys[key_index],
                    },
                )
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

            rows = []
            for result in payload.get("organic_results", []):
                url = _normalize_url(str(result.get("link") or ""))
                if not urlsplit(url).scheme in {"http", "https"}:
                    continue
                docid = _numeric_docid(url)
                row = {
                    "docid": docid,
                    "url": url,
                    "text": self._result_text(result),
                    "title": str(result.get("title") or ""),
                    "backend": "serpapi",
                }
                self._by_docid[docid] = row
                self._by_url[url] = row
                rows.append(row)
                if len(rows) >= limit:
                    break
            self._search_cache[cache_key] = rows
            return [dict(row) for row in rows]

        detail = "; ".join(failures) or "no key slots attempted"
        raise RuntimeError(f"SerpAPI search failed for all configured key slots ({detail})")

    async def _fetch_page(self, url: str) -> tuple[str, str]:
        current = url
        for _redirect in range(6):
            public = await (_is_public_http_url(current) if self.proxy_url is None
                            else _is_public_http_url(current, resolve=False))
            if not public:
                raise RuntimeError("page URL is not a public HTTP(S) address")
            try:
                async with self._client.stream("GET", current) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise RuntimeError(f"redirect HTTP {response.status_code} has no location")
                        current = _normalize_url(urljoin(current, location))
                        continue
                    if response.status_code != 200:
                        raise RuntimeError(f"page returned HTTP {response.status_code}")
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        remaining = self.max_page_bytes - len(body)
                        if remaining <= 0:
                            break
                        body.extend(chunk[:remaining])
                    content_type = response.headers.get("content-type", "")
            except httpx.HTTPError as error:
                raise RuntimeError(f"page request failed: {type(error).__name__}") from None

            decoded = _decode_body(bytes(body), content_type)
            if "html" in content_type.lower() or "<html" in decoded[:1000].lower():
                decoded = html_to_text(decoded)
            else:
                decoded = re.sub(r"\s+", " ", decoded).strip()
            return current, decoded
        raise RuntimeError("page exceeded the redirect limit")

    async def open(self, url: str | None = None, docid: str | None = None) -> list[dict]:
        seed = self._by_docid.get(str(docid)) if docid else None
        normalized = _normalize_url(url or "") if url else ""
        if seed is None and normalized:
            seed = self._by_url.get(normalized)
        if seed is None:
            return [{
                "docid": str(docid or ""),
                "url": normalized,
                "text": "Document not found. Open a docid or URL returned by SerpAPI search.",
            }]

        target = seed["url"]
        if target in self._page_cache:
            return [dict(row) for row in self._page_cache[target]]
        try:
            final_url, text = await self._fetch_page(target)
            if not text:
                raise RuntimeError("page contained no extractable text")
        except RuntimeError as error:
            final_url = target
            snippet = seed.get("text") or ""
            text = f"[Online page unavailable: {error}]\n{snippet}".strip()
        row = {"docid": seed["docid"], "url": final_url, "text": text, "backend": "serpapi"}
        self._page_cache[target] = [row]
        return [dict(row)]


DEFAULT_TEXT_SERPAPI_CACHE_DIR = "runtime/cache/text_search"


class SerpApiDiskCache:
    """Persistent write-through cache for live SerpAPI searches, Lens matches and fetched pages.

    Keyed by a hash of the request (engine + parameters), shared across runs, methods and checkpoints so
    repeated queries never spend quota twice. Files are written atomically; unreadable entries are misses.
    Pass ``root=None`` (env var set to "") to disable.
    """

    def __init__(self, root: str | Path | None, kinds: tuple[str, ...] = ("search", "lens", "page")) -> None:
        self.root = Path(root) if root else None
        if self.root is not None:
            for kind in kinds:
                (self.root / kind).mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, kind: str, params: dict) -> Path:
        digest = hashlib.sha256(json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        return self.root / kind / f"{digest}.json"

    def get(self, kind: str, params: dict):
        if self.root is None:
            return None
        path = self._path(kind, params)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.misses += 1
            return None
        if payload.get("params") != params:
            self.misses += 1
            return None
        self.hits += 1
        return payload.get("rows")

    def put(self, kind: str, params: dict, rows: list) -> None:
        if self.root is None:
            return
        path = self._path(kind, params)
        payload = {"kind": kind, "params": params, "rows": rows, "cached_at": time.time()}
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)


class CachedSerpApiSearchClient(SerpApiSearchClient):
    """Text live-search client with a persistent cache for searches and fetched pages.

    Used for ``retrieval_mode=serpapi_live`` rows (text OOD split) and uncovered
    WebWalkerQA test items. A re-run of the same checkpoint (temperature 0) replays
    from disk and spends no quota; other checkpoints reuse whatever queries/URLs overlap.
    """

    def __init__(self, api_keys: tuple[str, ...], *, cache_dir: str | Path | None = None, **kwargs) -> None:
        super().__init__(api_keys, **kwargs)
        self.disk_cache = SerpApiDiskCache(cache_dir, kinds=("search", "page"))

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
    def from_env(cls) -> "CachedSerpApiSearchClient":
        keys = parse_serpapi_keys(os.getenv("SERPAPI_API_KEYS") or os.getenv("SERPAPI_API_KEY"))
        if not keys:
            raise RuntimeError(
                "SERPAPI_API_KEYS is required for serpapi_live rows and uncovered WebWalkerQA test items"
            )
        return cls(
            keys,
            search_url=os.getenv("SERPAPI_SEARCH_URL", SERPAPI_SEARCH_URL),
            timeout=float(os.getenv("SERPAPI_TIMEOUT", "45")),
            max_results=int(os.getenv("SERPAPI_MAX_RESULTS", "10")),
            max_page_bytes=int(os.getenv("SERPAPI_MAX_PAGE_BYTES", "5000000")),
            cache_dir=os.getenv("SERPAPI_CACHE_DIR", DEFAULT_TEXT_SERPAPI_CACHE_DIR) or None,
            proxy_url=os.getenv("MM_OPEN_PAGE_PROXY"),
        )
