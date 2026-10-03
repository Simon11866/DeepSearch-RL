# -*- coding: utf-8 -*-
"""
搜索引擎后端
============

统一封装多家搜索服务，返回相同结构的结果列表。默认按可用 key 自动选择后端。
所有后端返回 ``List[SearchItem]``，再由 search.py 格式化为给模型的文本。

支持后端：
    serper   https://serper.dev   （便宜、稳定，推荐）
    serpapi  https://serpapi.com
    bing     Azure Bing Search v7
    brave    Brave Search API
    tavily   Tavily Search（面向 LLM）
    ddg      DuckDuckGo（免费，无需 key，仅建议调试/兜底）
    bing_web Bing 网页搜索（免费，无需 key；本机常返回无关结果）
    so_web   360 桌面搜索（免费；本机容易触发验证码）
    quark    夸克搜索（免费，无需 key）
    shenma   神马搜索（免费，无需 key）
    so_m     360 移动搜索（免费，无需 key）
    sogou_wx 搜狗微信搜索（免费，无需 key）
    toutiao  头条搜索（免费，无需 key）
    free     按 quark → so_m → shenma → sogou_wx → toutiao 依次尝试

只用标准库 urllib（避免额外依赖），超时与异常由上层统一分类、重试。
"""

from __future__ import annotations

import abc
import html
import json
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger("deepsearch_rl.search")

from .exceptions import ToolError, ToolErrorType, classify_exception, classify_http_status

# 浏览器 UA，降低被反爬概率
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


@dataclass
class SearchItem:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


def _http_request(
    url: str,
    *,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    payload: Optional[bytes] = None,
    timeout: float = 15.0,
) -> tuple:
    """发起 HTTP 请求，返回 (status_code, text)。把 HTTP 层错误转成 ToolError。"""
    req = urllib.request.Request(url, data=payload, method=method)
    req.add_header("User-Agent", _UA)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise classify_http_status(e.code, body)
    except Exception as e:  # 网络/超时类
        from .exceptions import classify_exception

        raise classify_exception(e)


class SearchProvider(abc.ABC):
    name: str = "base"

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    @abc.abstractmethod
    def search(self, query: str, key: Optional[str] = None, num: int = 5) -> List[SearchItem]:
        ...


class SerperProvider(SearchProvider):
    name = "serper"

    def search(self, query, key=None, num=5):
        if not key:
            raise ToolError(ToolErrorType.AUTH, "serper 需要 API key")
        body = json.dumps({"q": query, "num": num, "gl": "us", "hl": "en"}).encode()
        status, text = _http_request(
            "https://google.serper.dev/search",
            method="POST",
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            payload=body,
            timeout=self.timeout,
        )
        data = json.loads(text)
        items: List[SearchItem] = []
        for r in data.get("organic", [])[:num]:
            items.append(
                SearchItem(
                    title=r.get("title", ""),
                    url=r.get("link", ""),
                    snippet=r.get("snippet", ""),
                )
            )
        return items


class SerpApiProvider(SearchProvider):
    name = "serpapi"

    def search(self, query, key=None, num=5):
        if not key:
            raise ToolError(ToolErrorType.AUTH, "serpapi 需要 API key")
        params = urllib.parse.urlencode(
            {"q": query, "api_key": key, "engine": "google", "num": num}
        )
        status, text = _http_request(
            f"https://serpapi.com/search.json?{params}", timeout=self.timeout
        )
        data = json.loads(text)
        if "error" in data:
            raise ToolError(ToolErrorType.QUOTA_EXHAUSTED, str(data["error"]))
        items = []
        for r in data.get("organic_results", [])[:num]:
            items.append(
                SearchItem(
                    title=r.get("title", ""),
                    url=r.get("link", ""),
                    snippet=r.get("snippet", ""),
                )
            )
        return items


class BingProvider(SearchProvider):
    name = "bing"

    def search(self, query, key=None, num=5):
        if not key:
            raise ToolError(ToolErrorType.AUTH, "bing 需要 API key")
        params = urllib.parse.urlencode({"q": query, "count": num})
        status, text = _http_request(
            f"https://api.bing.microsoft.com/v7.0/search?{params}",
            headers={"Ocp-Apim-Subscription-Key": key},
            timeout=self.timeout,
        )
        data = json.loads(text)
        items = []
        for r in data.get("webPages", {}).get("value", [])[:num]:
            items.append(
                SearchItem(
                    title=r.get("name", ""),
                    url=r.get("url", ""),
                    snippet=r.get("snippet", ""),
                )
            )
        return items


class BingWebProvider(SearchProvider):
    """Bing 网页结果，不使用 Azure API key。"""

    name = "bing_web"

    def search(self, query, key=None, num=5):
        params = urllib.parse.urlencode({"q": query, "count": max(num, 1)})
        status, text = _http_request(
            f"https://www.bing.com/search?{params}",
            timeout=self.timeout,
        )
        if status != 200:
            raise classify_http_status(status, text[:300])
        return _parse_bing_html(text, num)


class SoWebProvider(SearchProvider):
    """360 搜索网页结果，不需要 API key。"""

    name = "so_web"

    def search(self, query, key=None, num=5):
        params = urllib.parse.urlencode({"q": query})
        status, text = _http_request(
            f"https://www.so.com/s?{params}",
            timeout=self.timeout,
        )
        if status != 200:
            raise classify_http_status(status, text[:300])
        if (
            "antispider" in text[:800].lower()
            or "验证码" in text[:1200]
            or "访问异常" in text[:1200]
        ):
            raise ToolError(ToolErrorType.RATE_LIMIT, "360 搜索触发反爬验证")
        return _parse_so_html(text, num)


def _strip_html(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", " ", fragment or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _parse_bing_html(text: str, num: int) -> List[SearchItem]:
    items: List[SearchItem] = []
    pattern = re.compile(
        r'<li class="b_algo"[\s\S]*?<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>'
        r"[\s\S]*?</h2>([\s\S]*?)(?=<li class=\"b_algo\"|</ol>)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text or ""):
        url, title_html, tail = match.group(1), match.group(2), match.group(3)
        title = _strip_html(title_html)
        if not title or not url.startswith("http"):
            continue
        snippet_match = re.search(r"<p[^>]*>([\s\S]*?)</p>", tail, re.IGNORECASE)
        snippet = _strip_html(snippet_match.group(1) if snippet_match else "")
        items.append(SearchItem(title=title, url=url, snippet=snippet))
        if len(items) >= num:
            break
    return items


def _parse_so_html(text: str, num: int) -> List[SearchItem]:
    """解析 360 搜索结果。真实链接在 data-mdurl，摘要在 p.res-desc。"""
    items: List[SearchItem] = []
    for block in re.split(r'<li class="res-list"', text or "")[1:]:
        title_match = re.search(
            r'<h3 class="res-title[^"]*"[^>]*>\s*<a([^>]*)>([\s\S]*?)</a>',
            block,
            re.IGNORECASE,
        )
        if not title_match:
            continue
        attrs, title_html = title_match.group(1), title_match.group(2)
        title = _strip_html(title_html)
        url_match = re.search(r'data-mdurl="([^"]+)"', attrs)
        if url_match is None:
            url_match = re.search(r'href="(https?://[^"]+)"', attrs)
        if not title or url_match is None:
            continue
        url = html.unescape(url_match.group(1))
        if not url.startswith("http") or "so.com/link" in url:
            continue
        snippet_match = re.search(
            r'<p class="res-desc"[^>]*>([\s\S]*?)</p>', block, re.IGNORECASE
        )
        snippet = _strip_html(snippet_match.group(1) if snippet_match else "")
        items.append(SearchItem(title=title, url=url, snippet=snippet))
        if len(items) >= num:
            break
    return items


class BraveProvider(SearchProvider):
    name = "brave"

    def search(self, query, key=None, num=5):
        if not key:
            raise ToolError(ToolErrorType.AUTH, "brave 需要 API key")
        params = urllib.parse.urlencode({"q": query, "count": num})
        status, text = _http_request(
            f"https://api.search.brave.com/res/v1/web/search?{params}",
            headers={"X-Subscription-Token": key, "Accept": "application/json"},
            timeout=self.timeout,
        )
        data = json.loads(text)
        items = []
        for r in data.get("web", {}).get("results", [])[:num]:
            items.append(
                SearchItem(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("description", ""),
                )
            )
        return items


class TavilyProvider(SearchProvider):
    name = "tavily"

    def search(self, query, key=None, num=5):
        if not key:
            raise ToolError(ToolErrorType.AUTH, "tavily 需要 API key")
        body = json.dumps(
            {"query": query, "max_results": num, "search_depth": "basic"}
        ).encode()
        status, text = _http_request(
            "https://api.tavily.com/search",
            method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            payload=body,
            timeout=self.timeout,
        )
        data = json.loads(text)
        items = []
        for r in data.get("results", [])[:num]:
            items.append(
                SearchItem(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                )
            )
        return items


class DuckDuckGoProvider(SearchProvider):
    """免费兜底：优先使用 ddgs 包；未安装时解析 html.duckduckgo.com。"""

    name = "ddg"

    def search(self, query, key=None, num=5):
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(self._search_impl, query, num)
            try:
                return fut.result(timeout=max(float(self.timeout), 15.0))
            except concurrent.futures.TimeoutError:
                raise ToolError(ToolErrorType.TIMEOUT, f"ddg 搜索超时: {query!r}")

    def _search_impl(self, query, num):
        last_exc: Optional[Exception] = None
        try:
            from ddgs import DDGS  # type: ignore

            items = []
            with DDGS(timeout=8) as ddgs:
                for r in ddgs.text(query, max_results=num):
                    items.append(
                        SearchItem(
                            title=r.get("title", "") or "",
                            url=r.get("href", "") or r.get("url", "") or "",
                            snippet=r.get("body", "") or r.get("snippet", "") or "",
                        )
                    )
            if items:
                return items
        except ImportError:
            last_exc = None
        except Exception as exc:  # noqa: BLE001 - 转成 ToolError 或走 HTML 兜底
            last_exc = classify_exception(exc)

        # HTML 兜底
        try:
            params = urllib.parse.urlencode({"q": query})
            status, text = _http_request(
                f"https://html.duckduckgo.com/html/?{params}", timeout=self.timeout
            )
            import re

            items = []
            for block in re.findall(
                r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
                r'[\s\S]*?<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                text,
            ):
                url, title, snippet = block
                url = urllib.parse.unquote(re.sub(r"^//duckduckgo.com/l/\?uddg=", "", url))
                title = re.sub(r"<[^>]+>", "", title)
                snippet = re.sub(r"<[^>]+>", "", snippet)
                items.append(SearchItem(title=title, url=url, snippet=snippet))
                if len(items) >= num:
                    break
            if items:
                return items
        except Exception as exc:  # noqa: BLE001
            last_exc = classify_exception(exc)

        if last_exc is not None:
            raise last_exc
        return []


# 本机实测能返回与查询相关摘要的免费引擎，按优先顺序。
# 先用 Tavily。它不可用时再换免费网页搜索。
FREE_SEARCH_CHAIN = ("tavily", "quark", "so_m", "shenma", "sogou_wx", "toutiao")

KEYLESS_BACKENDS = {
    "ddg",
    "bing_web",
    "so_web",
    "quark",
    "shenma",
    "so_m",
    "sogou_wx",
    "toutiao",
    "free",
}


def _page_blocked(text: str) -> bool:
    head = (text or "")[:1500].lower()
    return any(
        mark in head
        for mark in (
            "qcaptcha",
            "antispider",
            "验证码",
            "访问异常",
            "安全验证",
            '"action":"captcha"',
            "_____tmd_____/punish",
        )
    )


def _abs_url(url: str, base: str) -> str:
    url = html.unescape(url or "").strip()
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/"):
        return base.rstrip("/") + url
    return url


class _HtmlSearchProvider(SearchProvider):
    """抓取一个搜索页并解析出结果。"""

    page_url = ""
    mobile = False

    def search(self, query, key=None, num=5):
        params = urllib.parse.urlencode({"q": query})
        url = self.page_url.format(query=urllib.parse.quote(query), params=params)
        headers = {}
        if self.mobile:
            headers["User-Agent"] = (
                "Mozilla/5.0 (Linux; Android 12; Pixel 6) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0.0.0 Mobile Safari/537.36"
            )
        status, text = _http_request(url, headers=headers, timeout=self.timeout)
        if status != 200:
            raise classify_http_status(status, text[:300])
        if _page_blocked(text):
            raise ToolError(ToolErrorType.RATE_LIMIT, f"{self.name} 触发反爬验证")
        return self.parse(text, num)

    def parse(self, text: str, num: int) -> List[SearchItem]:
        raise NotImplementedError


def _parse_qk_html(text: str, num: int) -> List[SearchItem]:
    """夸克 / 神马共用的结果卡片。"""
    items: List[SearchItem] = []
    for block in re.split(r'class="sc sc_structure_template_normal"', text or "")[1:]:
        title_match = re.search(
            r'class="qk-title-text[^"]*"[^>]*>([\s\S]*?)</div>', block, re.IGNORECASE
        )
        snippet_match = re.search(
            r'class="qk-paragraph-text"[^>]*>([\s\S]*?)</div>', block, re.IGNORECASE
        )
        title = _strip_html(title_match.group(1) if title_match else "")
        snippet = _strip_html(snippet_match.group(1) if snippet_match else "")
        url_match = re.search(r'href="(https?://[^"]+)"', block)
        url = html.unescape(url_match.group(1)) if url_match else ""
        if len(snippet) < 8 or not url.startswith("http"):
            continue
        if not title:
            title = snippet[:48]
        items.append(SearchItem(title=title, url=url, snippet=snippet))
        if len(items) >= num:
            break
    return items


class QuarkProvider(_HtmlSearchProvider):
    name = "quark"
    page_url = "https://quark.sm.cn/s?q={query}"

    def parse(self, text: str, num: int) -> List[SearchItem]:
        return _parse_qk_html(text, num)


class ShenmaProvider(_HtmlSearchProvider):
    name = "shenma"
    page_url = "https://m.sm.cn/s?q={query}"

    def parse(self, text: str, num: int) -> List[SearchItem]:
        return _parse_qk_html(text, num)


class SoMobileProvider(_HtmlSearchProvider):
    name = "so_m"
    page_url = "https://m.so.com/s?q={query}"
    mobile = True

    def parse(self, text: str, num: int) -> List[SearchItem]:
        items: List[SearchItem] = []
        for block in re.split(r'class="g-card res-list', text or "")[1:]:
            title_match = re.search(
                r'<h3 class="res-title">([\s\S]*?)</h3>', block, re.IGNORECASE
            )
            if not title_match:
                continue
            title = _strip_html(title_match.group(1))
            snippet_match = re.search(
                r'<p class="[^"]*summary[^"]*"[^>]*>([\s\S]*?)</p>', block, re.IGNORECASE
            )
            snippet = _strip_html(snippet_match.group(1) if snippet_match else "")
            url_match = re.search(r'data-pcurl="([^"]+)"', block)
            url = html.unescape(url_match.group(1)) if url_match else ""
            if len(snippet) < 8 or not title or not url.startswith("http"):
                continue
            items.append(SearchItem(title=title, url=url, snippet=snippet))
            if len(items) >= num:
                break
        return items


class SogouWeixinProvider(_HtmlSearchProvider):
    name = "sogou_wx"
    page_url = "https://weixin.sogou.com/weixin?type=2&query={query}"

    def parse(self, text: str, num: int) -> List[SearchItem]:
        items: List[SearchItem] = []
        pattern = re.compile(
            r'<h3[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>'
            r'[\s\S]*?<p class="txt-info"[^>]*>([\s\S]*?)</p>',
            re.IGNORECASE,
        )
        for href, title_html, snippet_html in pattern.findall(text or ""):
            title = _strip_html(title_html)
            snippet = _strip_html(snippet_html)
            url = _abs_url(href, "https://weixin.sogou.com")
            if len(snippet) < 8 or not title or not url.startswith("http"):
                continue
            items.append(SearchItem(title=title, url=url, snippet=snippet))
            if len(items) >= num:
                break
        return items


def _toutiao_target(href: str) -> str:
    href = html.unescape(href or "")
    match = re.search(r"(?:\?|&)url=([^&]+)", href)
    if not match:
        return ""
    outer = urllib.parse.unquote(match.group(1))
    inner = re.search(r"[?&]h5_url=([^&]+)", outer)
    if inner:
        return urllib.parse.unquote(urllib.parse.unquote(inner.group(1)))
    return outer if outer.startswith("http") else ""


class ToutiaoProvider(_HtmlSearchProvider):
    name = "toutiao"
    page_url = "https://so.toutiao.com/search?keyword={query}"

    def parse(self, text: str, num: int) -> List[SearchItem]:
        items: List[SearchItem] = []
        parts = re.split(r'<a href="(/search/jump\?[^"]+)"', text or "")
        for index in range(1, len(parts), 2):
            url = _toutiao_target(parts[index])
            body = parts[index + 1] if index + 1 < len(parts) else ""
            title_match = re.search(
                r'line-clamp-1[\s\S]{0,240}?>([\s\S]*?)</(?:div|span)>', body, re.IGNORECASE
            )
            snippet_match = re.search(
                r'line-clamp-[23][\s\S]{0,200}?>([\s\S]*?)</(?:div|span)>', body, re.IGNORECASE
            )
            title = _strip_html(title_match.group(1) if title_match else "")
            snippet = _strip_html(snippet_match.group(1) if snippet_match else "")
            if len(snippet) < 12:
                continue
            if not title:
                title = snippet[:48]
            if not url.startswith("http"):
                continue
            items.append(SearchItem(title=title, url=url, snippet=snippet))
            if len(items) >= num:
                break
        return items


_LATIN_STOP = {
    "a", "an", "the", "of", "and", "or", "in", "on", "to", "for", "from", "by",
    "with", "what", "when", "where", "who", "which", "how", "is", "was", "were",
    "are", "year", "years", "height", "meter", "meters", "metre", "metres",
    "capital", "building", "completed", "complete", "built", "founded", "born",
    "died", "name", "called", "that", "this", "into", "over", "after", "before",
}


def _focus_terms(query: str) -> List[str]:
    """抽出查询里最有辨识度的词，用来丢掉对不上的结果。"""
    latin = [
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9]{3,}", query or "")
        if word.lower() not in _LATIN_STOP
    ]
    if latin:
        return [max(latin, key=len)]
    text = "".join(re.findall(r"[\u4e00-\u9fff]{2,}", query or ""))
    for ch in "的了是在有吗呢吧啊和与及或被把将":
        text = text.replace(ch, "")
    for stop in ("什么", "哪里", "哪年", "是谁", "多高", "高度", "哪个", "多少", "如何"):
        text = text.replace(stop, "")
    text = text.strip()
    if len(text) >= 2:
        return [text[:4]]
    return []


def _keep_relevant(items: List[SearchItem], query: str, num: int) -> List[SearchItem]:
    terms = _focus_terms(query)
    kept: List[SearchItem] = []
    for item in items:
        blob = f"{item.title} {item.snippet}".lower()
        if terms and not any(term.lower() in blob for term in terms):
            continue
        if len((item.snippet or "").strip()) < 8:
            continue
        kept.append(item)
        if len(kept) >= num:
            break
    return kept


class FallbackProvider(SearchProvider):
    """一个引擎失败、被反爬或没有对上查询的结果时，换下一个免费引擎。"""

    name = "free"

    def __init__(self, timeout: float = 15.0, chain=FREE_SEARCH_CHAIN) -> None:
        super().__init__(timeout=timeout)
        self.chain = tuple(chain)
        self.last_engine = ""
        self.attempts: List[str] = []

    def search(self, query, key=None, num=5):
        self.attempts = []
        errors: List[str] = []
        for name in self.chain:
            provider = PROVIDERS[name](timeout=self.timeout)
            try:
                items = provider.search(query, key=key, num=max(num, 8))
            except ToolError as exc:
                note = f"{name}:{exc.error_type.value}"
                self.attempts.append(note)
                errors.append(note)
                logger.info("搜索引擎 %s 失败，换下一个: %s", name, exc.error_type.value)
                continue
            items = _keep_relevant(items, query, num)
            if not items:
                note = f"{name}:empty"
                self.attempts.append(note)
                errors.append(note)
                logger.info("搜索引擎 %s 没有对上查询的结果，换下一个", name)
                continue
            self.last_engine = name
            self.attempts.append(f"{name}:ok")
            return items
        raise ToolError(
            ToolErrorType.UNKNOWN,
            "免费搜索引擎都失败: " + "; ".join(errors),
        )


PROVIDERS: Dict[str, type] = {
    "serper": SerperProvider,
    "serpapi": SerpApiProvider,
    "bing": BingProvider,
    "brave": BraveProvider,
    "tavily": TavilyProvider,
    "ddg": DuckDuckGoProvider,
    "bing_web": BingWebProvider,
    "so_web": SoWebProvider,
    "quark": QuarkProvider,
    "shenma": ShenmaProvider,
    "so_m": SoMobileProvider,
    "sogou_wx": SogouWeixinProvider,
    "toutiao": ToutiaoProvider,
    "free": FallbackProvider,
}


PROBE_QUERY = "capital of France"


def probe_engine(name: str, timeout: float = 8.0, key: Optional[str] = None) -> tuple:
    """探测一个引擎现在能不能返回对上查询的结果。返回 (ok, detail)。"""
    provider = PROVIDERS[name](timeout=timeout)
    try:
        items = provider.search(PROBE_QUERY, key=key, num=2)
    except ToolError as exc:
        return False, exc.error_type.value
    if not _keep_relevant(items, PROBE_QUERY, 1):
        return False, "empty"
    return True, "ok"


def build_provider(name: str, timeout: float = 15.0) -> SearchProvider:
    name = (name or "ddg").lower()
    if name not in PROVIDERS:
        raise ToolError(
            ToolErrorType.BAD_REQUEST,
            f"未知搜索后端: {name}，可选 {list(PROVIDERS)}",
        )
    return PROVIDERS[name](timeout=timeout)


__all__ = [
    "SearchItem",
    "SearchProvider",
    "PROVIDERS",
    "FREE_SEARCH_CHAIN",
    "build_provider",
    "probe_engine",
]
