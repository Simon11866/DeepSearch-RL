# -*- coding: utf-8 -*-
"""训练步开始前，向检索服务确认当前搜索引擎可用。"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


def ensure_search_ready(
    base_url: str | None = None,
    retries: int = 3,
    pause: float = 5.0,
) -> dict:
    """调用检索服务的 /search_ready。

    当前引擎能搜到结果就继续用它。否则服务会改用下一个可用引擎。
    五个都不可用时重试几次，仍然不行就抛错，这一步不开始。
    """
    root = (base_url or os.environ.get("RETRIEVAL_SERVICE_URL") or "http://127.0.0.1:8000").rstrip("/")
    url = root + "/search_ready"
    last = "检索服务没有响应"
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=70) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("ok"):
                return payload
            last = json.dumps(payload, ensure_ascii=False)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last = f"HTTP {exc.code} {body[:400]}"
        except Exception as exc:  # 网络超时或服务未起
            last = f"{type(exc).__name__}: {exc}"
        if attempt + 1 < retries:
            time.sleep(pause)
    raise RuntimeError(f"搜索引擎都不可用，本步不开始: {last}")
