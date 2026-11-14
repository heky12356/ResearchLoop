"""OpenAI-compatible cloud chat client.

Cloud inference is opt-in: nothing here runs unless the user enables a cloud
model in settings and a run explicitly selects `model_source = "cloud"`.
The API key is never written to logs, run records or the state endpoint.
"""
import json
import time
import urllib.parse
import urllib.request

from .ollama import BudgetError
from . import llmjson

JSON_OBJECT = {"response_format": {"type": "json_object"}}


class TruncatedCompletionError(RuntimeError):
    """The provider exhausted its output limit before returning a complete answer."""


def normalise(base_url):
    value = str(base_url or "").strip().rstrip("/")
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("云端服务地址必须是 http(s) URL，并包含主机名")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("云端服务地址不能包含用户名、密码、查询参数或片段")
    return value


def headers(api_key):
    value = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        value["Authorization"] = "Bearer " + str(api_key)
    return value


def probe(base_url, api_key, timeout=8):
    try:
        url = normalise(base_url)
        request = urllib.request.Request(url + "/models", headers=headers(api_key))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        models = [{"name": item.get("id") or item.get("name")}
                  for item in payload.get("data", []) if (item.get("id") or item.get("name"))]
        return {"available": True, "models": models[:200], "error": None, "base_url": url}
    except Exception as exc:
        return {"available": False, "models": [], "error": str(exc), "base_url": str(base_url or "")}


class Cloud:
    """Minimal chat-completions client with the same budget accounting as Ollama."""

    def __init__(self, cloud, model, budget, on_call=None):
        self.base_url = normalise(cloud.get("base_url"))
        self.api_key = str(cloud.get("api_key") or "")
        self.model = str(model)
        self.budget = budget
        self.on_call = on_call

    def call(self, system, prompt, schema, purpose, seed=None, max_output=4096, reserve=None):
        reserve = reserve or 6144 + max_output
        self.budget.reserve(reserve)
        body = {"model": self.model, "temperature": 0, "stream": False,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": prompt}],
                "max_tokens": max_output}
        if schema:
            body.update(JSON_OBJECT)
            body["messages"][0]["content"] += "\n严格遵守此JSON Schema，只返回JSON：" + json.dumps(schema, ensure_ascii=False)
        started = time.perf_counter()
        item = {"purpose": purpose, "model": self.model, "source": "cloud",
                "cloud": True, "reserved_upper_bound": reserve, "request": body}
        try:
            request = urllib.request.Request(self.base_url + "/chat/completions",
                                             data=json.dumps(body).encode(), headers=headers(self.api_key))
            left = self.budget.seconds_left()
            timeout = 180 if left is None else max(1, min(180, left))
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            usage = result.get("usage") or {}
            if not isinstance(usage.get("prompt_tokens"), int) or not isinstance(usage.get("completion_tokens"), int):
                raise ValueError("云端模型未返回完整 token usage，已按预留量记账并停止")
            details = usage.get("prompt_tokens_details") or {}
            cached = details.get("cached_tokens")
            if cached is None:
                cached = usage.get("prompt_cache_hit_tokens", 0)
            item.update(input_tokens=usage["prompt_tokens"], output_tokens=usage["completion_tokens"],
                        cached_tokens=int(cached or 0), context_tokens=usage["prompt_tokens"],
                        cloud_cost=None, status="ok")
            item["response"] = result
            choice = (result.get("choices") or [{}])[0]
            if choice.get("finish_reason") == "length":
                raise TruncatedCompletionError(
                    "云端模型输出达到长度上限（finish_reason=length），JSON 尚未完成；须重新生成完整的简短回答")
            content = (choice.get("message") or {}).get("content", "")
            parsed = llmjson.parse(content)
        except Exception as exc:
            item.update(status="error", error=str(exc))
            if "input_tokens" not in item:
                item["unknown_reserved"] = reserve
            if isinstance(exc, TruncatedCompletionError):
                raise
            raise RuntimeError(f"云端模型调用失败：{exc}") from exc
        finally:
            item["wall_seconds"] = round(time.perf_counter() - started, 3)
            self.budget.record(item)
            if self.on_call:
                self.on_call(item)
        if self.budget.token_limit is not None and self.budget.used > self.budget.token_limit:
            raise BudgetError("模型实际 usage 超过预留上界，停止后续调用")
        return parsed, item
