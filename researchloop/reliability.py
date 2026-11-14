"""Bounded recovery policy shared by model requests and read-only tools."""
import socket
import time
import urllib.error
from email.utils import parsedate_to_datetime


class Failure(ValueError):
    def __init__(self, message, *, resource="", kind="validation", status=None,
                 retryable=False, repairable=True, retry_after=.25, blocked=False):
        super().__init__(message)
        self.info = {"resource": resource, "kind": kind, "http_status": status,
                     "retryable": retryable, "repairable": repairable,
                     "retry_after": retry_after, "blocked": blocked, "message": message}


def classify(error, resource=""):
    if isinstance(error, Failure):
        return dict(error.info)
    cause = error
    visited = set()
    while cause is not None and id(cause) not in visited:
        visited.add(id(cause))
        if isinstance(cause, Failure):
            return dict(cause.info)
        if isinstance(cause, urllib.error.HTTPError):
            code = cause.code
            delay = .25
            raw = (cause.headers or {}).get("Retry-After")
            if raw:
                try:
                    delay = max(0, float(raw))
                except (TypeError, ValueError):
                    try:
                        delay = max(0, parsedate_to_datetime(raw).timestamp() - time.time())
                    except (TypeError, ValueError, OverflowError):
                        pass
            auth = code in (401, 403)
            retryable = code in (408, 429) or 500 <= code <= 599
            kind = "authentication" if auth else "rate_limit" if code == 429 else "temporary_http" if retryable else "http_error"
            message = (f"HTTP {code}：鉴权或权限失败；相同配置不再重试，请检查服务凭据，或选择其他证据来源。" if auth
                       else f"HTTP {code}：服务限流或暂时不可用。" if retryable else f"HTTP {code}：请求被服务拒绝。")
            if resource:
                message = resource + " · " + message
            return Failure(message, resource=resource, kind=kind, status=code, retryable=retryable,
                           blocked=auth, retry_after=delay).info
        if isinstance(cause, (TimeoutError, socket.timeout, ConnectionError, urllib.error.URLError)):
            return Failure("网络超时或连接失败，可在预算内有限重试。", resource=resource,
                           kind="network", retryable=True).info
        cause = cause.__cause__ or cause.__context__
    text = str(error)
    configuration = resource.startswith("search:") and any(part in text for part in ("API Key", "未配置", "未启用", "搜索提供方"))
    return Failure(text, resource=resource, kind="configuration" if configuration else "validation", blocked=configuration).info


def wrap(error, resource):
    info = classify(error, resource)
    return Failure(info["message"], resource=info["resource"], kind=info["kind"], status=info["http_status"],
                   retryable=info["retryable"], repairable=info["repairable"],
                   retry_after=info["retry_after"], blocked=info["blocked"])


def blocked(run, resource, config):
    item = run.get("resource_health", {}).get(resource) or {}
    return (item.get("blocked", False) or item.get("cooldown_until", 0) > time.time()) and item.get("revision") == config.get("revision")


def record(run, info, config, turn):
    if (info.get("blocked") or info.get("retryable")) and info.get("resource"):
        run.setdefault("resource_health", {})[info["resource"]] = {
            **info, "revision": config.get("revision"), "turn": turn,
            "cooldown_until": time.time() + info.get("retry_after", 0) if info.get("retryable") else 0}
    run["recovery_feedback"] = info


def resource(tool, args, config):
    if tool == "web_search":
        return "search:" + str(config["web"].get("search_provider") or "searxng")
    if tool == "platform_search":
        name = str(args.get("platform") or "")
        return "platform:" + name if name in ("bilibili", "tieba") else "search:" + str(config["web"].get("search_provider") or "searxng")
    if tool == "web_fetch":
        from urllib.parse import urlparse
        return "fetch:" + (urlparse(str(args.get("url") or "")).hostname or "")
    return "literature:crossref" if tool == "literature" else "tool:" + str(tool)


READ_TOOLS = {"web_search", "platform_search", "web_fetch", "literature", "read", "read_pdf", "read_docx", "read_pptx"}


def check_search_config(run, config, turn):
    """Known invalid configuration is a capability gap, not a failed HTTP call."""
    web = config.get("web") or {}
    if not web.get("enabled"):
        return
    provider = str(web.get("search_provider") or "searxng")
    resource = "search:" + provider
    if blocked(run, resource, config):
        return
    try:
        from .webaccess import validate_search_config
        validate_search_config(provider, web.get("search_endpoint"))
        if provider == "tavily" and not web.get("search_api_key"):
            raise ValueError("未配置 Tavily 搜索 API Key；云端模型的 Key 不能替代搜索服务 Key。")
        if provider == "searxng" and not web.get("search_endpoint"):
            raise ValueError("未配置 SearXNG 搜索服务地址。")
    except ValueError as error:
        record(run, Failure(str(error), resource=resource, kind="configuration", blocked=True).info, config, turn)
