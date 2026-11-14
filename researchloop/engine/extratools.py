"""Web and fact-memory tools, ported from the ReEscape Protocol design.

Registered through the same registry as built-in tools; no dispatch chain edit
is needed to add a capability.
"""
from .. import facts, storage, webaccess
from . import registry

RESERVED_PREDICATES = {"user_preference", "user_note", "method_lesson"}


def _web_settings():
    return storage.settings()["web"]


def web_search(engine, args):
    config = _web_settings()
    if not config.get("enabled"):
        raise ValueError("联网工具未启用：请先在设置中启用并配置搜索服务")
    query = str(args.get("query") or "").strip()
    result = webaccess.search(query, provider=config.get("search_provider"),
                              endpoint=config.get("search_endpoint"), api_key=config.get("search_api_key"),
                              max_results=args.get("max_results") or config.get("max_results") or 5,
                              recency=args.get("recency") or "any")
    seen = {str(source.get("url") or "") for source in engine.run["sources"]}
    added = []
    for row in result["results"]:
        url = str(row.get("url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        row["id"] = f"p{len(engine.run['sources']) + 1}"
        row.setdefault("abstract", row.get("snippet", ""))
        row["evidence_scope"] = f"网页搜索摘要（{result['provider']}），媒体/平台内容，不是同行评议"
        engine.run["sources"].append(row)
        added.append(row)
    engine.run["evidence_collected"] = True
    return {"provider": result["provider"], "query": result["query"], "added": len(added),
            "skipped_duplicates": len(result["results"]) - len(added), "results": added,
            "note": "搜索摘要只作线索；需要正文时用 web_fetch 读取具体 URL。换关键词可继续检索，重复 URL 会被跳过。"}


def web_fetch(engine, args):
    config = _web_settings()
    if not config.get("enabled"):
        raise ValueError("联网工具未启用：请先在设置中启用并配置搜索服务")
    url = str(args.get("url") or "")
    url = webaccess.normalise_url(url) if url else url
    existing = next((source for source in engine.run["sources"]
                     if source.get("url") == url and source.get("content")), None)
    if existing and existing.get("truncated") is False:
        return {"id": existing["id"], "url": url, "title": existing["title"],
                "content": existing["content"], "chars": len(existing["content"]),
                "truncated": False, "reused": True,
                "note": "本轮已读取完整页面，直接复用已有正文；重复读取不是独立来源。"}
    page = webaccess.fetch(url, max_chars=args.get("max_chars") or config.get("fetch_max_chars") or 6000)
    existing = next((source for source in engine.run["sources"]
                     if source.get("url") == page["url"]), None)
    if existing and (not existing.get("content") or page["content"] == existing["content"] or
                     page["content"].startswith(existing["content"])):
        if not existing.get("content"):
            existing["search_evidence_scope"] = existing.get("evidence_scope")
            existing["evidence_scope"] = "网页正文"
        if not existing.get("content") or len(page["content"]) > len(existing["content"]):
            existing.update(content=page["content"], content_chars=len(page["content"]),
                            truncated=page["truncated"], fetched_at=page["fetched_at"])
            existing.setdefault("abstract", page["content"][:1800])
        engine.run["evidence_collected"] = True
        return {"id": existing["id"], "url": page["url"], "title": existing["title"],
                "content": existing["content"], "chars": len(existing["content"]),
                "truncated": existing.get("truncated", False), "reused": True,
                "note": "同一URL的搜索摘要或正文已存在；抓取结果更新原证据ID，不能作为另一独立来源。"}
    note = ""
    if len(page["content"]) < 200:
        note = "正文很短：页面可能需要登录或由脚本渲染，未获取到主要内容"
    entry = {"id": f"p{len(engine.run['sources']) + 1}", "title": page["title"], "url": page["url"],
             "abstract": page["content"][:1800], "content": page["content"], "evidence_scope": "网页正文",
             "fetched_at": page["fetched_at"], "content_chars": len(page["content"]),
             "truncated": page["truncated"]}
    engine.run["sources"].append(entry)
    engine.run["evidence_collected"] = True
    return {"id": entry["id"], "url": page["url"], "title": page["title"],
            "content": page["content"], "chars": len(page["content"]),
            "truncated": page["truncated"], "note": note}


def remember_fact(engine, args):
    ids = args.get("evidence_ids")
    allowed = set(engine.memory_evidence_ids())
    if not isinstance(ids, list) or not ids or any(identifier not in allowed for identifier in ids):
        raise ValueError("研究事实记忆必须绑定已有材料、含正文或摘要的来源或已复核实验的 evidence_ids；"
                         "只有元数据的来源和未核实判断不能写成事实。偏好可在记忆页面添加。")
    if str(args.get("predicate") or "").strip().casefold() in RESERVED_PREDICATES:
        raise ValueError("该谓词保留给用户在记忆页面添加的偏好、备注或方法经验，研究事实请使用描述关系的谓词")
    pending = {"predicate": str(args.get("predicate") or ""), "object": str(args.get("object") or ""),
               "summary": str(args.get("summary") or ""), "tags": args.get("tags"),
               "confidence": args.get("confidence") or 0.6, "evidence_ids": list(ids),
               "source_message": str(args.get("source_message") or engine.run["question"]),
               "status": "pending_report"}
    engine.run.setdefault("pending_facts", []).append(pending)
    return {"predicate": pending["predicate"], "object": pending["object"],
            "summary": pending["summary"], "status": pending["status"],
            "note": "报告完成并引用全部对应证据后才保存事实记忆。"}


def commit_report_facts(engine, citations):
    cited = set(citations)
    for pending in engine.run.get("pending_facts", []):
        if pending["status"] != "pending_report":
            continue
        if not set(pending["evidence_ids"]) <= cited:
            pending["status"] = "not_cited"
            continue
        try:
            fact = facts.remember(engine.run["id"], pending["predicate"], pending["object"],
                                  pending["summary"], tags=pending["tags"],
                                  confidence=pending["confidence"],
                                  source_message=pending["source_message"] + "\n证据ID：" +
                                  ", ".join(pending["evidence_ids"]))
            pending.update(status="saved", id=fact["id"])
        except Exception as error:
            pending.update(status="save_failed", error=str(error))


def get_memory_context(engine, args):
    from .. import memory
    query = str(args.get("query") or engine.question_text()).strip()
    if engine.run.get("memory_mode") == "disabled":
        return {"query": query, "facts": [], "method": None, "text": "", "note": "本轮显式关闭历史记忆召回"}
    engine.run["memory_context"] = memory.recall(query)
    return engine.run["memory_context"]


def platform_search(engine, args):
    from .. import platforms
    platform = str(args.get("platform") or "").strip().lower()
    query = str(args.get("query") or "").strip()
    from .. import reliability
    cfg = storage.settings()
    disabled = [key for key in engine.run.get("resource_health", {}) if reliability.blocked(engine.run, key, cfg)]
    result = platforms.search(platform, query, limit=args.get("limit") or 5, blocked_resources=disabled)
    for row in result["results"]:
        row["evidence_scope"] = f"{result['label']}平台检索结果（平台内容，不是同行评议）"
        row.setdefault("abstract", row.get("snippet", ""))
        engine.add_source(row)
    engine.run["evidence_collected"] = True
    return result


registry.register_tool("web_search", web_search,
                       description="搜索外部网页信息，适合最新消息、陌生实体或需要来源核实时使用。",
                       arguments={"query": "搜索关键词", "max_results": 5, "recency": "any"}, required=["query"])
registry.register_tool("web_fetch", web_fetch,
                       description="读取指定 http/https URL 的网页正文，直接返回 content；网页不在本地 workspace，不能用 read_file 读取。",
                       arguments={"url": "要读取的 URL", "max_chars": 6000}, required=["url"])
registry.register_tool("remember_fact", remember_fact,
                       description="基于已有证据ID暂存事实候选；完成报告引用全部证据后才写入记忆。必须附evidence_ids，不能把推测写成事实。",
                       arguments={"predicate": "事实谓词", "object": "事实对象", "summary": "简短摘要",
                                  "tags": ["标签"], "confidence": 0.6, "evidence_ids": ["p1"]}, required=["predicate", "object", "summary", "evidence_ids"])
registry.register_tool("get_memory_context", get_memory_context,
                       description="按查询召回事实记忆，返回适用条件与证据来源。",
                       arguments={"query": "检索文本"}, required=[])

registry.register_tool("platform_search", platform_search,
                       description="按平台检索（Bilibili、小红书、贴吧、微博、抖音、知乎）；公开接口优先，需登录的平台走站内限定搜索。",
                       arguments={"platform": "bilibili/xiaohongshu/tieba/weibo/douyin/zhihu",
                                  "query": "检索关键词", "limit": 5},
                       required=["platform", "query"])
