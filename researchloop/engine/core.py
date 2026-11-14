"""A model decides the next research action from observations, not task names."""
import hashlib
import json
import base64
import copy
import functools
import math
import re
import secrets
import threading
import time
import urllib.request
import urllib.parse
import uuid
from pathlib import Path

from researchloop import activity, storage, db, documents
from researchloop import cloudllm
from researchloop.engine import registry
from researchloop.engine import roles
from researchloop.engine import workflow as workflow_module
from researchloop.engine import extratools  # registers web_search/web_fetch/remember_fact/get_memory_context
from researchloop.engine import source_assessment
from researchloop.engine import doctools  # noqa: F401  registers read_docx/read_pptx/read_pdf
from researchloop import memory, reliability, llmjson, reporting
from researchloop.ollama import Budget, BudgetError
from . import tools
from . import protocol, routing, history
from .permissions import PermissionGate
from . import engineering, delegation

EVIDENCE_TOOLS = {"inspect", "literature", "web_search", "web_fetch", "platform_search",
                  "read", "read_pdf", "read_docx", "read_pptx"}
GAP_FILL_MIN = 3

SYSTEM = """你是执行科研的Agent。输入是开放研究问题，而不是预设算子任务。
你根据工具返回的真实证据决定下一步：检查材料/检索，注册可证伪假设，设计并运行Python实验，独立重跑验证，再根据结果修正或结束。
每次只调用一个工具；不预设结果，不编造文献和数据。工具失败时根据错误修复代码或换方法；已经成功的工具不要无意义重复。
没有用户数据时可以设计数值模拟，但必须明确模拟假设，不声称真实世界结论；不能以模拟替代原问题所需的真实实验或数据。文献元数据不等于论文结论。
先判断问题需要哪类证据。需要外部事实或真实世界效果时，缺少证据须换关键词/平台、抓正文、查文献或读材料；若问题可由本地计算或模拟检验，可先plan设计可证伪假设、指标与实验，再实际运行和独立复核，无须先检索。模拟结论仅限于所设机制。
plan、hypothesis、experiment、verify、finish是本地实验的常见顺序；材料或外部事实问题可按需检索、读取、提出新假设或修正失败实验。
Python只能用np(numpy)、math、statistics；随机数必须用提供的rng。提供load_csv('a1')返回字符串字典列表、load_text('a1')返回文本。
实验必须实际计算并emit_result({'metrics': {'预注册指标名': float(实测值)}, 'notes':'方法说明'})。不能填写预想常数冒充实测值。
不导入os/sys/文件/网络，不使用np.random或自定固定随机种子；命名空间已提供 rng 变量，直接用 rng.normal/rng.uniform，不要为 rng 重新赋值或新建生成器；verify会用新的隐藏种子运行同一份代码。只写精简代码。
假设必须给出statement,prediction,controls,metric,relation(gt/ge/lt/le),threshold。实验必须关联hypothesis_id。
可以证伪假设；复核未通过阈值是有效负结果。你不能修改已经注册的阈值或冻结的复核代码。
finish是最后一步：有可引用证据时写出结论、局限、引用IDs和lesson；确实取不到证据时，说明已尝试的来源与缺口并明确未完成。每次运行都必须以报告与终态结束，不能停在“不可行判断”。
自主选择下一项工具；严格返回 {"tool":"工具名","reason":"选择依据","args":{工具参数}}，不要把参数直接放在顶层。定性和文献问题可以采用证据分析，无须强行运行数值实验。
异常或玩笑式提问也应先核实前提、语境或出处，不能在取得证据前直接认定它是梗或无研究价值。工作流目标也是待检验设想，不是已知事实。
记忆和错误日志不是研究证据；检索来源先用 assess_source 依据当前问题核对原文、判断适用性和局限，再在 finish 引用。标题只能支持标题所能证明的事项。finish.citations只能从allowed_citation_ids选择，没有证据时用[]明确报告未完成。401/403应换来源或修复配置，不能只改关键词重复访问同一服务。"""

ACTION_SCHEMA = {"type": "object", "properties": {
    "tool": {"type": "string", "enum": [t["name"] for t in tools.TOOLS]}, "reason": {"type": "string"},
    "args": {"type": "object", "properties": {
        **{key: {"type": "string"} for key in ("query", "asset_id", "statement", "prediction", "controls", "metric", "hypothesis_id", "experiment_id", "code", "conclusion", "lesson")},
        "relation": {"type": "string", "enum": ["gt", "ge", "lt", "le"]}, "threshold": {"type": "number"},
        "citations": {"type": "array", "items": {"type": "string"}}}}},
    "required": ["tool", "reason", "args"]}

ARGUMENTS = {
    **engineering.ARGUMENTS, **delegation.ARGUMENTS,
    "plan": ({"feasible": {"type": "boolean"}, **{k: {"type": "string"} for k in ("method", "limitations", "literature_query")},
              "research_kind": {"type": "string", "enum": ["quantitative", "simulation", "qualitative", "literature", "mixed"]}},
             ["feasible", "method", "limitations", "literature_query"]),
    "inspect": ({}, []),
    "literature": ({"query": {"type": "string"}}, ["query"]),
    "read": ({"asset_id": {"type": "string"}}, ["asset_id"]),
    "hypothesis": ({**{k: {"type": "string"} for k in ("statement", "prediction", "controls", "metric")},
                    "relation": {"type": "string", "enum": ["gt", "ge", "lt", "le"]}, "threshold": {"type": "number"}},
                   ["statement", "prediction", "controls", "metric", "relation", "threshold"]),
    "experiment": ({"hypothesis_id": {"type": "string"}, "code": {"type": "string"}}, ["hypothesis_id", "code"]),
    "verify": ({"experiment_id": {"type": "string"}}, ["experiment_id"]),
    "reflect": ({"next_action": {"type": "string", "enum": ["revise", "finish"]}, "reason": {"type": "string"}}, ["next_action", "reason"]),
    "finish": ({"conclusion": {"type": "string"}, "citations": {"type": "array", "items": {"type": "string"}},
                "lesson": {"type": "string"}, "manuscript": reporting.MANUSCRIPT_SCHEMA}, ["conclusion", "citations", "lesson"]),
    "web_search": ({"query": {"type": "string"},
                    "max_results": {"type": "integer"}, "recency": {"type": "string", "enum": ["any", "day", "week", "month", "year"]}},
                   ["query"]),
    "web_fetch": ({"url": {"type": "string"}, "max_chars": {"type": "integer"}}, ["url"]),
    "assess_source": ({"source_id": {"type": "string"},
                       "verdict": {"type": "string", "enum": ["supports", "context", "reject"]},
                       **{key: {"type": "string"} for key in ("quote", "claim", "reason", "limitation")}},
                      ["source_id", "verdict", "claim", "reason", "limitation"]),
    "remember_fact": ({"predicate": {"type": "string"}, "object": {"type": "string"}, "summary": {"type": "string"},
                       "tags": {"type": "array", "items": {"type": "string"}}, "confidence": {"type": "number"},
                       "source_message": {"type": "string"}}, ["predicate", "object", "summary"]),
    "platform_search": ({"platform": {"type": "string", "enum": ["bilibili", "xiaohongshu", "tieba",
                                                                    "weibo", "douyin", "zhihu"]},
                         "query": {"type": "string"}, "limit": {"type": "integer"}},
                        ["platform", "query"]),
    "get_memory_context": ({"query": {"type": "string"}}, []),
    "design_workflow": ({"reason": {"type": "string"},
                          "stages": {"type": "array", "items": {
                              "type": "object",
                              "properties": {"role": {"type": "string", "enum": list(roles.ROLES)},
                                             "goal": {"type": "string"}, "why": {"type": "string"},
                                             "max_steps": {"type": "integer"}},
                              "required": ["role", "goal", "why"]}}},
                         ["reason", "stages"]),
    "read_docx": ({"asset_id": {"type": "string"}, "max_chars": {"type": "integer"}}, ["asset_id"]),
    "read_pptx": ({"asset_id": {"type": "string"}, "max_chars": {"type": "integer"}}, ["asset_id"]),
    "read_pdf": ({"asset_id": {"type": "string"}, "max_chars": {"type": "integer"}}, ["asset_id"]),
}
ARGUMENTS['design_workflow'][0]['stages']['items']['properties'].update({
    'role':{'type':'string','description':'角色名或自定义阶段名；自定义阶段按通用角色执行'},
    'model_tier':{'type':'string','enum':['auto','light','strong']},
    'parallel_group':{'type':'string'},
    'tools':{'type':'array','items':{'type':'string'}}})
SYSTEM += """\n工程研究可以使用 list_files/read_file/write_file/run_command，不受 NumPy worker 的接口限制。
写入和命令必须遵守权限请求，等待用户批准，不得绕过。工作目录为本研究 workspace；命令没有操作系统沙箱隔离。
工程定量实验使用 command_experiment，命令最后输出 JSON {"metrics":{"注册指标":实测值},"correctness_passed":true/false}，再用 verify 在冻结工作区重跑；不要用打印常数代替测量。
可用 design_workflow 为阶段设置 model_tier=light/strong/auto；相邻同名 parallel_group 由独立子Agent并行。也可用 delegate_research 委派独立任务、read_subagents 读取内容、select_model 改变后续模型档位。子Agent不能递归委派，不能自行增加用户预算。"""


def _memory_locked(function):
    @functools.wraps(function)
    def call(*args, **kwargs):
        with storage.LOCK:
            return function(*args, **kwargs)
    return call


@_memory_locked
def memory_state():
    root = storage.MEMORY / "research"
    versions = sorted([storage.read_json(p) for p in root.glob("v*.json")], key=lambda v: v["version"])
    return {"active": storage.read_json(root / "active.json", {"version": 0})["version"], "versions": versions}


@_memory_locked
def activate(version):
    version = int(version)
    if version and not (storage.MEMORY / "research" / f"v{version}.json").exists():
        raise ValueError("研究经验版本不存在")
    storage.write_json(storage.MEMORY / "research" / "active.json", {"version": version})
    return memory_state()


@_memory_locked
def publish(run):
    if run.get("domain") != "research" or run["status"] != "completed" or not run.get("report", {}).get("lesson"):
        raise ValueError("需要完成复核并形成经验的研究")
    state = memory_state()
    existing = next((v for v in state["versions"] if v["source_run"] == run["id"]), None)
    if existing:
        return existing
    version = max([v["version"] for v in state["versions"]] + [0]) + 1
    value = {"version": version, "parent_version": state["active"], "source_run": run["id"],
             "question": run["question"], "lesson": run["report"]["lesson"],
             "evidence_ids": [e["id"] for e in run["experiments"] if e.get("verification")],
             "created_at": storage.now(), "status": "provisional",
             "limitations": "仅为该问题的方法经验；下一项任务仍须验证，未证明净收益。"}
    storage.write_json(storage.MEMORY / "research" / f"v{version}.json", value)
    activate(version)
    db.memory_add(session_id=run["id"], version=version, scope="research", lesson=value["lesson"],
                  applicability={"question": run["question"], "limitations": value["limitations"]},
                  evidence=value["evidence_ids"], status="provisional", source_run=run["id"],
                  parent_version=value["parent_version"], created_at=value["created_at"])
    return value


class ResearchEngine:
    def __init__(self, run, folder, *, decide=None):
        self.run, self.folder = run, Path(folder)
        self.budget = Budget(run["token_limit"], run["seconds_limit"])
        self.cancelled = threading.Event()
        self.pause_requested = threading.Event()
        if run.get("restore_budget"):
            self.budget.restore(run.get("budget") or {}, run.get("calls", [])[run.get("budget_call_start", 0):])
            run.pop("restore_budget", None)
        self.decide = decide or self.agent_step
        self.permission_gate = PermissionGate(self)
        self.turn_id = ((run.get("turns") or [{}])[-1] or {}).get("id") or "t1"

    def save(self):
        run = self.run
        run["updated_at"], run["budget"] = storage.now(), self.budget.snapshot()
        run['budget']['delegated_reserved_tokens'] = run.get('delegation_reserved',0)
        run["budget"]["tool_seconds"] = round(sum(s.get("elapsed_seconds", 0)
            for s in run["steps"][run.get("branch_step_start", 0):] if s.get("turn", "t1") == self.turn_id), 3)
        turn = next((item for item in run.get("turns", []) if item["id"] == self.turn_id), None)
        if turn is not None:
            turn["status"], turn["budget"] = run["status"], dict(run["budget"])
        storage.write_json(self.folder / "run.json", run)
        db.sync_run(run)

    def question_text(self):
        return self.run.get("current_question") or self.run.get("question", "")

    def seconds_left(self):
        left = self.budget.seconds_left()
        return float("inf") if left is None else left

    def steps_exhausted(self):
        limit = self.run.get("max_steps")
        return limit is not None and len(self._turn_steps()) >= limit

    def event(self, kind, text, **data):
        entry = activity.record(kind, text, log_file=self.folder.parent / "activity.jsonl",
                                source="research", actor="ollama" if kind in ("planning", "model_call") else "tool",
                                run_id=self.run["id"], task=self.run["task"], data=data,
                                status=data.get("status", "completed"), level=data.get("level", "info"))
        self.run["events"].append(entry)
        with (self.folder / "events.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.save()

    def active_identity(self):
        agent = self.run.get("active_agent") or {}
        return (agent.get("model") or self.run["model"],
                agent.get("model_source") or self.run.get("model_source", "local"),
                agent.get("role"), agent.get("id"))

    def resolve_model(self, role):
        conf = (self.run.get("roles") or {}).get(role) or {}
        source = conf.get("model_source") or self.run.get("model_source", "local")
        if source == "cloud":
            model = conf.get("model") or storage.settings()["cloud"].get("model") or ""
        else:
            model = conf.get("model") or self.run["model"]
        if not model:
            raise ValueError(f"角色 {role} 没有可用模型")
        return str(model), source

    def guard_pause(self):
        if self.pause_requested.is_set():
            raise history.Paused()

    def reserve_for(self, system, text, max_output=1500):
        """Conservative reservation: CJK prompts cost roughly one token per 2 characters."""
        chars = len(system or "") + len(text or "")
        return max(2048, chars // 2) + int(max_output)

    def complete(self, system, prompt, schema, purpose="research_decision", role=None, max_output=4096):
        """One admitted request, one trace record, including failed attempts."""
        cached = self.run.pop("pending_response", None)
        replay = self.run.pop("replay_response", False)
        if cached and replay and (cached.get("purpose") == purpose or
                                  cached.get("purpose", "").startswith("agent_step") and purpose.startswith("agent_step")):
            return cached["parsed"], cached["call"]
        self.guard_pause()
        route = routing.choose(self, purpose, prompt)
        model, source = route["model"], route["model_source"]
        if source != "cloud":
            max_output = 1500
        text = json.dumps(prompt, ensure_ascii=False) if isinstance(prompt, (dict, list)) else str(prompt)
        if source!='cloud':
            reserve=max(7644,self.reserve_for(system, text))  # Ollama context 6144 + at most 1500 generated tokens.
        else:
            # Cloud adds the schema to its prompt. UTF-8 bytes bound byte-tokenized input conservatively.
            prompt_bound = max(self.reserve_for(system, text, max_output=0),
                               len((system+text+json.dumps(schema,ensure_ascii=False)).encode('utf-8')))
            if self.budget.token_limit is not None:
                room = self.budget.token_limit - self.budget.used - prompt_bound
                if room < 2012:
                    raise BudgetError("剩余 token 不足以保守预留云端输入与完整输出")
                max_output = min(max_output, room - 512)
            reserve = prompt_bound + max(2012, max_output + 512)
        self.last_reserve = reserve
        self.budget.reserve(reserve)
        node = history.begin(self, "model_call", model=model, source=source, route=route,
                             role=role or route["role"], agent=route["agent"], purpose=purpose,
                             input={"system": system, "prompt": prompt, "schema": schema})
        self.event("planning", f"{'云端' if source == 'cloud' else '本地'}模型正在生成 {purpose} 内容", status="running")
        started = time.perf_counter()
        item = {"purpose": purpose, "model": model, "source": "cloud" if source == "cloud" else "ollama",
                "agent": route["agent"], "role": role or route["role"], "turn": self.turn_id,
                "node_id": node["id"], "route": route}
        parsed = None
        try:
            if source == "cloud":
                def capture(value):
                    item.update(value)
                client = cloudllm.Cloud(storage.settings()["cloud"], model, self.budget, on_call=capture)
                parsed, _ = client.call(system, text, schema, purpose, self.run.get("seed"),
                                        max_output=max_output, reserve=reserve)
            else:
                body = {"model": model, "stream": False, "think": False, "format": schema,
                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
                        "options": {"temperature": 0, "num_ctx": 6144, "num_predict": 1500}, "keep_alive": "20m"}
                item["request"] = body
                request = urllib.request.Request(storage.settings()["ollama_url"] + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=max(1, min(150, self.seconds_left()))) as response:
                    result = json.load(response)
                item["response"] = result
                if not all(isinstance(result.get(k), int) for k in ("prompt_eval_count", "eval_count")):
                    raise ValueError("模型未返回usage")
                item.update(input_tokens=result["prompt_eval_count"], output_tokens=result["eval_count"],
                            cached_tokens=result.get("prompt_eval_cached_count", 0),
                            context_tokens=result["prompt_eval_count"], status="ok")
                parsed = llmjson.parse(result.get("message", {}).get("content", ""))
            self.run["pending_response"] = {"purpose": purpose, "parsed": parsed, "call": dict(item)}
            return parsed, item
        except Exception as error:
            item.update(status="error", error=str(error))
            failure = reliability.classify(error, "llm:" + source)
            reliability.record(self.run, failure, storage.settings(), self.turn_id)
            self.run["decision_feedback"] = {"model_error": str(error), "purpose": purpose}
            if source != "cloud" and "input_tokens" not in item:
                item["unknown_reserved"] = reserve
            raise
        finally:
            item["wall_seconds"] = round(time.perf_counter() - started, 3)
            if source != "cloud":
                self.budget.record(item)
            self.run["calls"].append(item)
            history.settle(self, node, status="completed" if item.get("status") == "ok" else "failed",
                           output=parsed if item.get("status") == "ok" else {"error": item.get("error"), "response": item.get("response")},
                           usage={key: item.get(key, 0) for key in ("input_tokens", "output_tokens", "unknown_reserved", "cached_tokens")},
                           elapsed_seconds=item["wall_seconds"])
            self.event("model_call", "模型调用完成" if item.get("status") == "ok" else "模型调用失败",
                       status="completed" if item.get("status") == "ok" else "failed", call=item)

    def ask(self):
        tool, automatic_args = self.next_protocol_action()
        if automatic_args is not None:
            return {"tool": tool, "reason": "按研究协议执行确定性证据检查或独立验证，不额外消耗模型", "args": automatic_args, "source": "controller"}
        properties, required = ARGUMENTS[tool]
        schema = {"type": "object", "properties": properties, "required": required}
        observations = [{"tool": s["tool"], "status": s["status"], "observation": s["observation"],
                         "failed_code": s["args"].get("code", "") if s["status"] == "failed" else ""} for s in self.run["steps"][-3:]]
        prompt = {"question": self.question_text(), "current_action": tool,
                  "tool": {k: v for k, v in next(t for t in tools.TOOLS if t["name"] == tool).items() if k != "arguments"}, "plan": self.run.get("plan"), "assets": self.run["assets"],
                  "hypotheses": self.run["hypotheses"],
                  "experiments": [{k: e.get(k) for k in ("id", "hypothesis_id", "status", "result", "verification", "error")} for e in self.run["experiments"]],
                  "evidence": self.run["sources"], "observations": observations,
                  "documents": [{key: document.get(key) for key in
                                 ("asset_id", "name", "type", "chars", "note")} for document in self.run.get("documents", [])],
                  "memory": self.run.get("memory_context"),
                  "experience": self.run.get("experience"), "budget": self.budget.snapshot(),
                  "args_schema": schema,
                  "instruction": "本次只生成current_action的参数JSON，不要返回tool/reason/args外壳；必须只使用args_schema中的字段名。metric必须是一个简短英文标识符，实验代码须使用冻结的metric名称。"}
        system = SYSTEM.replace('自主选择下一项工具；严格返回 {"tool":"工具名","reason":"选择依据","args":{工具参数}}，不要把参数直接放在顶层。', "") + f"\n当前控制器已检查前置条件，进入{tool}。只返回该工具参数，不要选择其他工具。"
        if tool == "finish":
            system += "\n" + reporting.WRITING_GUIDE
            verified = [e for e in self.run["experiments"] if e.get("verification")]
            if verified:
                ids = "、".join(e["id"] for e in verified)
                hypotheses = "、".join(sorted({e["hypothesis_id"] for e in verified}))
                system += f"\n本次citations必须包含已复核实验 {ids} 及其假设 {hypotheses}；只引用文献会使报告判为未完成。"
        parsed, item = self.complete(system, prompt, schema,
                                     max_output=8192 if tool == "finish" else 4096)
        self.run.pop("pending_response", None)
        return {"tool": tool, "reason": f"按通用研究协议进入{tool}，研究内容由{item.get('source') == 'cloud' and '云端' or '本地'}模型生成",
                "args": parsed, "source": item.get("source", "ollama")}

    def tool_catalog(self, names):
        catalog = []
        for name in names:
            entry = registry.get(name) or {}
            properties, required = ARGUMENTS.get(name, ({}, []))
            properties = copy.deepcopy(properties)
            if name == "platform_search":
                properties["platform"]["enum"] = self.available_platforms()
            if name == "assess_source":
                properties["source_id"]["enum"] = [source["id"] for source in self.run["sources"]]
            if name in ("finish", "remember_fact"):
                field = "citations" if name == "finish" else "evidence_ids"
                properties[field] = {"type": "array", "items": {"type": "string"}}
                if name == "remember_fact":
                    identifiers = self.memory_evidence_ids()
                    if identifiers:
                        properties[field]["items"]["enum"] = identifiers
                    else:
                        properties[field]["maxItems"] = 0
                    properties[field]["minItems"] = 1
                    required = [*required, "evidence_ids"]
            catalog.append({"tool": name, "description": entry.get("description", ""),
                            "args_schema": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}})
        return catalog

    def note_evidence_attempt(self, tool, args, status):
        fill = self.run.get("gap_fill") or {}
        if not fill.get("required") or status != "completed" or tool not in EVIDENCE_TOOLS:
            return
        marker = f"{tool}:{args.get('query') or args.get('url') or args.get('platform') or args.get('asset_id') or ''}"
        markers = fill.setdefault("markers", [])
        if marker in markers:
            return
        markers.append(marker)
        fill["attempts"] = len(markers)
        if fill["attempts"] >= GAP_FILL_MIN:
            fill["required"] = False
            self.event("gap_fill", f"已尝试 {fill['attempts']} 种不同取证，可以重新评估或收尾",
                       status="completed")

    def add_source(self, row):
        """Repeated hits of one DOI or URL keep their first evidence ID; they are not independent sources."""
        def keys(item):
            return {str(item.get(key) or "").strip().lower() for key in ("doi", "url")} - {""}
        existing = next((source for source in self.run["sources"] if keys(row) & keys(source)), None)
        if existing is None:
            row["id"] = f"p{len(self.run['sources']) + 1}"
            self.run["sources"].append(row)
            return row
        if row.get("abstract") and not existing.get("abstract"):
            existing.update(abstract=row["abstract"], evidence_scope=row.get("evidence_scope") or existing.get("evidence_scope"))
        row["id"] = existing["id"]
        return row

    def evidence_ids(self):
        return sorted({row["id"] for group in ("assets", "sources", "hypotheses", "experiments")
                       for row in self.run.get(group, [])})

    def memory_evidence_ids(self):
        sources = [row for row in self.run.get("sources", [])
                   if (source_assessment.current(self, row["id"]) or {}).get("verdict") == "supports"]
        return sorted({row["id"] for row in [*self.run.get("assets", []), *sources]} |
                      {row["id"] for row in self.run.get("experiments", []) if row.get("verification")})

    def agent_step(self):
        """The model chooses the next tool itself; the controller only gates validity."""
        run = self.run
        options = self.available_tools()
        schema = protocol.action_schema(self.tool_catalog(options))
        def compact(value, limit=600):
            raw = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value or "")
            return raw[:limit]

        observations = [{"tool": step["tool"], "status": step["status"],
                         "observation": compact(step.get("observation"), 5000 if step['tool'] in ('read_file','read_subagents','web_fetch') else 4000 if step['tool']=='run_command' else 500)}
                        for step in run["steps"][-3:]]
        prompt = {"question": self.question_text(), "branch_note": run.get("branch_note"),
                  "branch_context": run.get("branch_context"), "decision_feedback": run.get("decision_feedback"),
                  "recovery": run.get("recovery_feedback"), "allowed_citation_ids": self.evidence_ids(),
                  "citation_policy": "记忆ID不能引用。经独立复核的本轮实验ID可作为实测证据引用；无实测或来源证据时 citations=[]，说明尚未核实并以未完成状态收尾。", "plan": run.get("plan"),
                  "research_kind": run.get("research_kind"), "assets": run["assets"][:4],
                  "documents": run.get("documents", [])[:6], "hypotheses": run["hypotheses"][:6],
                  "experiments": [{key: experiment.get(key) for key in
                                   ("id", "hypothesis_id", "status", "check", "verification", "error")}
                                  for experiment in run["experiments"][-6:]],
                  "evidence": [{key: source.get(key) for key in ("id", "title", "evidence_scope")}
                               for source in run["sources"][-12:]],
                  "source_assessments": [
                      review for source in run["sources"][-12:]
                      if (review := source_assessment.current(self, source["id"]))],
                  "evidence_count": len(run["sources"]),
                  "memory": {"text": str((run.get("memory_context") or {}).get("text") or "")[:1200]}
                  if run.get("memory_context") else None,
                  "experience": compact(run.get("experience"), 1200) if run.get("experience") else None,
                  "observations": observations, "budget": self.budget.snapshot(),
                  "gap_fill": run.get("gap_fill"), "feasibility": run.get("feasibility"),
                  "allowed_tools": self.tool_catalog(options),
                  "instruction": "你决定下一步：从allowed_tools选一个tool并给出符合其args_schema的args。"
                                 "外部事实证据不足可继续取证；可执行的本地问题可先plan，再注册假设、实验及独立复核。"
                                 "实验完成后应复核并引用实验ID；无实测或来源证据时finish(citations=[])如实报告未完成。"
                                 "模拟只能回答模拟机制内的问题，不能替代真实世界所需的数据。"
                                 "feasible=false 或 gap_fill.required 表示当前证据不足：可以继续换来源取证并重新 plan，"
                                 "也可用 finish 生成带明确缺口和检查提醒的未完成报告；每次运行都要以报告结束。"
                                 "如果需要多阶段编排，先调用design_workflow并写明每个阶段的设计理由。"}
        if "design_workflow" in options:
            prompt["role_tools"] = workflow_module.role_tools()
            prompt["instruction"] += ("规划阶段先判断问题需要的证据。定性、文献和概念比较安排取证、分析、报告；"
                                      "无需外部资料即可检验的本地计算或模拟可安排实验与复核，由模型先制定计划及可证伪假设。"
                                      "真实世界效果仍需相关数据，模拟结果只能解释模拟机制。"
                                      "实验角色没有finish权限；缺少假设时先注册。无论取证是否成功，最后安排reporter总结结论或证据缺口。"
                                      "role_tools 列出各角色能用的工具，自定义阶段名按 generalist；stages[].tools 通常省略，"
                                      "填写时只能取自该阶段角色的 role_tools，不能照抄你此刻的 allowed_tools。")
        if (run.get("active_agent") or {}).get("role") == "experimenter":
            prompt["instruction"] += ("当前是实验阶段，不能调用finish。若还没有计划，先用plan说明可执行方法和适用边界；"
                                      "若还没有假设，先调用hypothesis注册可检验的指标与阈值；"
                                      "再调用experiment实际运行。实验不可执行时保留原因，后续报告阶段再总结。")
        if "assess_source" in options:
            prompt["source_candidates"] = [
                {"id": source["id"], "title": source.get("title"), "scope": source.get("evidence_scope"),
                 "excerpt": str(source.get("content") or source.get("text") or source.get("abstract")
                                or source.get("snippet") or source.get("title") or "")[:1600]}
                for source in run["sources"][-6:]]
            prompt["instruction"] += ("评估来源须引用 source_candidates 中真实出现的片段；根据本题区分支持、背景或不适用，"
                                      "明确实际能支持的说法及局限。需要更多正文可先抓取或读取，不得凭标题推断文章结论。")
        prompt['execution_context'] = {'workspace':str(self.folder/'workspace'), 'access_mode':run.get('access_mode','approval'),
                                       'available_shells':[name for name,path in engineering.shells().items() if path],
                                       'model_profiles':{name:conf for name,conf in (run.get('routing') or {}).get('profiles',{}).items() if conf.get('model')},
                                       'subagents':run.get('subagents',[])[-6:]}
        correction = ""
        report_stage = "finish" in options and (options == ["finish"] or (run.get("active_agent") or {}).get("role") == "reporter")
        if report_stage:
            # Report prose needs content and actual measurements, not source titles alone.
            selected_sources = []
            seen_urls = set()
            supported = [source for source in run["sources"]
                         if (source_assessment.current(self, source["id"]) or {}).get("verdict") == "supports"]
            with_body = [source for source in run["sources"] if source.get("content") or source.get("text")]
            for source in [*reversed(supported), *reversed(with_body[-4:]), *reversed(run["sources"])]:
                key = source.get("url") or source["id"]
                if key in seen_urls:
                    continue
                seen_urls.add(key)
                selected_sources.append(source)
                if len(selected_sources) == 6:
                    break
            prompt["report_evidence"] = [{"id": source["id"], "title": source.get("title"),
                                          "scope": source.get("evidence_scope"),
                                          "excerpt": str(source.get("text") or source.get("content") or source.get("abstract") or "")[:2200],
                                          "assessment": source_assessment.current(self, source["id"])}
                                         for source in selected_sources]
            prompt["report_measurements"] = [{key: experiment.get(key) for key in
                                               ("id", "hypothesis_id", "result", "check", "verification", "error")}
                                              for experiment in run["experiments"][-6:]]
        writing = "\n" + reporting.WRITING_GUIDE if "finish" in options else ""
        for attempt in range(2):
            try:
                parsed, item = self.complete(SYSTEM + writing + correction, prompt, schema,
                                             purpose="agent_step" if attempt == 0 else "agent_step_retry",
                                             max_output=8192 if report_stage or (attempt and
                                                 prompt.get("repair", {}).get("truncated")) else 4096)
            except (BudgetError, history.Paused):
                raise
            except Exception as error:
                if attempt == 1:
                    raise RuntimeError(f"模型未返回可解析的决策：{error}") from error
                failure = reliability.classify(error)
                if failure.get("retryable"):
                    self.wait_for_recovery(failure["retry_after"])
                if isinstance(error, cloudllm.TruncatedCompletionError):
                    prompt["repair"] = {
                        "truncated": True, "error": str(error),
                        "instruction": "上次输出因长度限制截断。重新生成独立完整的JSON；"
                                       "若选择finish，只提供conclusion、citations、lesson，省略可选manuscript。"}
                    correction = ("\n上次输出被截断。只返回完整的 tool/reason/args JSON；"
                                  "finish 的 args 只需 conclusion、citations、lesson，不要续写截断内容。")
                else:
                    prompt["repair"] = {"error": str(error), "instruction": "重新返回满足schema的完整JSON对象"}
                    correction = "\n上次输出不是可解析的 JSON；只返回一个 JSON 对象，字段为 tool、reason、args。"
                continue
            self.run.pop("pending_response", None)
            issues = []
            tool = str(parsed.get("tool") or "").strip() if isinstance(parsed, dict) else ""
            args = parsed.get("args") if isinstance(parsed, dict) else None
            try:
                if tool not in options:
                    raise ValueError(f"tool={tool!r} 不在可用集合 {options}")
                if not isinstance(parsed.get("reason"), str) or not str(parsed.get("reason")).strip():
                    parsed["reason"] = "模型未说明选择依据"
                entry = next(row for row in self.tool_catalog(options) if row["tool"] == tool)
                protocol.validate(args, entry["args_schema"])
            except ValueError as error:
                issues.append(str(error))
            if not issues:
                self.run.pop("decision_feedback", None)
                break
            self.run["decision_feedback"] = {"invalid_response": parsed, "errors": issues}
            if tool == "finish":
                self.run["draft_report"] = {"args": args, "errors": issues, "verified": False}
            check = history.begin(self, "decision_error", tool="validate_decision", source="controller",
                                  input=parsed, parent_id=item.get("node_id"), reason="校验模型选择与工具参数")
            history.settle(self, check, status="failed", output={"error": "；".join(issues)}, elapsed_seconds=0)
            self.guard_pause()
            if attempt == 1:
                raise ValueError(f"模型两次返回无效决策：{'；'.join(issues)}")
            prompt["repair"] = {"invalid_response": parsed, "errors": issues,
                                "instruction": "根据工具schema修复整个tool/reason/args对象，不得编造缺失研究证据"}
            correction = "\n修复上次无效决策。必须返回 tool/reason/args 外壳及所选工具的完整参数。"
        return {"tool": tool, "reason": str(parsed.get("reason") or "模型自主选择"),
                "args": args, "source": item.get("source", "ollama")}

    def next_protocol_action(self):
        run = self.run
        if not run.get("evidence_collected"):
            return "inspect", {}
        read_ids = {item.get("asset_id") for item in run.get("documents", [])}
        pending = next((asset for asset in run.get("assets", [])
                        if asset.get("kind") == "document" and asset["id"] not in read_ids), None)
        if pending:
            tool = {".pdf": "read_pdf", ".docx": "read_docx", ".pptx": "read_pptx"}.get(
                Path(pending["name"]).suffix.lower())
            if tool:
                return tool, {"asset_id": pending["id"]}
        if not run.get("plan"):
            return "plan", None
        web = storage.settings()["web"]
        empirical_only = run.get("research_kind") == "simulation"
        if not empirical_only and web.get("enabled") and not any(s["tool"] in ("web_search", "literature") for s in run["steps"]):
            return "web_search", None
        if not empirical_only and web.get("enabled") and any(s["tool"] == "web_search" and s["status"] == "completed" for s in run["steps"]) \
                and not any(s["tool"] == "web_fetch" for s in run["steps"]):
            return "web_fetch", None
        query = run["plan"].get("literature_query", "").strip()
        if not empirical_only and query and not any(s["tool"] == "literature" for s in run["steps"]):
            return "literature", {"query": query}
        if not run["hypotheses"]:
            return "hypothesis", None
        reflections = run.get("reflections", [])
        if reflections and reflections[-1]["next_action"] == "revise" and len(run["hypotheses"]) == reflections[-1]["hypothesis_count"]:
            return "hypothesis", None
        current = run["hypotheses"][-1]
        candidates = [e for e in run["experiments"] if e["hypothesis_id"] == current["id"] and e["status"] == "completed"]
        if not candidates:
            return "experiment", None
        if not candidates[-1].get("verification"):
            return "verify", {"experiment_id": candidates[-1]["id"]}
        if not reflections or reflections[-1]["after_experiment"] != candidates[-1]["id"]:
            return "reflect", None
        return "finish", None

    def available_tools(self):
        run = self.run
        reliability.check_search_config(run, storage.settings(), self.turn_id)
        agent = run.get("active_agent") or {}
        if run.get("workflow_mode") == "plan" and not run.get("workflow") and agent.get("id") == "g0":
            return ["get_memory_context", "inspect", "design_workflow"]
        if self.in_wrap_up():
            return ["finish"]
        names = []

        def add(*items):
            for item in items:
                if item not in names:
                    names.append(item)

        add("get_memory_context")
        add('list_files','read_file','write_file','run_command','select_model','read_subagents')
        if not run.get('is_subagent'):
            add('delegate_research')
        if not run.get("evidence_collected"):
            add("inspect")
        add("literature", "platform_search")
        if storage.settings()["web"].get("enabled"):
            add("web_search", "web_fetch")
        assets = run.get("assets", [])
        if assets:
            suffixes = {Path(asset["name"]).suffix.lower() for asset in assets}
            if ".pdf" in suffixes:
                add("read_pdf")
            if ".docx" in suffixes:
                add("read_docx")
            if ".pptx" in suffixes:
                add("read_pptx")
            if any(asset.get("kind") == "text" for asset in assets):
                add("read")
        if run.get("workflow_mode") == "plan" and not run.get("workflow"):
            add("design_workflow")
        if not run.get("plan"):
            add("plan")
        else:
            add("hypothesis")
        # A follow-up turn clears the plan but keeps registered hypotheses and experiments.
        if run.get("hypotheses"):
            add("experiment",'command_experiment')
        if any(e.get("status") == "completed" and not e.get("verification") for e in run.get("experiments", [])):
            add("verify")
        if any(experiment.get("verification") for experiment in run.get("experiments", [])):
            add("reflect")
        if run.get("assets") or any(e.get("verification") for e in run.get("experiments", [])) or any(
                (source_assessment.current(self, source["id"]) or {}).get("verdict") == "supports"
                for source in run.get("sources", [])):
            add("remember_fact")
        if run.get("sources"):
            add("assess_source")
        if run.get("plan") or not (run.get("sources") or run.get("assets") or run.get("experiments")):
            add("finish")
        role = (agent.get("role") or "") if isinstance(agent, dict) else ""
        if role:
            permitted = set(roles.tools_for(role))
            names = [name for name in names if name in permitted]
            if role == "reporter" and not (run.get("gap_fill") or {}).get("required"):
                names = ["assess_source", "finish"] if "assess_source" in names and any(
                    source_assessment.current(self, source["id"]) is None for source in run["sources"]) else ["finish"]
            elif run.get("workflow_mode") == "plan" and role != "reporter":
                names = [name for name in names if name != "finish"]
            if not names:
                names = ["plan"] if not run.get("plan") else ["finish"]
        if run.get('is_subagent'):
            names = [name for name in names if name not in ('delegate_research','design_workflow','remember_fact')]
        restrict = set(agent.get("tools") or []) if isinstance(agent, dict) and agent.get("tools") else None
        if restrict:
            names = [name for name in names if name in restrict]
        fill = run.get("gap_fill") or {}
        if fill.get("required"):
            for candidate in ("inspect", "literature", "platform_search", "web_search", "web_fetch"):
                if candidate in names:
                    continue
                if candidate in ("web_search", "web_fetch") and not storage.settings()["web"].get("enabled"):
                    continue
                names.append(candidate)
            if fill.get("attempts", 0) >= GAP_FILL_MIN and "plan" not in names:
                names.append("plan")
        cfg = storage.settings()
        names = [name for name in names if name not in ("web_search", "literature") or
                 not reliability.blocked(run, reliability.resource(name, {}, cfg), cfg)]
        if "platform_search" in names and not self.available_platforms():
            names.remove("platform_search")
        return names

    def wrap_up_budget(self):
        """Reserve enough room to write one evidence-gap report, not for exploration."""
        token_limit = self.run.get("token_limit")
        seconds_limit = self.run.get("seconds_limit")
        reserve_tokens = 0 if token_limit is None else max(9000, int(token_limit * 0.15))
        reserve_seconds = 0 if seconds_limit is None else max(30, min(90, int(seconds_limit * 0.15)))
        return reserve_tokens, reserve_seconds

    def in_wrap_up(self):
        if not self.run.get("steps"):
            return False
        tokens, seconds = self.wrap_up_budget()
        token_wrap = False
        if self.run.get("token_limit") is not None:
            needed = max(tokens, getattr(self, "last_reserve", 0) + 2000)
            token_wrap = self.budget.token_limit - self.budget.used <= needed
        second_wrap = (self.run.get("seconds_limit") is not None
                       and self.budget.seconds_limit - self.budget.elapsed <= seconds)
        return token_wrap or second_wrap

    def infer_kind(self):
        run = self.run
        if run.get("experiments") and (run.get("sources") or run.get("documents")):
            return "mixed"
        if run.get("documents"):
            return "mixed"
        if run.get("sources"):
            return "literature"
        if run.get("experiments"):
            return "quantitative"
        return "quantitative"  # nothing collected: keep the strict "incomplete" grade

    def available_platforms(self):
        cfg = storage.settings()
        reliability.check_search_config(self.run, cfg, self.turn_id)
        return [name for name in ARGUMENTS["platform_search"][0]["platform"]["enum"]
                if (cfg["web"].get("enabled") or name in ("bilibili", "tieba")) and
                not reliability.blocked(self.run, reliability.resource("platform_search", {"platform": name}, cfg), cfg)]

    def allowed_tools(self):
        if self.in_wrap_up():
            # Wrap-up deliberately offers only finish; the role gate must not veto it.
            return None
        agent = self.run.get("active_agent") or {}
        role = agent.get("role")
        allowed = set(roles.tools_for(role)) if role else None
        if allowed is not None and agent.get("tools"):
            allowed &= set(agent["tools"])
        if allowed is not None and (self.run.get("gap_fill") or {}).get("required"):
            allowed |= EVIDENCE_TOOLS
        return allowed

    def execute(self):
        self.finishing = False
        self.run["status"] = "running"
        self.event("setup", "开始研究开放问题，按工具反馈决定下一步", question=self.run["question"])
        try:
            if self.run.get("workflow_mode") == "plan":
                self.orchestrate()
            else:
                self._loop(self.run.get("max_steps") if self.run.get("max_steps") is not None else 10 ** 9)
        except history.Paused:
            self.run["status"] = "paused"
            self.run["paused_at"] = storage.now()
        except BudgetError as error:
            self.run["status"], self.run["stop_reason"] = "budget_stopped", str(error)
        except Exception as error:
            self.run["status"], self.run["error"] = "failed", str(error)
        finally:
            with storage.LOCK:
                self.finishing = True
                if self.run["status"] == "pausing":
                    # The pause request arrived after the last pause check; keep the real outcome.
                    self.run["status"] = getattr(self, "status_before_pause", "running")
            if self.run["status"] == "running":
                self.run["status"] = "incomplete"
            if self.run["status"] == "paused":
                self.event("paused", "研究已暂停，可从本页继续", status="paused")
                self.run["restore_budget"] = True
                self.save()
                return self.run
            self.run["finished_at"] = storage.now()
            turn = next((item for item in self.run.get("turns", []) if item["id"] == self.turn_id), None)
            report_now = self.run.get("report") or {}
            if not str(report_now.get("conclusion") or "").strip():
                report_now = {**report_now, **self.failure_report()}
                self.run["report"] = report_now
                if self.run["status"] in ("running", "incomplete") and (
                        self.run.get("sources") or self.run.get("documents") or self.run.get("experiments")):
                    self.run["status"] = "inconclusive"
                    self.run["stop_reason"] = self.run.get("stop_reason") or "达到步骤或预算上限，已生成证据缺口报告"
            if self.run["status"] != "completed":
                for pending in self.run.get("pending_facts", []):
                    if pending["status"] == "pending_report":
                        pending["status"] = "not_published"
            if turn is not None:
                turn["status"], turn["finished_at"] = self.run["status"], self.run["finished_at"]
                turn["conclusion"] = str(report_now.get("conclusion", ""))
                turn["report"] = copy.deepcopy(report_now)
            self.write_report()
            if self.run["status"] == "completed" and self.run.get("memory_mode") != "disabled" and report_now.get("lesson"):
                try:
                    saved = publish(self.run)
                    self.run["memory_saved_version"] = saved["version"]
                except Exception as error:
                    self.run["memory_save_error"] = str(error)
            self.event("completed", f"研究结束：{self.run['status']}", status=self.run["status"])
        return self.run

    def wait_for_recovery(self, delay):
        if delay + 2 >= self.seconds_left():
            raise BudgetError("剩余时间不足以等待服务重试，保留已有结果")
        until = time.monotonic() + delay
        while time.monotonic() < until:
            self.guard_pause()
            if self.cancelled.is_set():
                self.run["status"] = "cancelled"
                return
            self.pause_requested.wait(min(.1, max(0, until - time.monotonic())))
        self.guard_pause()

    def _turn_steps(self):
        return [step for step in self.run["steps"][self.run.get("branch_step_start", 0):]
                if step.get("turn", "t1") == self.turn_id and step.get('status')!='paused']

    def _perform_tool(self, decision, retry=None):
        """Only retry read-only transient failures; settle each physical call."""
        tool, args = decision["tool"], decision.get("args", {})
        retry = retry or {}
        attempt = retry.get("attempt", 1)
        retry_of = retry.get("retry_of")
        if retry:
            self.run["pending_tool_retry"] = retry
            delay = max(0, retry.get("retry_at", 0) - time.time())
            if delay:
                self.wait_for_recovery(delay)
        while True:
            self.guard_pause()
            if self.cancelled.is_set():
                self.run["status"] = "cancelled"
                return
            if self.steps_exhausted():
                self.run["status"], self.run["stop_reason"] = "incomplete", "达到研究步骤上限，保留已有证据"
                return
            if self.seconds_left() <= 1:
                raise BudgetError("时间预算耗尽，不再启动工具")
            self.run.pop("pending_tool_retry", None)
            self.run.pop("pending_decision", None)
            self.run.pop("pending_response", None)
            agent = self.run.get("active_agent") or {}
            step = {"id": f"s{len(self.run['steps']) + 1}", "at": storage.now(), "tool": tool,
                    "reason": decision.get("reason", ""), "args": args, "source": decision.get("source", "injected"),
                    "agent": agent.get("id"), "role": agent.get("role"), "turn": self.turn_id,
                    "attempt": attempt, "retry_of": retry_of}
            node = history.begin(self, "tool_call", tool=tool, input=args, reason=step["reason"], source="local",
                                 role=step["role"], agent=step["agent"], attempt=attempt, retry_of=retry_of,
                                 parent_id=next((n["id"] for n in reversed(self.run.get("trace", [])) if n["kind"] == "model_call"), None))
            step["node_id"] = node["id"]
            started = time.perf_counter()
            next_retry = None
            try:
                observation = self.apply(tool, args)
                step.update(status="completed", observation=observation)
                self.run.pop("recovery_feedback", None)
            except history.Paused:
                step.update(status='paused',observation={'message':'操作在等待批准时暂停，未执行；恢复后重新请求批准。'})
                self.run['pending_decision']=decision
            except Exception as error:
                info = reliability.classify(error, reliability.resource(tool, args, storage.settings()))
                delay = info.get("retry_after", .25)
                can_retry = (tool in reliability.READ_TOOLS and info.get("retryable") and attempt < 2 and
                             delay <= 30 and self.seconds_left() > delay + 2 and
                             (self.run.get("max_steps") is None or len(self._turn_steps()) + 1 < self.run["max_steps"]))
                if info.get("retryable") and attempt >= 2:
                    info["blocked"] = True
                    info["message"] += " 已达到两次尝试上限，本轮改用其他来源。"
                info["action"] = "retry" if can_retry else "switch_source" if info.get("blocked") else "repair_decision"
                if info.get("retryable") and not can_retry and attempt < 2:
                    info["action"] = "defer"
                    info["message"] += " 当前预算或等待上限不允许立即重试。"
                reliability.record(self.run, info, storage.settings(), self.turn_id)
                step.update(status="failed", observation={**getattr(error,'result',{}), "error": str(error), "recovery": info,
                            "instruction": info["message"] + "；只能使用真实证据，必要时明确未完成。"})
                if can_retry:
                    next_retry = {"decision": decision, "attempt": attempt + 1, "retry_of": node["id"],
                                  "retry_at": time.time() + delay}
                    self.run["pending_tool_retry"] = next_retry
            step["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            self.run["steps"].append(step)
            history.settle(self, node, status=step["status"], output=step["observation"], elapsed_seconds=step["elapsed_seconds"])
            self.event("tool_result", f"{tool} 第 {attempt} 次尝试：{step['status']}", status=step["status"], step=step)
            self.note_evidence_attempt(tool, args, step["status"])
            if not next_retry:
                return step
            self.event("recovery", f"{tool} 暂时失败，等待后进行第 2 次尝试", status="running", retry=next_retry)
            self.wait_for_recovery(max(0, next_retry["retry_at"] - time.time()))
            if self.run["status"] == "cancelled":
                return step
            attempt, retry_of = next_retry["attempt"], next_retry["retry_of"]

    def _loop(self, limit, mark_incomplete=True):
        agent_id = (self.run.get("active_agent") or {}).get("id")
        experiment_stage = (self.run.get("active_agent") or {}).get("role") == "experimenter"
        extension_key = f"{self.turn_id}:{agent_id or 'react'}"
        rejected_decisions = 0
        while True:
            self.guard_pause()
            if self.cancelled.is_set():
                self.run["status"] = "cancelled"
                return
            steps = self._turn_steps()
            if self.steps_exhausted():
                self.run["status"], self.run["stop_reason"] = "incomplete", "达到研究步骤上限，保留已有证据"
                return
            scoped = [step for step in steps if step.get("agent") == agent_id
                      and not (experiment_stage and step.get("rejected_by_role"))]
            extensions = self.run.get("recovery_extensions", {}).get(extension_key, 0)
            pending_retry = self.run.get("pending_tool_retry")
            if len(scoped) >= int(limit) + extensions and not pending_retry:
                last = scoped[-1] if scoped else {}
                observation = last.get("observation")
                recovery = observation.get("recovery", {}) if isinstance(observation, dict) else {}
                if (last.get("status") == "failed" and recovery.get("repairable") and extensions < 1 and
                        self.budget.token_room(7644) and self.seconds_left() > 2):
                    self.run.setdefault("recovery_extensions", {})[extension_key] = extensions + 1
                    self.event("recovery", "阶段最后一步失败，使用剩余总预算给予一次修正机会", status="running", agent=agent_id)
                else:
                    break
            if self.seconds_left() <= 0:
                raise BudgetError("研究时间预算耗尽")
            if (agent_id and not pending_retry and not self.run.get("pending_decision")
                    and not self.available_tools()):
                if not scoped:
                    self.run["active_agent"]["status"] = "skipped"
                self.event("stage", "当前阶段没有可用工具，不调用模型，转入后续阶段", status="skipped",
                           level="warning", agent=agent_id)
                break
            try:
                retry = self.run.get("pending_tool_retry")
                decision = retry["decision"] if retry else self.run.pop("pending_decision", None) or self.decide()
                self.run["pending_decision"] = decision
                self.guard_pause()
            except (BudgetError, history.Paused):
                raise
            except Exception as error:
                info = reliability.classify(error)
                failed = {"id": f"s{len(self.run['steps']) + 1}", "at": storage.now(), "tool": "invalid",
                          "reason": "模型调用或工具选择失败", "args": {}, "source": "controller",
                          "agent": agent_id, "role": (self.run.get("active_agent") or {}).get("role"),
                          "turn": self.turn_id, "status": "failed", "elapsed_seconds": 0,
                          "observation": {"error": str(error), "recovery": info, "instruction": "按错误修正后重新选择工具。"}}
                if experiment_stage and isinstance(error, ValueError) and "模型两次返回无效决策" in str(error):
                    failed["rejected_by_role"] = True
                    rejected_decisions += 1
                self.run["steps"].append(failed)
                self.event("tool_result", f"决策失败：{error}", status="failed", step=failed)
                if not info.get("repairable", True):
                    self.run["status"], self.run["stop_reason"] = "incomplete", info["message"]
                    return
                if rejected_decisions >= 3:
                    self.event("stage", "连续三次无效决策，转入后续阶段处理", status="skipped", level="warning")
                    break
                continue
            tool = decision.get("tool")
            allowed = self.allowed_tools()
            if allowed is not None and tool not in allowed:
                self.run.pop("pending_decision", None)
                failed = {"id": f"s{len(self.run['steps'])+1}", "tool": tool, "status": "failed", "turn": self.turn_id,
                          "agent": agent_id, "role": (self.run.get("active_agent") or {}).get("role"), "args": decision.get("args", {}),
                          "elapsed_seconds": 0, "rejected_by_role": experiment_stage,
                          "observation": {"error": f"角色不可使用工具 {tool}"}}
                self.run["steps"].append(failed)
                self.event("tool_result", "工具被角色边界拒绝", status="failed", step=failed)
                if experiment_stage:
                    rejected_decisions += 1
                if rejected_decisions >= 3:
                    self.event("stage", "连续三次越权选择，转入后续阶段处理", status="skipped", level="warning")
                    break
                continue
            rejected_decisions = 0
            self._perform_tool(decision, retry=retry)
            if self.run["status"] in ("completed", "incomplete", "inconclusive", "cancelled"):
                return
            self.guard_pause()
        if mark_incomplete and self.run["status"] == "running":
            self.run["status"], self.run["stop_reason"] = "incomplete", "达到研究步骤上限，保留已有证据"

    def controller_step(self, tool, args):
        """Deterministic controller action that does not consume a model call."""
        index = len(self.run["steps"])
        agent = self.run.get("active_agent") or {}
        step = {"id": f"s{index + 1}", "at": storage.now(), "tool": tool,
                "reason": f"控制器执行确定性检查：{tool}", "args": args, "source": "controller",
                "agent": agent.get("id"), "role": agent.get("role"), "turn": self.turn_id}
        self.guard_pause()
        node = history.begin(self, "tool_call", tool=tool, input=args, reason=step["reason"],
                             source="controller", role=step["role"], agent=step["agent"])
        step["node_id"] = node["id"]
        started = time.perf_counter()
        try:
            observation = self.apply(tool, args)
            step.update(status="completed", observation=observation)
        except Exception as error:
            step.update(status="failed", observation={"error": str(error), "instruction": "按错误修正或跳过"})
        step["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        self.run["steps"].append(step)
        history.settle(self, node, status=step["status"], output=step["observation"], elapsed_seconds=step["elapsed_seconds"])
        self.event("tool_result", f"{tool} 控制器执行完成" if step["status"] == "completed" else f"{tool} 控制器执行失败",
                   status=step["status"], level="info" if step["status"] == "completed" else "warning", step=step)
        return step

    def orchestrate(self):
        def activate(identifier, role, goal, why="", label="", tools=None, model_tier='auto'):
            agent = next((a for a in self.run["agents"] if a["id"] == identifier and a.get("turn") == self.turn_id), None)
            if agent is None:
                model, source = self.resolve_model(role)
                configured=routing.profile(self,model_tier)
                if configured:
                    model,source=configured['model'],configured.get('model_source','local')
                agent = {"id": identifier, "role": role, "label": label or roles.ROLES[role]["label"],
                         "tools": tools, "goal": goal, "why": why, "status": "running",
                         "model": model, "model_source": source, "steps": [], "turn": self.turn_id}
                agent['model_tier'] = model_tier
                self.run["agents"].append(agent)
            self.run["active_agent"] = agent
            return agent

        if not self.run.get("workflow"):
            agent = activate("g0", "planner", "设计工作流并说明设计理由")
            if not any(s.get("agent") == "g0" and s.get("turn") == self.turn_id and s["tool"] == "get_memory_context" for s in self.run["steps"]):
                self.controller_step("get_memory_context", {})
            self._loop(4, mark_incomplete=False)
            self.guard_pause()
            if self.run["status"] != "running":
                return
            agent["status"] = "done"
            if not self.run.get("workflow"):
                workflow_state = workflow_module.plan(self)
                self.controller_step("design_workflow", {"reason": workflow_state.get("reason") or "规划器补齐工作流",
                    "stages": workflow_state["stages"]})
        stages=self.run['workflow']['stages']
        consumed=set()
        for index, stage in enumerate(stages):
            if index in consumed:
                continue
            self.guard_pause()
            if self.cancelled.is_set():
                self.run["status"] = "cancelled"
                break
            if self.run["status"] != "running":
                break
            group=stage.get('parallel_group')
            if group:
                peers=[]
                for peer_index in range(index,len(stages)):
                    if stages[peer_index].get('parallel_group')!=group:
                        break
                    peers.append((peer_index,stages[peer_index]))
                if not 1<=len(peers)<=3:
                    raise ValueError('一组并行子 Agent 最多 3 项')
                group_agent=activate(f'group{index+1}','generalist','并行研究：'+group)
                if group_agent['status'] in ('done','completed'):
                    consumed.update(i for i,_ in peers)
                    continue
                tasks=[{'goal':s['goal'],'role':s['role'],'model_tier':s.get('model_tier','auto'),'max_steps':s.get('max_steps',3)} for _,s in peers]
                self.controller_step('delegate_research',{'tasks':tasks,'batch_key':f'{self.turn_id}:{index}:{group}'})
                self.guard_pause()
                group_agent['status']='done'
                if (self.run.get('pending_decision') or {}).get('tool')=='delegate_research':
                    self.run.pop('pending_decision',None)
                consumed.update(i for i,_ in peers)
                continue
            stage_tools = workflow_module.stage_tools(stage["role"], stage.get("tools"))
            agent = activate(f"g{index + 1}", stage["role"], stage["goal"], stage.get("why", ""),
                             stage.get("label", ""), stage_tools, stage.get('model_tier','auto'))
            if agent["status"] in ("done", "completed"):
                continue
            skip_reason = ""
            if stage["role"] == "experimenter" and not self.run.get("hypotheses") and (
                    self.run.get("research_kind") in ("qualitative", "literature")
                    or (self.run.get("plan") or {}).get("feasible") is False):
                skip_reason = "研究方法无需数值实验，或尚无可执行的实验计划"
            elif stage["role"] == "verifier" and not any(
                    experiment.get("status") == "completed" and not experiment.get("verification")
                    for experiment in self.run.get("experiments", [])):
                skip_reason = "没有待复核的实验"
            if skip_reason:
                agent["status"] = "skipped"
                self.event("stage", f"阶段 {agent['id']}（{stage.get('label') or stage['role']}）跳过：{skip_reason}",
                           status="skipped", level="warning")
                self.save()
                continue
            self.event("stage", f"阶段 {agent['id']}（{stage.get('label') or stage['role']}）：{stage['goal']}", status="running")
            stage_limit = int(stage.get("max_steps", 3))
            if stage["role"] == "experimenter" and not self.run.get("plan") and not self.run.get("hypotheses"):
                stage_limit = max(stage_limit, 3)
            if stage["role"] == "reporter" and self.run.get("sources") and not any(
                    source_assessment.current(self, source["id"]) for source in self.run["sources"]):
                stage_limit = max(stage_limit, 2)
            if stage["role"] == "reporter" and (self.run.get("gap_fill") or {}).get("required"):
                remaining = max(0, GAP_FILL_MIN - self.run["gap_fill"].get("attempts", 0))
                stage_limit = max(stage_limit, remaining + 1)
            self._loop(stage_limit, mark_incomplete=False)
            self.guard_pause()
            agent["steps"] = [s["id"] for s in self.run["steps"] if s.get("agent") == agent["id"] and s.get("turn") == self.turn_id]
            if agent["status"] != "skipped":
                agent["status"] = "completed" if self.run["status"] == "completed" else "done"
            self.event("stage", f"阶段 {agent['id']} 结束：{agent['status']}", status=agent["status"])
        self.run["active_agent"] = None
        if self.run["status"] == "running":
            self.run["status"], self.run["stop_reason"] = "incomplete", "研究阶段已结束，仍未形成通过校验的报告"

    def apply(self, tool, args):
        if not isinstance(args, dict):
            raise ValueError("工具参数必须是对象")
        cfg = storage.settings()
        resource = reliability.resource(tool, args, cfg)
        if reliability.blocked(self.run, resource, cfg):
            raise reliability.Failure("该服务当前配置已被停用，未再次发起请求。请换来源或修复配置。",
                                      resource=resource, kind="circuit_open", blocked=True)
        return registry.dispatch(self, tool, args)

    def _apply_builtin(self, tool, args):
        run = self.run
        if not isinstance(args, dict):
            raise ValueError("工具参数必须是对象")
        if tool == "plan":
            if not isinstance(args.get("feasible"), bool) or not str(args.get("method", "")).strip():
                raise ValueError("计划需要feasible判断和method研究设计")
            kind = str(args.get("research_kind") or "quantitative").strip().lower()
            if kind not in ("quantitative", "simulation", "qualitative", "literature", "mixed"):
                raise ValueError("plan的research_kind必须是 quantitative/simulation/qualitative/literature/mixed")
            run["research_kind"] = kind
            run["plan"] = args
            run["feasibility"] = "ok" if args["feasible"] else "insufficient"
            if not args["feasible"]:
                fill = run.setdefault("gap_fill", {"attempts": 0, "markers": []})
                fill.update(required=True, reason=str(args.get("limitations", ""))[:300])
                self.event("gap_fill",
                           f"计划判定当前证据不足（{fill.get('attempts', 0)}/{GAP_FILL_MIN}）：继续换来源取证后重新评估",
                           status="running", level="warning", gap_fill=dict(fill))
            else:
                run.pop("gap_fill", None)
            return args
        if tool == "reflect":
            verified = [e for e in run["experiments"] if e.get("verification")]
            if not verified:
                raise ValueError("当前没有已复核实验，reflect 不适用：请继续检索/读取证据，或直接 finish 说明未完成与缺口")
            synonyms = {"finish": "finish", "done": "finish", "stop": "finish", "end": "finish",
                        "revise": "revise", "continue": "revise", "retry": "revise", "again": "revise"}
            action = synonyms.get(str(args.get("next_action") or "").strip().lower())
            if action is None:
                raise ValueError("reflect 的 next_action 只能是 revise 或 finish；"
                                 f"当前已复核实验：{', '.join(e['id'] for e in verified)}")
            args = {**args, "next_action": action,
                    "reason": str(args.get("reason") or "模型未说明反思理由")}
            item = {**args, "after_experiment": verified[-1]["id"], "hypothesis_count": len(run["hypotheses"])}
            run.setdefault("reflections", []).append(item)
            return item
        if tool == "inspect":
            result = tools.inspect_assets(run, self.folder)
            run["evidence_collected"] = True
            return result
        if tool == "literature":
            rows = tools.literature(args.get("query"), timeout=max(1, min(15, self.seconds_left())))
            for row in rows:
                self.add_source(row)
            run["evidence_collected"] = True
            return rows
        if tool == "read":
            asset = next((a for a in run["assets"] if a["id"] == args.get("asset_id")), None)
            if asset is None:
                raise ValueError("材料不存在，先inspect")
            run["evidence_collected"] = True
            if asset.get("kind") == "document":
                payload = (self.folder / "inputs" / asset["id"]).read_bytes()
                extracted = documents.extract(asset["name"], payload, max_chars=12000)
                return {"id": asset["id"], "name": asset["name"], "text": extracted["text"],
                        "note": extracted["note"], "truncated": extracted["truncated"]}
            return {"id": asset["id"], "text": (self.folder / "inputs" / asset["id"]).read_text(encoding="utf-8-sig")[:12000]}
        if tool == "hypothesis":
            if not run.get("evidence_collected") and not (
                    (run.get("plan") or {}).get("feasible") is True
                    and run.get("research_kind") in ("simulation", "quantitative")):
                raise ValueError("先检查材料或收集证据；自洽的本地实验可先plan说明方法与边界")
            if not all(isinstance(args.get(k), str) and args[k].strip() for k in ("statement", "prediction", "controls", "metric")):
                raise ValueError("假设需要statement、prediction、controls、metric")
            metric_name = str(args["metric"]).strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,40}", metric_name):
                raise ValueError("metric 必须是字母开头的短英文标识符（≤40字符，可含数字与下划线），不能使用中文句子或标点")
            args["metric"] = metric_name
            threshold = args.get("threshold")
            if args.get("relation") not in ("gt", "ge", "lt", "le") or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
                raise ValueError("需要合法的比较关系和有限数值阈值")
            item = {k: args[k] for k in ("statement", "prediction", "controls", "metric", "relation", "threshold")}
            item.update(id=f"h{len(run['hypotheses'])+1}", registered_at=storage.now())
            run["hypotheses"].append(item)
            return item
        if tool == "experiment":
            hypothesis = next((h for h in run["hypotheses"] if h["id"] == args.get("hypothesis_id")), None)
            if hypothesis is None:
                raise ValueError("先注册有效假设，再引用其hypothesis_id")
            item = {"id": f"x{len(run['experiments'])+1}", "hypothesis_id": hypothesis["id"], "code": args.get("code"), "status": "running"}
            run["experiments"].append(item)
            try:
                item["result"] = tools.experiment(item["code"], run, self.folder, item["id"], "explore", self.seconds_left(), run["seed"] + len(run["experiments"]))
                item["check"] = tools.check(hypothesis, item["result"])
                item["status"] = "completed"
            except Exception as error:
                item.update(status="failed", error=str(error))
                raise
            return {"id": item["id"], "result": item["result"], "check": item["check"]}
        if tool == "verify":
            item = next((e for e in run["experiments"] if e["id"] == args.get("experiment_id") and e["status"] == "completed"), None)
            if item is None or item.get("verification"):
                raise ValueError("需要一个尚未复核的成功实验ID")
            hypothesis = next(h for h in run["hypotheses"] if h["id"] == item["hypothesis_id"])
            command=item.get('kind')=='command'
            result = engineering.verify_command(self,item,secrets.randbelow(2**31)) if command else tools.experiment(item["code"], run, self.folder, item["id"], "verify", self.seconds_left(), secrets.randbelow(2**31))
            item["verification"] = {"result": result, "check": tools.check(hypothesis, result),
                                     "protocol": "同一冻结工作区与命令，独立重跑并检查正确性；不等于统计显著性或隔离了所有外部环境" if command else "同一冻结代码，新隐藏种子，预注册数值判据；不等于统计显著性或跨领域有效性"}
            if command and not result['correctness_passed']:
                item['verification']['check']['passed']=False
            return {"id": item["id"], **item["verification"]}
        if tool == "finish":
            fill = run.get("gap_fill") or {}
            role = (run.get("active_agent") or {}).get("role")
            if role and role != "reporter" and not self.in_wrap_up():
                raise ValueError("只有报告角色可以结束研究；当前阶段请继续其职责，或让流程进入报告阶段")
            valid = {a["id"] for a in run["assets"]} | {p["id"] for p in run["sources"]} | {h["id"] for h in run["hypotheses"]} | {e["id"] for e in run["experiments"]}
            warnings = []
            if fill.get("required"):
                warnings.append(
                    f"研究计划尚有证据缺口；已尝试 {fill.get('attempts', 0)}/{GAP_FILL_MIN} 种来源。"
                    "本报告只能说明现有记录，不能视为已完成的研究结论。")
            conclusion_text = str(args.get("conclusion") or "").strip()
            raw_manuscript = args.get("manuscript")
            if not conclusion_text:
                conclusion_text = str(raw_manuscript.get("conclusion") or "").strip() if isinstance(raw_manuscript, dict) else ""
            missing_conclusion = not conclusion_text
            if not conclusion_text:
                conclusion_text = "本轮未形成可核验的研究结论。"
                warnings.append("模型未提供结论；报告仅汇总已有记录，不能视为已完成研究。")
            raw_citations = args.get("citations", [])
            if not isinstance(raw_citations, list):
                raw_citations = []
                warnings.append("引用列表格式无效；本轮没有从该列表采纳任何证据。")
            citations = list(dict.fromkeys(c for c in raw_citations if isinstance(c, str) and c in valid))
            unknown = [str(c) for c in raw_citations if not isinstance(c, str) or c not in valid]
            if unknown:
                warnings.append("引用列表中的证据 ID 不在本轮记录中，未采纳：" + "、".join(unknown[:10]))
            manuscript, manuscript_warnings, added = reporting.review_manuscript(
                raw_manuscript, citations, valid)
            warnings.extend(manuscript_warnings)
            citations.extend(added)
            verified = [e for e in run["experiments"] if e.get("verification") and e["id"] in citations]
            kind = run.get("research_kind") or self.infer_kind()
            run["research_kind"] = kind
            reviewed = [source_assessment.current(self, source["id"]) for source in run["sources"]
                        if source["id"] in citations]
            supportive = [review for review in reviewed if review and review["verdict"] == "supports"]
            substantive = any(asset["id"] in citations for asset in run.get("assets", [])) or any(
                document.get("asset_id") in citations for document in run.get("documents", [])) or any(
                supportive)
            report = {"conclusion": conclusion_text, "citations": citations,
                      "lesson": str(args.get("lesson", ""))}
            if manuscript:
                report["manuscript"] = manuscript
            if warnings:
                report["warnings"] = warnings

            if kind in ("qualitative", "literature", "mixed"):
                if (substantive or (kind == "mixed" and verified)) and not missing_conclusion and not fill.get("required"):
                    report["evidence_scope_note"] = "来源评估的适用范围与局限：" + "；".join(
                        f"{review['source_id']}：{review['limitation']}" for review in supportive[:6]) if supportive else (
                        "本轮仅有用户材料或已复核的本地实验；不得外推未核实的真实世界效果。")
                    run["status"] = "completed"
                else:
                    report["evidence_gaps"] = [
                        "本轮引用来源尚无针对当前问题且附有原文引句的支持性评估",
                        *[f"{review['source_id']}：{review['limitation']}" for review in reviewed if review][:3]]
                    run["status"] = "inconclusive"
            else:
                run["status"] = "completed" if verified and not missing_conclusion and not fill.get("required") else "incomplete"
            if fill.get("required"):
                run["status"] = "inconclusive"
                report.setdefault("evidence_gaps", []).append(fill.get("reason") or "计划中的证据缺口尚未解决")
            run["report"] = report
            if run["status"] == "completed":
                extratools.commit_report_facts(self, citations)
            return {"status": run["status"], "verified_experiments": len(verified),
                    "warnings": warnings,
                    "evidence_gaps": report.get("evidence_gaps", []),
                    "note": "有复核记录不代表假设成立，须查看固定判据结果。"}
        raise ValueError(f"未开放工具 {tool}；请从可用工具选择")

    def failure_report(self):
        errors = []
        for step in self._turn_steps():
            if step.get("status") == "failed":
                info = step.get("observation", {}).get("recovery", {})
                text = info.get("message") or step.get("observation", {}).get("error")
                if text and text not in errors:
                    errors.append(text)
        if self.run.get("sources") or self.run.get("experiments"):
            reason = (f"已收集 {len(self.run.get('sources', []))} 条来源、"
                      f"{len(self.run.get('experiments', []))} 项实验，但未形成通过校验的报告")
        else:
            reason = "未取得外部证据或可验证实验结果；记忆不是证据，常识推断不作为研究证据"
        gaps = [reason]
        return {"conclusion": "未完成：" + reason, "citations": [], "lesson": "", "evidence_gaps": gaps,
                "recovery": {"reason": reason, "execution_errors": errors[-4:],
                             "next_steps": ["查看失败调用并修正工作流或工具参数", "依据已有来源重新核验问题前提与结论"]}}

    def write_report(self):
        current = reporting.markdown(reporting.document(self.run))
        (self.folder / f"report-{self.turn_id}.md").write_text(current, encoding="utf-8")
        sections = []
        for turn in self.run.get("turns", []):
            file = self.folder / f"report-{turn['id']}.md"
            if file.exists():
                sections.append(file.read_text(encoding="utf-8"))
        (self.folder / "report.md").write_text("\n\n---\n\n".join(sections) or current, encoding="utf-8")


def create(body):
    question = str(body.get("question", "")).strip()
    if not 8 <= len(question) <= 3000:
        raise ValueError("请写8至3000字符的研究问题")
    budget_enabled = bool(body.get("budget_enabled", True))
    access_mode=body.get('access_mode','approval')
    if access_mode not in ('approval','full'):
        raise ValueError('access_mode 只能为 approval 或 full')

    def optional_int(value, label, default=None):
        if value in (None, ""):
            return default
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label}必须是整数，或留空表示不限")
        if number <= 0:
            raise ValueError(f"{label}必须为正数，或留空表示不限")
        return number

    tokens = optional_int(body.get("token_limit"), "token 预算", 60000) if budget_enabled else None
    seconds = optional_int(body.get("seconds_limit"), "时间预算", None) if budget_enabled else None
    steps = optional_int(body.get("max_steps"), "步骤上限", None) if budget_enabled else None
    cfg = storage.settings()
    source = str(body.get("model_source") or cfg.get("model_source") or "local").strip().lower()
    if source not in ("local", "cloud"):
        raise ValueError("模型来源只能是 local 或 cloud")
    provider = None
    if source == "cloud":
        cloud = cfg["cloud"]
        missing = [label for key, label in (("base_url", "服务地址"), ("model", "模型名称"), ("api_key", "API Key"))
                   if not str(cloud.get(key) or "").strip()]
        if not cloud.get("enabled") or missing:
            raise ValueError("云端模型未配置完整：" + "、".join(missing or ["启用开关"]) + "；请先在模型设置中填写并保存")
        model = str(cloud["model"]).strip()
        provider = urllib.parse.urlparse(cloudllm.normalise(cloud["base_url"])).hostname
    else:
        model = str(body.get("model") or cfg["model"])
        if not model or "cloud" in model.lower() or len(model) > 120:
            raise ValueError("只允许已有的本地Ollama模型")
    mode = str(body.get("workflow_mode") or "react").strip().lower()
    if mode not in ("react", "plan"):
        raise ValueError("workflow_mode 只能是 react 或 plan")
    role_models = body.get("roles") if isinstance(body.get("roles"), dict) else {}
    identifier = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    folder = storage.RUNS / identifier
    assets = body.get("assets", [])
    if not isinstance(assets, list) or len(assets) > 8:
        raise ValueError("最多附加 8 份材料")
    prepared = []
    for index, asset in enumerate(assets):
        if not isinstance(asset, dict):
            raise ValueError("材料格式无效")
        name = Path(str(asset.get("name", ""))).name
        suffix = Path(name).suffix.lower()
        encoded = asset.get("content_base64")
        if encoded:
            try:
                payload = base64.b64decode(str(encoded), validate=True)
            except Exception as error:
                raise ValueError(f"{name} 的 base64 内容无效") from error
            kind = "document"
            if suffix not in documents.SUFFIXES:
                raise ValueError(f"不支持的文档类型：{suffix or name}")
            limit = 20 * 1024 * 1024
        else:
            payload = str(asset.get("content", "")).encode()
            kind = "text"
            if suffix not in ('.txt','.md','.csv','.json','.py','.js','.ts','.jsx','.tsx','.c','.cpp','.h','.rs','.go','.java','.toml','.yaml','.yml','.sh','.ps1','.bat','.sql','.html','.css'):
                raise ValueError('仅接受文本/代码材料，或 PDF/DOCX/PPTX 文档')
            limit = 500000
        if len(payload) > limit:
            raise ValueError(f"{name} 超过 {limit // 1024}KB")
        prepared.append({"id": f"a{index+1}", "name": name, "kind": kind, "bytes": len(payload),
                         "sha256": hashlib.sha256(payload).hexdigest(), "payload": payload})
    (folder / "inputs").mkdir(parents=True)
    for asset in prepared:
        payload=asset.pop('payload')
        (folder / "inputs" / asset["id"]).write_bytes(payload)
        if asset['kind']=='text':
            work=folder/'workspace';work.mkdir(exist_ok=True)
            destination=work/asset['name']
            if destination.exists():
                destination=work/(Path(asset['name']).stem+'-'+asset['id']+Path(asset['name']).suffix)
            destination.write_bytes(payload)
            asset['workspace_path']=destination.name
    context = memory.recall(question) if body.get("use_memory", True) else {"facts": [], "method": None, "text": ""}
    experience = context.get("method")
    manifest = {}
    for module in sorted((storage.ROOT / "researchloop").rglob("*.py")):
        relative = module.relative_to(storage.ROOT)
        target = folder / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(module.read_bytes())
        manifest[str(relative)] = hashlib.sha256(module.read_bytes()).hexdigest()
    storage.write_json(folder / "source-manifest.json", manifest)
    run = {"id": identifier, "session_id": identifier, "version": "0.4.0", "domain": "research",
           "task": question[:32], "question": question, "current_question": question,
           "turns": [{"id": "t1", "question": question, "at": storage.now(), "status": "queued"}],
           "mode": "agent", "model": model, "created_at": storage.now(), "status": "queued",
           "budget_enabled": budget_enabled, "token_limit": tokens, "seconds_limit": seconds,
           "max_steps": steps, "seed": int(body.get("seed", 42)),
           "assets": prepared, "sources": [], "source_assessments": [],
           "hypotheses": [], "experiments": [], "steps": [], "events": [], "calls": [],
           "experience": experience, "memory_version": experience["version"] if experience else 0,
           "memory_context": context, "memory_mode": "auto" if body.get("use_memory", True) else "disabled",
           "model_source": source, "provider": provider, "cloud_enabled": source == "cloud",
           "workflow_mode": mode, "roles": role_models, "routing": routing.clean(body.get("routing") or cfg.get("routing")), "workflow": None, "agents": [], "active_agent": None,
           'access_mode':access_mode,'permission_grants':[],'approval_requests':[],
           "source_hash": storage.fingerprint(manifest)}
    storage.write_json(folder / "run.json", run)
    return ResearchEngine(run, folder)


def resume(run, folder, question, *, token_limit=None, seconds_limit=None, max_steps=None):
    """Continue an existing session with a follow-up question."""
    text = str(question or "").strip()
    if not 8 <= len(text) <= 3000:
        raise ValueError("追问需要 8 至 3000 个字符")
    folder = Path(folder)
    turns = run.setdefault("turns", [])
    if not turns:
        turns.append({"id": "t1", "question": run.get("question", ""),
                      "at": run.get("created_at"), "status": run.get("status", "completed")})
    turns.append({"id": f"t{len(turns) + 1}", "question": text, "at": storage.now(), "status": "queued"})
    run["current_question"] = text
    context = memory.recall(text)
    run["memory_context"], run["experience"] = context, context.get("method")
    run["memory_version"] = (context.get("method") or {}).get("version", 0)
    run["status"], run["finished_at"], run["stop_reason"], run["error"] = "queued", None, None, None
    for key in ("pending_response", "pending_decision", "pending_tool_retry", "decision_feedback", "recovery_feedback", "draft_report", "restore_budget", "report"):
        run.pop(key, None)
    run["budget_call_start"] = len(run.get("calls", []))
    run["plan"], run["research_kind"], run["workflow"] = None, None, None
    run["agents"], run["active_agent"], run["reflections"] = [], None, []
    run["gap_fill"], run["feasibility"] = None, None
    budget_enabled = bool(run.get("budget_enabled", True))

    def pick(value, current, label, default=None):
        if value in (None, ""):
            return current if current is not None else default
        number = int(value)
        if number <= 0:
            raise ValueError(f"{label}必须为正数，或留空表示不限")
        return number

    if budget_enabled:
        run["token_limit"] = pick(token_limit, run.get("token_limit"), "token 预算", 60000)
        run["seconds_limit"] = pick(seconds_limit, run.get("seconds_limit"), "时间预算", None)
        run["max_steps"] = pick(max_steps, run.get("max_steps"), "步骤上限", None)
    else:
        run["token_limit"] = run["seconds_limit"] = run["max_steps"] = None
    storage.write_json(folder / "run.json", run)
    return ResearchEngine(run, folder)


def resume_paused(run, folder):
    if run.get("domain") != "research" or run.get("status") != "paused":
        raise ValueError("只有已暂停的研究可以原位继续")
    run["status"], run["restore_budget"] = "queued", True
    run["replay_response"] = bool(run.get("pending_response") and not run.get("pending_decision"))
    run.pop("finished_at", None)
    return ResearchEngine(run, folder)


def _builtin_handler(name):
    """Built-in tools register through the same interface as third-party tools."""
    def handler(engine, args):
        return engine._apply_builtin(name, args)
    handler.__qualname__ = f"builtin:{name}"
    return handler


for _name in ("plan", "inspect", "literature", "read", "hypothesis", "experiment", "verify", "reflect", "finish"):
    registry.register_tool(_name, _builtin_handler(_name), required=ARGUMENTS[_name][1])
