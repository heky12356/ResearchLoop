"""Workflow planning for a research session.

The planner proposes a stage list (role, goal, step budget); the orchestrator
executes the stages in order, each as a sub-agent with its own tools and model.
Roles are free-form within the known set; the plan is retried once with the
parse error quoted before falling back to the deterministic default, and the
fallback is recorded rather than hidden.
"""

from . import roles

SCHEMA = {
    "type": "object",
    "properties": {
        "stages": {"type": "array", "items": {
            "type": "object",
            "properties": {"role": {"type": "string",
                                       "description": "已知角色名（planner/analyst/generalist/experimenter/verifier/reporter），或自定义阶段名"},
                           "goal": {"type": "string"},
                           "why": {"type": "string"},
                           "max_steps": {"type": "integer"},
                           "model_tier": {"type":"string","enum":["auto","light","strong"]},
                           "parallel_group": {"type":"string","description":"相邻且同名的 1–3 阶段交给独立子 Agent 并行；后续阶段汇总结果。报告阶段保持串行。"},
                           "tools": {"type": "array", "items": {"type": "string"},
                                     "description": "可选：只允许该阶段使用的工具子集"}},
            "required": ["role", "goal", "why"]}},
        "reason": {"type": "string"},
    },
    "required": ["stages"],
}

MAX_STAGES = 8
MAX_STAGE_STEPS = 12
ROLE_ALIASES = {"规划": "planner", "计划": "planner", "实验": "experimenter", "复核": "verifier",
                "验证": "verifier", "报告": "reporter", "撰写": "reporter",
                "planner": "planner", "experimenter": "experimenter", "verifier": "verifier", "reporter": "reporter"}
STAGE_ESSENTIALS = {"experimenter": ["plan", "hypothesis", "experiment", "command_experiment"],
                    "verifier": ["verify"], "reporter": ["assess_source"]}


def stage_tools(role, tools):
    """A restricted stage keeps the tools its role exists to use."""
    if not tools:
        return tools
    return list(dict.fromkeys([*tools, *STAGE_ESSENTIALS.get(role, [])]))


def role_tools():
    """Tool ceilings shown to a workflow designer; custom stage names run as generalist."""
    return {name: roles.tools_for(name) for name in roles.ROLES}


def default_stages():
    return [
        {"role": "planner", "goal": "判断问题所需证据及研究方法", "max_steps": 4,
         "why": "判断外部取证或本地实验是否足以回答，限定模拟的适用范围"},
        {"role": "generalist", "goal": "根据问题继续取证；需要实验时注册假设、执行并复核", "max_steps": 6,
         "why": "定性问题核查来源，定量问题检验预注册判据"},
        {"role": "reporter", "goal": "依据复核结果形成结论、局限与方法经验", "max_steps": 2,
         "why": "只报告复核过的证据，并写清证据缺口与适用边界"},
    ]


def validate(stages):
    if not isinstance(stages, list) or not 1 <= len(stages) <= MAX_STAGES:
        raise ValueError(f"工作流需要 1–{MAX_STAGES} 个阶段")
    cleaned = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ValueError(f"第{index + 1}阶段必须是对象（role/goal/max_steps），收到 {type(stage).__name__}")
        raw_role = str(stage.get("role", "")).strip()
        if not raw_role or len(raw_role) > 40:
            raise ValueError(f"第{index + 1}阶段缺少角色或阶段名")
        role = ROLE_ALIASES.get(raw_role.lower(), raw_role.lower())
        label = ""
        if role not in roles.ROLES:
            # Unknown names become a custom stage with the generalist profile,
            # keeping the author's name for display.
            label = raw_role[:40]
            role = "generalist"
        goal = str(stage.get("goal", "")).strip()
        if not goal or len(goal) > 400:
            raise ValueError(f"第{index + 1}阶段缺少目标描述")
        steps = stage.get("max_steps")
        steps = int(steps) if isinstance(steps, int) else 3
        tools = stage.get("tools")
        selected = None
        if tools:
            if not isinstance(tools, list) or any(not isinstance(item, str) for item in tools):
                raise ValueError(f"第{index + 1}阶段 tools 必须是工具名数组")
            ceiling = set(roles.tools_for(role))
            chosen = [item for item in tools if item in ceiling]
            unknown = [item for item in tools if item not in ceiling]
            if unknown:
                raise ValueError(f"第{index + 1}阶段工具不在{role}角色范围内：{unknown}；该角色可用：{roles.tools_for(role)}。"
                                 "tools 是该阶段角色的工具子集，不是你当前可用的工具；不确定时省略 tools")
            if not chosen:
                raise ValueError(f"第{index + 1}阶段 tools 不能为空")
            selected = chosen
        elif tools is not None and tools != []:
            raise ValueError(f"第{index + 1}阶段 tools 必须是工具名数组")
        if role == "reporter" and selected and "finish" not in selected:
            raise ValueError("报告阶段必须包含 finish 工具")
        cleaned.append({"role": role, "label": label or roles.ROLES[role]["label"],
                        "goal": goal, "max_steps": max(1, min(MAX_STAGE_STEPS, steps)),
                        "tools": selected,
                        "why": str(stage.get("why") or "").strip()[:400]})
        tier=stage.get('model_tier','auto')
        group=str(stage.get('parallel_group') or '').strip()
        if tier not in ('auto','light','strong') or len(group)>60:
            raise ValueError('模型档位或并行组无效')
        if group and (role=='reporter' or selected):
            raise ValueError('报告阶段及自定义工具子集阶段暂不支持并行组，请保持串行')
        cleaned[-1].update(model_tier=tier,parallel_group=group)
    for index,stage in enumerate(cleaned):
        group=stage['parallel_group']
        if group and sum(s['parallel_group']==group for s in cleaned)>3:
            raise ValueError('每组最多 3 个并行子 Agent')
        if group and any(s['parallel_group']==group for s in cleaned[:index]) and cleaned[index-1]['parallel_group']!=group:
            raise ValueError('同一并行组的阶段必须相邻')
    return cleaned



def design(engine, args):
    """The design_workflow tool: the agent proposes stages and must justify them."""
    raw = args.get("stages")
    stages = validate(raw)
    from . import routing
    for stage in stages:
        routing.profile(engine,stage['model_tier'])
    reason = str(args.get("reason") or "").strip()
    if not reason:
        raise ValueError("design_workflow 需要整体设计理由 reason")
    for stage, source in zip(stages, raw):
        if not stage["why"]:
            raise ValueError(f"阶段 {stage['role']} 缺少设计理由 why")
    workflow = {"source": "agent_tool", "reason": reason[:600], "stages": stages,
                "planned_at": engine.run.get("updated_at")}
    engine.run["workflow"] = workflow
    engine.event("workflow", "工作流设计完成：" + " → ".join(stage["role"] for stage in stages),
                 status="completed", workflow=workflow)
    return workflow


def plan(engine):
    """Ask the planner role for a workflow, retry once, then fall back deterministically."""
    run = engine.run
    prompt = {"question": engine.question_text(), "assets": run["assets"], "budget": engine.budget.snapshot(),
              "roles": roles.describe(),
              "model_profiles": (run.get('routing') or {}).get('profiles',{}),
              "instruction": "只输出stages数组：阶段数量与划分由这个问题本身决定（1–8个），不要套用固定的四阶段模板。"
                             "每个阶段给出role、goal、max_steps；可以为阶段取自定义名称（role填自定义名，会作为通用阶段执行），"
                             "也可以用 tools 限定该阶段只使用哪些工具。先判断问题所需证据：定性、文献、概念比较等无需可执行数值检验的问题，应安排取证、分析和报告；不要为凑阶段而用模拟代替真实文献。"
                             "可由本地计算或模拟检验的问题，无须先找外部资料：安排experimenter、verifier与reporter；实验阶段可先plan制定方法，再hypothesis注册可证伪判据。真实世界效果仍需真实数据，模拟结论只适用于所设机制。"
                             "即使取证失败，也应继续可行的取证步骤，最后交给reporter总结已尝试来源和证据缺口。"
                             "每个阶段必须写 why：为什么需要、解决什么、为什么这个顺序。可用 model_tier 选择已配置的 light/strong，auto 沿用默认路由。"
                             "tools 是可选的角色工具子集；留空数组表示不限制。报告阶段须使用 reporter 且保留 finish，不能用 inspect。"
                             "独立取证/分析任务可放进相邻同名 parallel_group（每组最多3个），实际由独立子Agent并行；指定 tools 的阶段和报告阶段不能加入并行组。"
                             "有依赖的实验/复核及最终汇总放在之后。"
                             "阶段目标不能预设未核实的结论；不要编造材料中没有的数据。"}
    attempts = []
    stages = None
    source = "default"
    reason = ""
    for purpose in ("workflow_plan", "workflow_plan_retry"):
        try:
            extra = "" if not attempts else f"\n上次输出无效：{attempts[-1]}。必须严格按 schema 返回对象数组。"
            parsed, _ = engine.complete(roles.ROLES["planner"]["system"] + extra, prompt, SCHEMA,
                                        purpose=purpose, role="planner")
            stages = validate(parsed.get("stages"))
            source = "model" if purpose == "workflow_plan" else "model_retry"
            reason = str(parsed.get("reason", ""))[:200]
            break
        except Exception as error:
            attempts.append(str(error)[:200])
    if stages is not None:
        defaults = {
            "planner": "先确认证据与可行性，避免在缺数据时用计数或模拟替代真实研究",
            "experimenter": "用可执行实验检验假设，而不是由文本推断结论",
            "verifier": "独立重跑暴露偶然结果，复核不通过也是有效结果",
            "reporter": "只报告复核过的证据，并写清证据缺口与适用边界",
        }
        for stage in stages:
            if not stage.get("why"):
                stage["why"] = defaults.get(stage["role"], "该阶段是该方法链条上的必要环节")
        if not reason:
            reason = "规划器未给出整体理由；各阶段理由按角色职责补齐"
    if stages is None:
        stages = default_stages()
        source = "default"
        reason = "模型工作流两次不可用，使用通用三阶段：" + "；".join(attempts)
    workflow = {"source": source, "reason": reason, "stages": stages,
                "attempts": attempts, "planned_at": run.get("updated_at")}
    run["workflow"] = workflow
    engine.event("workflow", f"工作流已生成（{source}）：" + " → ".join(stage["role"] for stage in stages),
                 status="completed", level="info" if source.startswith("model") else "warning", workflow=workflow)
    return workflow

from . import registry

registry.register_tool("design_workflow", design,
                       description="设计研究的工作流：阶段数量与划分由问题决定（1–8），可自定义阶段名（按 generalist 执行）。"
                                   "tools 可省略；填写时只能取自 role_tools 中该阶段角色的工具，与你当前可用的工具无关。",
                       arguments={"reason": "整体设计理由",
                                  "stages": [{"role": "planner", "goal": "阶段目标", "why": "为什么需要这个阶段",
                                              "max_steps": 3}]},
                       required=["reason", "stages"])
