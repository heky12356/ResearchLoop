"""Research roles. A role decides the prompt, the tool subset and the model.

Different tasks use different sub-agents inside one session. Roles are the
place where a task type changes its model, its available tools and its goal.
"""

ROLES = {
    "planner": {
        "label": "规划",
        "goal": "判断问题需要外部来源、用户材料或本地实验，给出匹配的工作流",
        "tools": ["design_workflow", "inspect", "literature", "platform_search", "read", "read_docx", "read_pptx", "read_pdf", "assess_source",
                  "plan", "hypothesis", "web_search", "web_fetch", "get_memory_context"],
        "system": "你是科研任务的规划者。判断问题所需证据并写出研究设计。定性、文献、概念比较需取证；可由本地计算或模拟检验的问题可先设计实验，不以检索作为前置条件。实验前制定计划并注册可证伪假设；模拟不能回答未经数据验证的真实世界效果。不要在没有证据或实测时声称完成。",
    },
    "experimenter": {
        "label": "实验",
        "goal": "必要时先注册可证伪假设，再运行受约束实验并登记真实测量",
        "tools": ["plan", "read", "inspect", "hypothesis", "experiment"],
        "system": "你是实验执行者。无计划时先用 plan 说明可执行方法、假设边界与研究类型；无已注册假设时用 hypothesis 注册数值判据，再用 experiment 针对其ID写精炼NumPy代码，实际计算并emit_result。本地实验无需外部检索前置，模拟只支持模拟机制内的结论。不得调用 finish；报告由后续报告角色提交。失败时按错误修正代码，不编造数值。",
    },
    "verifier": {
        "label": "复核",
        "goal": "用新的隐藏种子重跑冻结代码，按预注册判据复核",
        "tools": ["verify"],
        "system": "你是独立复核者。只对已完成实验执行复核；复核可能否定假设，这也是有效结果。不要修改阈值或代码。",
    },
    "analyst": {
        "label": "分析",
        "goal": "围绕阶段目标检索、读取并比较真实来源，整理证据与缺口",
        "tools": ["inspect", "literature", "platform_search", "read", "read_docx", "read_pptx", "read_pdf", "assess_source",
                  "web_search", "web_fetch", "get_memory_context", "remember_fact"],
        "system": "你是证据分析者。围绕阶段目标检索、读取并比较真实来源，记录证据缺口；不把摘要或元数据当作结论，不做未经复核的定量断言。",
    },
    "generalist": {
        "label": "通用",
        "goal": "按阶段目标自由组合取证、假设、实验、复核与整理",
        "tools": ["design_workflow", "inspect", "literature", "platform_search", "read", "read_docx", "read_pptx", "assess_source",
                  "read_pdf", "plan", "hypothesis", "experiment", "verify", "web_search", "web_fetch",
                  "get_memory_context", "remember_fact", "finish"],
        "system": "你是通用研究执行者。根据阶段目标自主选择工具：取证、注册假设、运行并复核实验或整理证据；不越过证据下结论，结束时说明缺口。",
    },
    "reporter": {
        "label": "报告",
        "goal": "根据真实探索与复核结果形成结论、局限与方法经验",
        "tools": ["assess_source", "reflect", "finish"],
        "system": "你是报告撰写者。按摘要、问题背景、研究方法、结果、讨论局限与结论组织连贯正文，在finish.manuscript提交。只依据真实证据形成结论，定量结论需复核。计划或记忆中的说法不是已知事实。没有证据时用空citations报告未完成，不把猜测写成事实记忆；检索失败与引用错误要明确说明。",
    },
}

DEFAULT_ORDER = ["planner", "generalist", "reporter"]

for _role in ('planner','analyst','generalist','experimenter','verifier'):
    ROLES[_role]['tools'] += ['list_files','read_file','select_model','read_subagents']
for _role in ('experimenter','generalist'):
    ROLES[_role]['tools'] += ['write_file','run_command','command_experiment']
for _role in ('planner','analyst','generalist'):
    ROLES[_role]['tools'] += ['delegate_research']
ROLES['verifier']['tools'] += ['run_command']


def get(role):
    if role not in ROLES:
        raise ValueError(f"未知研究角色 {role}")
    return ROLES[role]


def tools_for(role):
    return list(get(role)["tools"])


def describe():
    return [{"role": name, "label": value["label"], "goal": value["goal"], "tools": value["tools"]}
            for name, value in ROLES.items()]
