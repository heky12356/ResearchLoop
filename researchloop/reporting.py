"""Evidence-backed manuscripts and deterministic Markdown/LaTeX/PDF renderers.

Writing guidance borrows the structure/prose principles of writing-cs-report.
Exports never call a model, execute TeX, or modify historical research records.
"""
import html
import io
import math
import os
import re
import threading
from pathlib import Path

FIELDS = ("title", "abstract", "introduction", "methods", "results", "discussion", "conclusion")
MANUSCRIPT_SCHEMA = {"type": "object", "properties": {key: {"type": "string"} for key in FIELDS},
                     "additionalProperties": True}
WRITING_GUIDE = """收尾报告采用学术研究报告的行文方式，在 finish.manuscript 中提供 title、abstract、introduction、methods、results、discussion、conclusion。
摘要概括问题、实际方法、主要发现和边界；引言说明问题与相关背景，未检索文献时如实说明；方法交代数据或模拟假设、实验设计与复核；结果解释实际观测；讨论分析机制、反例、偏差与局限；结论直接回答问题。
借鉴 writing-cs-report：少用零碎小标题，正文用连贯段落展开因果与证据；不要把功能清单或执行日志当论文，不写空泛的“意义重大/综上所述”。不强凑字数、图表或个人经历。
每项实质性论断用 [证据ID] 就近引用，已复核实验可以作为本轮实测证据，无须附加外部文献。请核对 citations；正文漏列的真实证据会自动补入并提示，不存在的ID会标记为未核验。区分元数据、摘要、正文和实测；未做的实验、未获得的数值/出处不得补写，负结果照实分析，模拟不外推现实。
同一URL的搜索摘要与抓取正文只算一个来源；按本题 assess_source 的引句、判断和局限选择材料，标题只支持标题本身能证明的事项。把直接支持结论的来源与仅用于交代背景的来源区分开，遇到相互矛盾的描述须说明差异。
数据不足时按同样结构写简短的未完成研究报告，明确已尝试的方法、缺口和可继续步骤，不冒称完成。conclusion 参数保留简洁结论，lesson 仅写有条件的方法经验。
字数服从剩余预算：优先准确的方法、结果与局限；正文使用纯文本段落，不在字符串内嵌套 Markdown 标题或 LaTeX 命令。"""


def review_manuscript(value, citations, valid_ids):
    if value is None:
        return None, [], []
    if not isinstance(value, dict):
        return None, ["报告正文格式不是对象，已使用现有结论和执行记录生成报告。"], []
    warnings = []
    paper = {key: value[key].strip() for key in FIELDS
             if isinstance(value.get(key), str) and value[key].strip()}
    missing = [key for key in FIELDS if key not in paper]
    if missing:
        warnings.append("结构化报告缺少或无法使用的段落：" + "、".join(missing) + "；导出时使用已有记录填补。")
    added = []
    for text in paper.values():
        for evidence_id in re.findall(r"\[([adphx]\d+)\]", text):
            if evidence_id not in valid_ids:
                warning = f"正文引用 [{evidence_id}] 不在本轮真实证据中，无法核验；该引用不计入证据。"
            elif evidence_id not in citations and evidence_id not in added:
                added.append(evidence_id)
                warning = f"正文引用 [{evidence_id}] 已从本轮真实证据补入 citations。"
            else:
                continue
            if warning not in warnings:
                warnings.append(warning)
    return paper or None, warnings, added


def _number(value):
    if value is None:
        return "未记录"
    if isinstance(value, float) and math.isfinite(value):
        return f"{value:.6g}"
    return str(value)


def document(run):
    report = run.get("report") or {}
    paper = report.get("manuscript") or {}
    question = run.get("current_question") or run.get("question") or "研究报告"
    conclusion = report.get("conclusion") or run.get("stop_reason") or run.get("error") or "本轮尚未形成可引用结论。"
    hypotheses, experiments = run.get("hypotheses", []), run.get("experiments", [])
    plan = run.get("plan") or {}
    methods = paper.get("methods") or plan.get("method") or "本轮未保存完整方法正文；以下附录保留可核查的执行记录与来源。"
    if hypotheses:
        methods += "\n\n" + "\n".join(
            f"[{h['id']}] {h['statement']}；控制条件：{h.get('controls', '未记录')}；"
            f"预注册判据：{h['metric']} {dict(gt='>', ge='≥', lt='<', le='≤').get(h['relation'], h['relation'])} {h['threshold']}。" for h in hypotheses)
    rows = []
    for experiment in experiments:
        measured = (experiment.get("result") or {}).get("metrics") or {}
        verified = experiment.get("verification") or {}
        repeat = (verified.get("result") or {}).get("metrics") or {}
        for metric in dict.fromkeys([*measured, *repeat]):
            verdict = (verified.get("check") or {}).get("passed")
            target = next((h.get("metric") for h in hypotheses if h["id"] == experiment.get("hypothesis_id")), None)
            rows.append([experiment["id"], metric, _number(measured.get(metric)), _number(repeat.get(metric)),
                         "非判据指标" if target and metric != target else
                         "满足判据" if verdict is True else "未满足判据" if verdict is False else "未复核"])
    discussion = paper.get("discussion") or plan.get("limitations") or "现有证据只支持本报告所述范围；未进行的实验、统计检验及外部验证不作推断。"
    gaps = report.get("evidence_gaps") or []
    warnings = report.get("warnings") or []
    if gaps:
        discussion += "\n\n证据缺口：" + "；".join(gaps) + "。"
    if report.get("evidence_scope_note"):
        discussion += "\n\n" + report["evidence_scope_note"]
    references = []
    cited = set(report.get("citations") or [])
    for source in run.get("sources", []):
        if source["id"] in cited:
            references.append(f"[{source['id']}] {source.get('title', '未命名来源')}。"
                              f"{source.get('url') or source.get('doi') or ''} "
                              f"（读取范围：{source.get('evidence_scope', '未标注')}）")
    for asset in run.get("assets", []):
        if asset["id"] in cited:
            references.append(f"[{asset['id']}] 用户提供材料：{asset['name']}。SHA256: {asset.get('sha256', '未记录')}")
    for experiment in experiments:
        if experiment["id"] in cited:
            references.append(f"[{experiment['id']}] 本次研究的实验记录，关联假设 {experiment.get('hypothesis_id')}；"
                              f"{'已有独立重跑记录' if experiment.get('verification') else '未经独立重跑'}，"
                              f"代码与结果见 experiments/{experiment['id']}/。")
    for hypothesis in hypotheses:
        if hypothesis["id"] in cited:
            references.append(f"[{hypothesis['id']}] 本次预注册假设：{hypothesis['statement']}（研究设想，不是结果证据）。")
    budget = run.get("budget") or {}
    statuses = {"completed": "已完成", "inconclusive": "证据不足", "incomplete": "未完成", "failed": "执行失败",
                "budget_stopped": "预算停止", "cancelled": "已停止", "interrupted": "已中断", "paused": "已暂停"}
    appendix = [f"运行 ID：{run['id']}；状态：{statuses.get(run.get('status'), run.get('status'))}；"
                f"时间：{run.get('finished_at') or run.get('updated_at') or run.get('created_at', '未记录')}。",
                f"模型：{run.get('model', '未记录')}；模型 token：{_number(budget.get('total_tokens'))}；"
                f"研究用时：{_number(budget.get('elapsed_seconds'))} 秒；工具用时：{_number(budget.get('tool_seconds'))} 秒。",
                "研究记录、来源范围、检查点及冻结源码保存在导出记录包；本报告的排版导出不调用模型。"]
    if budget.get("unknown_reserved"):
        appendix.append(f"有 {_number(budget['unknown_reserved'])} token 为失败调用的保守预留，非已确认 usage。")
    appendix.append(f"模型请求：{_number(budget.get('model_calls'))} 次；云端请求：{_number(budget.get('cloud_calls'))} 次。")
    if budget.get("cloud_calls"):
        appendix.append(f"云端费用：{_number(budget.get('cloud_cost_usd'))} USD。" if budget.get("cloud_cost_known")
                        else "云端费用未完整计价；不能将未配置单价解释为零费用。")
    if any(e.get('kind')=='command' for e in experiments):
        appendix.append('工程实验：冻结工作区在 experiments/<id>/project；命令与参数见 run.json 对应 command_spec。'
                        'explore/result.json 与 verify/result.json 保存实际输出。使用匹配工具链，在冻结目录的副本执行该命令；'
                        '正确性检查由实验脚本实现，独立重跑不能消除外部服务或环境变化。')
    if any(e.get('kind')!='command' for e in experiments):
        appendix.append("实验复现：在导出目录使用匹配的 Python/NumPy 环境，运行 "
                        "python source/researchloop/engine/worker.py experiments/<id>/explore/input.json。"
                        "复核使用冻结代码与独立种子；判据通过不等于统计显著性。")
    if not paper:
        appendix.append("此运行未保存结构化论文正文；本文件根据已有结论、方法与结果整理，未补写新研究内容。")
    return {"title": paper.get("title") or question, "abstract": paper.get("abstract") or conclusion,
            "sections": [("研究问题与背景", paper.get("introduction") or f"本研究围绕以下问题展开：{question}"),
                         ("研究方法", methods), ("研究结果", paper.get("results") or conclusion),
                         ("讨论与局限", discussion), ("结论", paper.get("conclusion") or conclusion)],
            "measurements": rows, "references": references, "appendix": appendix,
            "warnings": warnings, "id": run["id"]}


def markdown(doc):
    parts = [f"# {doc['title']}", "## 摘要", doc["abstract"]]
    if doc.get("warnings"):
        parts += ["## 报告检查提醒", "\n".join("- " + warning for warning in doc["warnings"])]
    for index, (title, text) in enumerate(doc["sections"], 1):
        parts += [f"## {index}. {title}", text]
        if title == "研究结果" and doc["measurements"]:
            def cell(value):
                return str(value).replace("|", "\\|").replace("\n", " ")
            parts += ["实测与独立复核如下。判据为注册时的固定判据。",
                      "| 实验 | 指标 | 探索值 | 复核值 | 固定判据 |\n|---|---|---|---|---|\n" +
                      "\n".join("| " + " | ".join(map(cell, row)) + " |" for row in doc["measurements"])]
    parts += ["## 参考文献与证据", "\n\n".join(doc["references"]) or "本轮没有可引用的外部来源或实测证据。",
              "## 附录：复现与资源", "\n\n".join(doc["appendix"])]
    return "\n\n".join(parts) + "\n"


def latex_escape(value):
    chars = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
             "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    chars.update({"≥": r"\ensuremath{\geq}", "≤": r"\ensuremath{\leq}"})
    return "".join(chars.get(c, c) for c in str(value))


def latex(doc):
    esc = latex_escape
    parts = [r"% Compile with XeLaTeX and the ctex Chinese font packages. No shell escape required.",
             r"\documentclass[UTF8,a4paper,11pt]{ctexart}", r"\usepackage[margin=25mm]{geometry}",
             r"\usepackage{booktabs,longtable,hyperref}", r"\hypersetup{hidelinks}",
             r"\setlength{\emergencystretch}{3em}", r"\title{" + esc(doc["title"]) + "}",
             r"\author{ResearchLoop}", r"\date{}", r"\begin{document}", r"\maketitle",
             r"\begin{abstract}", esc(doc["abstract"]), r"\end{abstract}"]
    if doc.get("warnings"):
        parts += [r"\section*{报告检查提醒}", *[esc(warning) + r"\par" for warning in doc["warnings"]]]
    for title, text in doc["sections"]:
        parts += [r"\section{" + esc(title) + "}", esc(text)]
        if title == "研究结果" and doc["measurements"]:
            parts += [r"\begin{longtable}{p{.08\linewidth}p{.23\linewidth}p{.16\linewidth}p{.16\linewidth}p{.17\linewidth}}",
                      r"\toprule 实验 & 指标 & 探索值 & 复核值 & 固定判据 \\ \midrule\endhead"]
            parts += [" & ".join(map(esc, row)) + r" \\" for row in doc["measurements"]]
            parts += [r"\bottomrule\end{longtable}"]
    parts += [r"\section*{参考文献与证据}"]
    parts += [esc(reference) + "\n" for reference in doc["references"]] or ["本轮没有可引用证据。"]
    parts += [r"\appendix", r"\section{复现与资源}", *[esc(item) + "\n" for item in doc["appendix"]], r"\end{document}"]
    return "\n\n".join(parts) + "\n"


_font_lock = threading.Lock()


def _pdf_font():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    with _font_lock:
        if "ResearchSerif" in pdfmetrics.getRegisteredFontNames():
            return "ResearchSerif"
        candidates = [os.environ.get("RESEARCHLOOP_REPORT_FONT", ""),
                      str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/simsun.ttc"),
                      "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                      "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.ttf"]
        for file in candidates:
            if file and Path(file).is_file():
                try:
                    pdfmetrics.registerFont(TTFont("ResearchSerif", file, subfontIndex=0))
                    return "ResearchSerif"
                except Exception:
                    continue
        if "STSong-Light" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        return "STSong-Light"


def pdf(doc):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, LongTable, TableStyle
    font = _pdf_font()
    buffer = io.BytesIO()
    page = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=25*mm, rightMargin=25*mm,
                             topMargin=22*mm, bottomMargin=23*mm, title=doc["title"], author="ResearchLoop")
    body = ParagraphStyle("Body", fontName=font, fontSize=10.5, leading=18, spaceAfter=8,
                          alignment=TA_JUSTIFY, wordWrap="CJK", splitLongWords=True)
    title_style = ParagraphStyle("Title", parent=body, fontSize=18, leading=27, alignment=TA_CENTER, spaceAfter=20)
    heading = ParagraphStyle("Heading", parent=body, fontSize=12, leading=20, spaceBefore=13, spaceAfter=7, keepWithNext=True)
    small = ParagraphStyle("Small", parent=body, fontSize=9, leading=15)
    def paragraphs(text, style=body):
        return [Paragraph(html.escape(part).replace("\n", "<br/>"), style)
                for part in re.split(r"\n\s*\n", str(text)) if part.strip()]
    story = paragraphs(doc["title"], title_style) + paragraphs("摘要", heading) + paragraphs(doc["abstract"])
    if doc.get("warnings"):
        story += paragraphs("报告检查提醒", heading)
        for warning in doc["warnings"]:
            story += paragraphs(warning, small)
    for index, (title, text) in enumerate(doc["sections"], 1):
        story += paragraphs(f"{index}. {title}", heading) + paragraphs(text)
        if title == "研究结果" and doc["measurements"]:
            values = [["实验", "指标", "探索值", "复核值", "固定判据"], *doc["measurements"]]
            table = LongTable([[Paragraph(html.escape(str(v)), small) for v in row] for row in values],
                              colWidths=[14*mm, 48*mm, 29*mm, 29*mm, 40*mm], repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), .8, colors.HexColor("#223042")),
                                       ("LINEBELOW", (0, 0), (-1, 0), .4, colors.grey),
                                       ("LINEBELOW", (0, -1), (-1, -1), .8, colors.HexColor("#223042")),
                                       ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                       ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
            story += [Spacer(1, 5), table]
    story += paragraphs("参考文献与证据", heading)
    for ref in doc["references"] or ["本轮没有可引用的外部来源或实测证据。"]:
        story += paragraphs(ref, small)
    story += paragraphs("附录：复现与资源", heading)
    for item in doc["appendix"]:
        story += paragraphs(item, small)
    def footer(canvas, template):
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor("#687482"))
        canvas.drawString(25*mm, 13*mm, "ResearchLoop · " + doc["id"])
        canvas.drawRightString(A4[0]-25*mm, 13*mm, str(template.page))
        canvas.restoreState()
    page.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
