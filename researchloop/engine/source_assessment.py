"""Question-specific review of retrieved sources without changing their raw evidence."""
import hashlib
import json

from . import registry


FIELDS = ("content", "text", "abstract", "snippet", "title")


def _material(source):
    return {field: str(source.get(field) or "") for field in FIELDS}


def _fingerprint(source):
    return hashlib.sha256(json.dumps(_material(source), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def current(engine, source_id):
    source = next((item for item in engine.run.get("sources", []) if item["id"] == source_id), None)
    if source is None:
        return None
    return next((item for item in reversed(engine.run.get("source_assessments", []))
                 if item["source_id"] == source_id and item["turn"] == engine.turn_id
                 and item["question"] == engine.question_text()
                 and item["material_sha256"] == _fingerprint(source)), None)


def assess(engine, args):
    source_id = args["source_id"]
    source = next((item for item in engine.run.get("sources", []) if item["id"] == source_id), None)
    if source is None:
        raise ValueError("只能评估本轮已检索到的来源ID")
    verdict = args["verdict"]
    if verdict not in ("supports", "context", "reject"):
        raise ValueError("verdict 必须是 supports、context 或 reject")
    quote = str(args.get("quote") or "").strip()
    claim = str(args.get("claim") or "").strip()
    reason = str(args.get("reason") or "").strip()
    limitation = str(args.get("limitation") or "").strip()
    if not reason or not limitation or not claim:
        raise ValueError("评估须说明该资料与当前问题的关系、判断理由和局限")
    if any(len(value) > 600 for value in (quote, claim, reason, limitation)):
        raise ValueError("引句、判断、理由与局限各不能超过600字符")
    material = _material(source)
    matches = [field for field, text in material.items() if quote and quote in text]
    if verdict == "supports" and (len(quote) < 4 or not matches):
        raise ValueError("支持性判断必须逐字引用已保存的标题、摘要或正文片段；不能凭记忆或补写内容")
    if quote and not matches:
        raise ValueError("引句不在该来源已保存的原始材料中")
    item = {"source_id": source_id, "turn": engine.turn_id, "question": engine.question_text(),
            "material_sha256": _fingerprint(source), "verdict": verdict, "quote": quote,
            "basis": matches[0] if matches else None, "claim": claim, "reason": reason,
            "limitation": limitation}
    engine.run.setdefault("source_assessments", []).append(item)
    return {key: item[key] for key in ("source_id", "verdict", "quote", "basis", "claim", "reason", "limitation")}


registry.register_tool(
    "assess_source", assess,
    description="针对当前研究问题评估一条已检索来源：逐字引证，判断其支持的有限说法、背景用途或不适用原因，并记录局限。标题只可支持标题本身能证明的事项；不能用摘要推断全文结论。",
    arguments={"source_id": "p1", "verdict": "supports/context/reject",
               "quote": "来源中逐字出现的短句；context/reject 可留空",
               "claim": "这条来源对当前问题实际能说明什么",
               "reason": "适用性判断依据", "limitation": "不能据此推出什么"},
    required=["source_id", "verdict", "claim", "reason", "limitation"])
