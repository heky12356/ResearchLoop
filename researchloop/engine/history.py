"""Invocation settlements and immutable checkpoints for pause/resume/fork."""
import copy
import hashlib
import os
import shutil
import time
import uuid
from pathlib import Path
from researchloop import storage


class Paused(Exception):
    pass


def begin(engine, kind, **fields):
    node = {"id": "n" + uuid.uuid4().hex[:12], "kind": kind, "seq": len(engine.run.setdefault("trace", [])) + 1,
            "turn": engine.turn_id, "at": storage.now(), "status": "running", **copy.deepcopy(fields)}
    engine.run["trace"].append(node)
    append(engine, {"event": "started", "node": node})
    engine.save()
    return node


def append(engine, item):
    with (engine.folder / "trace.jsonl").open("a", encoding="utf-8") as file:
        import json
        file.write(json.dumps(item, ensure_ascii=False) + "\n")


def _files(base,ignored):
    for directory,dirs,names in os.walk(base,followlinks=False):
        for name in dirs[:]:
            path=Path(directory)/name
            if name in ignored or path.is_symlink():
                dirs.remove(name)
                if path.is_symlink() and name not in ignored:
                    yield path
        for name in names:
            if name not in ignored:
                yield Path(directory)/name


def settle(engine, node, **fields):
    node.update(ended_at=storage.now(), **copy.deepcopy(fields))
    engine.save()
    node["checkpoint"] = "cp-" + node["id"]
    state = copy.deepcopy(engine.run)
    # Only copy files which existed at this boundary, never future artifacts.
    artifacts = []
    excluded=[]
    size=0
    ignored={'.git','.venv','node_modules','.cache','.tmp','.packages','__pycache__','checkpoint-files','checkpoints'}
    for base in ("inputs", "source", "experiments", "documents", "evidence",'workspace','subagents'):
        for file in _files(engine.folder/base,ignored):
            relative=file.relative_to(engine.folder)
            if file.is_symlink():
                excluded.append(relative.as_posix())
                continue
            if file.is_file():
                length=file.stat().st_size
                if length>20*1024*1024 or size+length>100*1024*1024:
                    excluded.append(relative.as_posix())
                    continue
                size+=length
                payload = file.read_bytes()
                digest = hashlib.sha256(payload).hexdigest()
                blob = engine.folder / "checkpoint-files" / digest
                if not blob.exists():
                    blob.parent.mkdir(parents=True, exist_ok=True)
                    blob.write_bytes(payload)
                artifacts.append({"path": file.relative_to(engine.folder).as_posix(), "sha256": digest})
    storage.write_json(engine.folder / "checkpoints" / (node["checkpoint"] + ".json"),
                       {"version": 1, "node": node["id"], "state": state, "artifacts": artifacts,'excluded_files':excluded})
    engine.save()
    append(engine, {"event": "settled", "node": node})


def fork(run, folder, node_id, question=""):
    """Copy a frozen prefix. Legacy records explicitly inherit no later evidence."""
    folder = Path(folder)
    node = next((item for item in run.get("trace", []) if item["id"] == node_id), None)
    if node_id=='__initial__':
        first=next((entry for entry in run.get('trace',[]) if entry.get('checkpoint')),None)
        saved=storage.read_json(folder/'checkpoints'/(first['checkpoint']+'.json')) if first else None
        if not saved:
            raise ValueError('缺少初始输入的冻结快照，无法精确回退')
        state=copy.deepcopy(saved['state'])
        artifacts=[a for a in saved['artifacts'] if Path(a['path']).parts[0] in ('inputs','source')]
        state.update(sources=[],source_assessments=[],hypotheses=[],experiments=[],documents=[],steps=[],calls=[],events=[],trace=[],
                     agents=[],active_agent=None,workflow=None,plan=None,reflections=[],evidence_collected=False,
                     subagents=[],delegation_batches=[],approval_requests=[],permission_grants=[])
        for key in ('report','pending_response','pending_decision','pending_tool_retry','gap_fill','decision_feedback','recovery_feedback'):
            state.pop(key,None)
        state['turns']=[{'id':'t1','question':state.get('current_question') or state['question'],'status':'paused','at':storage.now()}]
        precision='initial_inputs'
    elif node:
        checkpoint = node.get("checkpoint", "")
        if not checkpoint.startswith("cp-n") or not checkpoint.replace("-", "").isalnum():
            raise ValueError("该调用尚未完成，没有可用检查点")
        saved = storage.read_json(folder / "checkpoints" / (checkpoint + ".json"))
        if not saved:
            raise ValueError("检查点文件不存在，不能精确分支")
        if saved.get('excluded_files'):
            raise ValueError('该检查点有超出冻结限额的文件，不能声称精确恢复；请从更早检查点或初始输入重新开始')
        state, artifacts = saved["state"], saved["artifacts"]
        precision = "checkpoint"
    else:
        # Historical runs have no reconstructable runtime snapshot. Preserve only
        # submitted inputs and observations through the selected record, as context.
        records = run.get("steps", [])
        index = next((i for i, item in enumerate(records) if item["id"] == node_id), None)
        legacy_call = None
        if node_id.startswith("call:") and node_id[5:].isdigit():
            call_index = int(node_id[5:])
            legacy_call = next((item for i, item in enumerate(run.get("calls", [])) if i == call_index), None)
        if index is None and legacy_call is None:
            raise ValueError("执行记录不存在")
        state = copy.deepcopy(run)
        state["branch_context"] = [{k: item.get(k) for k in ("id", "tool", "reason", "args", "observation")}
                                   for item in records[:index + 1]] if index is not None else [
                                       {"model_request": legacy_call.get("request"), "model_response": legacy_call.get("response")}]
        state.update(sources=[], source_assessments=[], hypotheses=[], experiments=[], documents=[], steps=[], calls=[], events=[], trace=[],
                     agents=[], active_agent=None, workflow=None, plan=None, reflections=[], evidence_collected=False)
        selected_turn = records[index].get("turn") if index is not None else legacy_call.get("turn")
        question_at_node = next((turn.get("question") for turn in run.get("turns", []) if turn.get("id") == selected_turn), run["question"])
        state["current_question"] = question_at_node
        state["turns"] = [{"id": "t1", "question": question_at_node, "at": storage.now(), "status": "paused"}]
        state.pop("budget", None)
        for key in ("report", "pending_response", "pending_decision", "memory_context", "decision_feedback", "research_kind"):
            state.pop(key, None)
        artifacts = [{"path": f.relative_to(folder).as_posix(), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
                     for base in ("inputs", "source") for f in (folder / base).rglob("*") if f.is_file()]
        precision = "legacy_context"
    identifier = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    target = storage.RUNS / identifier
    target.mkdir(parents=True)
    try:
        for artifact in artifacts:
            relative = Path(artifact["path"])
            source = (folder / relative).resolve()
            if not source.is_relative_to(folder.resolve()):
                raise ValueError("检查点引用文件无效")
            frozen_file = folder / "checkpoint-files" / artifact["sha256"]
            if frozen_file.is_file():
                source = frozen_file
            if not source.is_file():
                raise ValueError("检查点引用文件不存在")
            if hashlib.sha256(source.read_bytes()).hexdigest() != artifact["sha256"]:
                raise ValueError("检查点文件已改变，拒绝混用后续内容")
            dest = target / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
        if precision=='initial_inputs':
            for asset in state.get('assets',[]):
                if asset.get('kind')=='text':
                    name=Path(asset.get('workspace_path') or asset['name']).name
                    workspace=target/'workspace';workspace.mkdir(exist_ok=True)
                    shutil.copyfile(target/'inputs'/asset['id'],workspace/name)
        state["lineage"] = {"run_id": run["id"], "node_id": node_id, "precision": precision,
                            "inherited_budget": state.get("budget"), "at": storage.now()}
        state.update(id=identifier, session_id=identifier, created_at=storage.now(), updated_at=storage.now(),
                     status="paused", finished_at=None, archived=False, stop_reason=None, error=None)
        state.update(access_mode='approval',permission_grants=[],approval_requests=[])
        # Carry prefix checkpoints as well so inherited records remain forkable.
        for entry in state.get("trace", []):
            frozen = entry.get("checkpoint", "")
            if frozen.startswith("cp-n") and frozen.replace("-", "").isalnum():
                checkpoint_source = folder / "checkpoints" / (frozen + ".json")
                if checkpoint_source.is_file():
                    (target / "checkpoints").mkdir(exist_ok=True)
                    shutil.copyfile(checkpoint_source, target / "checkpoints" / checkpoint_source.name)
                    for item in (storage.read_json(checkpoint_source) or {}).get("artifacts", []):
                        digest = item.get("sha256", "")
                        if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
                            blob = folder / "checkpoint-files" / digest
                            if blob.is_file():
                                (target / "checkpoint-files").mkdir(exist_ok=True)
                                shutil.copyfile(blob, target / "checkpoint-files" / digest)
            entry["inherited_from"] = run["id"]
        state["branch_note"] = str(question or "").strip()[:3000]
        if state["branch_note"]:
            # A new direction must be seen before executing any pending old choice.
            state.pop("pending_response", None)
            state.pop("pending_decision", None)
            state.pop("pending_tool_retry", None)
        state["budget"] = {}  # The branch's new spending is separate from inherited cost.
        state["budget_call_start"] = len(state.get("calls", []))
        state["branch_step_start"] = len(state.get("steps", []))
        state["replay_response"] = bool(state.get("pending_response") and not state.get("pending_decision"))
        # At a terminal report boundary open a new follow-up turn on the frozen state.
        if saved_terminal(state) or precision == "legacy_context":
            state.update(agents=[], active_agent=None, workflow=None, plan=None, gap_fill=None, feasibility=None)
            state.pop("report", None)
            state.pop("pending_response", None)
            state.pop("pending_decision", None)
            state.setdefault("turns", []).append({"id": f"t{len(state.get('turns', []))+1}",
                "question": state.get("branch_note") or state.get("current_question", state["question"]),
                "at": storage.now(), "status": "paused"})
        storage.write_json(target / "run.json", state)
    except Exception:
        # Retain the directory for diagnosis; it is not a valid session without run.json.
        raise
    return state, target


def retry(run,folder,node_id):
    records=run.get('trace') or []
    index=next((i for i,node in enumerate(records) if node['id']==node_id),None)
    if index is None or records[index].get('status')=='running':
        raise ValueError('请选择已经结束的调用重试')
    previous=next((node for node in reversed(records[:index]) if node.get('checkpoint')),None)
    selected=previous['id'] if previous else '__initial__'
    error=(records[index].get('output') or {}).get('error','')
    state,target=fork(run,folder,selected,
        '从目标调用之前重新研究。原调用：'+str(records[index].get('tool') or records[index].get('purpose') or records[index]['kind'])+
        '；原反馈：'+str(error)[:1000]+'。根据真实反馈选择修正或重试，不重复已成功的实验。')
    state['lineage'].update(action='retry_before',retry_of=node_id)
    storage.write_json(target/'run.json',state)
    return state,target


def saved_terminal(state):
    last = (state.get("trace") or [{}])[-1]
    return last.get("kind") == "tool_call" and last.get("tool") == "finish" and last.get("status") == "completed"
