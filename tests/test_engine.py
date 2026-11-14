"""Research loop contracts: no operator task names, real subprocesses and rejection."""
import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from researchloop import storage
from researchloop.engine import core, workflow
from researchloop.engine.worker import validate


class ResearchEngineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=storage.ROOT / ".cache")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for name, value in (("RUNS", root / "runs"), ("MEMORY", root / "memory")):
            patcher = patch.object(storage, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def engine(self, question="研究随机采样的均值误差是否在合理范围内", **kwargs):
        return core.create({"question": question, "max_steps": 10, **kwargs})

    def review(self, engine, source_id, quote, verdict="supports"):
        return engine.apply("assess_source", {
            "source_id": source_id, "verdict": verdict, "quote": quote,
            "claim": "该引句只能支持本题中的有限判断",
            "reason": "根据保存的原文判断其与当前问题的关系",
            "limitation": "不能从这条资料推断未呈现的结果"})

    def actions(self, *, threshold=0.5, broken=False):
        rows = [{"tool": "inspect", "args": {}},
                {"tool": "hypothesis", "args": {"statement": "随机均值的绝对误差小于0.5", "prediction": "误差较小", "controls": "标准正态分布，相同样本量", "metric": "error", "relation": "lt", "threshold": threshold}}]
        if broken:
            rows.append({"tool": "experiment", "args": {"hypothesis_id": "h1", "code": "raise ValueError('错误方法，需要修正')"}})
        code = "values = rng.normal(size=200)\nemit_result({'metrics': {'error': float(abs(np.mean(values)))}})"
        identifier = "x2" if broken else "x1"
        rows.extend([{"tool": "experiment", "args": {"hypothesis_id": "h1", "code": code}},
                     {"tool": "verify", "args": {"experiment_id": identifier}},
                     {"tool": "finish", "args": {"conclusion": "仅适用于当前模拟机制", "citations": [identifier, "h1"], "lesson": "均值误差实验应保留独立随机重跑"}}])
        return iter(rows)

    def test_real_code_is_frozen_and_independently_reexecuted(self):
        engine = self.engine()
        actions = self.actions()
        engine.decide = lambda: next(actions)
        run = engine.execute()
        self.assertEqual(run["status"], "completed")
        item = run["experiments"][0]
        self.assertEqual(item["result"]["code_sha256"], item["verification"]["result"]["code_sha256"])
        self.assertNotEqual(item["result"]["seed"], item["verification"]["result"]["seed"])
        self.assertTrue((engine.folder / "report.md").exists())
        self.assertTrue((engine.folder / "source" / "researchloop" / "engine" / "core.py").exists())
        self.assertEqual(run["hypotheses"][0]["threshold"], 0.5)

    def test_failed_experiment_can_be_corrected_without_losing_failure(self):
        engine = self.engine()
        actions = self.actions(broken=True)
        engine.decide = lambda: next(actions)
        run = engine.execute()
        self.assertEqual(run["status"], "completed")
        self.assertEqual([e["status"] for e in run["experiments"]], ["failed", "completed"])
        self.assertIn("错误方法", run["steps"][2]["observation"]["error"])

    def test_refuted_hypothesis_is_an_observable_completed_research_result(self):
        engine = self.engine()
        actions = self.actions(threshold=-1)
        engine.decide = lambda: next(actions)
        run = engine.execute()
        self.assertEqual(run["status"], "completed")
        self.assertFalse(run["experiments"][0]["verification"]["check"]["passed"])

    def test_report_without_verified_experiment_is_incomplete(self):
        engine = self.engine()
        engine.decide = lambda: {"tool": "finish", "args": {"conclusion": "资料不足", "citations": []}}
        self.assertEqual(engine.execute()["status"], "incomplete")

    def test_unknown_evidence_and_unregistered_experiments_are_rejected(self):
        engine = self.engine()
        result = engine.apply("finish", {"conclusion": "本轮没有可核验的证据。",
                                          "citations": ["fabricated-paper"]})
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(engine.run["report"]["citations"], [])
        self.assertIn("fabricated-paper", str(result["warnings"]))
        with self.assertRaises(ValueError):
            engine.apply("experiment", {"hypothesis_id": "missing", "code": "result = {}"})

    def test_material_is_readable_by_id_and_cannot_escape_workspace(self):
        engine = self.engine(assets=[{"name": "values.csv", "content": "value\n1\n2\n3\n"}])
        content = engine.apply("read", {"asset_id": "a1"})
        self.assertIn("value", content["text"])
        with self.assertRaises(ValueError):
            engine.apply("read", {"asset_id": "../../config.json"})

    def test_execution_has_a_restricted_interface(self):
        for code in ("import os", "open('config.json')", "np.load('x')", "x.__class__", "np.random.seed(1)"):
            with self.assertRaises(ValueError):
                validate(code)
        validate("values = rng.random(100)\nemit_result({'metrics': {'mean': float(np.mean(values))}})")

    def test_module_and_file_escapes_are_rejected_statically(self):
        for code in ('p = statistics.sys.modules["os"].getcwd()',
                     'x = np.fromregex("config.json", r"(.*)", [("a", "U200")])',
                     'np.zeros(1).dump("x.pkl")'):
            with self.assertRaises(ValueError):
                validate(code)

    def test_worker_runtime_blocks_processes_and_files_outside_assets(self):
        import subprocess
        import sys
        directory = Path(tempfile.mkdtemp(dir=storage.ROOT / ".cache"))
        self.addCleanup(lambda: __import__("shutil").rmtree(directory, ignore_errors=True))
        asset = directory / "a1"
        asset.write_text("x\n1\n2\n", encoding="utf-8")
        secret = directory / "secret.txt"
        secret.write_text("key", encoding="utf-8")
        probes = {
            "system": "statistics.sys.modules['os'].system('echo escaped')",
            "read": f"statistics.sys.modules['io'].open({str(secret)!r}).read()",
            "write": f"statistics.sys.modules['io'].open({str(directory / 'out.txt')!r}, 'w').write('x')",
            "ok": "rows = load_csv('a1')\nv = np.linalg.norm(np.fft.fft(np.array([float(r['x']) for r in rows])))\n"
                  "emit_result({'metrics': {'norm': float(v)}})",
        }
        script = ("import ast, json, sys\nfrom researchloop.engine import worker\nworker.validate = ast.parse\n"
                  "try:\n    worker.run(json.loads(sys.argv[1]))\n    print('RESULT=ok')\n"
                  "except Exception as error:\n    print('RESULT=' + type(error).__name__)\n")
        for name, code in probes.items():
            payload = json.dumps({"code": code, "seed": 1, "phase": "explore", "assets": {"a1": str(asset)}})
            done = subprocess.run([sys.executable, "-c", script, payload], cwd=storage.ROOT,
                                  capture_output=True, text=True, timeout=60)
            outcome = done.stdout.rsplit("RESULT=", 1)[-1].strip()
            if name == "ok":
                self.assertEqual(outcome, "ok", done.stderr)
            else:
                self.assertEqual(outcome, "PermissionError", (name, done.stdout, done.stderr))
        self.assertFalse((directory / "out.txt").exists())

    def test_complex_metrics_are_rejected(self):
        from researchloop.engine import worker
        with self.assertRaises(ValueError):
            worker.execute({"code": "emit_result({'metrics': {'m': np.complex128(0.2 + 5j)}})",
                            "seed": 1, "phase": "explore", "assets": {}})

    def test_random_generator_rejection_names_an_actionable_fix(self):
        for code in ("rng = np.random.default_rng()\nvalues = rng.normal(size=10)",
                     "np.random.seed(42)\nvalues = rng.normal(size=10)",
                     "values = np.random.normal(0, 1, 10)"):
            with self.assertRaises(ValueError) as caught:
                validate(code)
            message = str(caught.exception)
            self.assertIn("np.random", message)
            self.assertIn("rng.normal", message)
            self.assertIn("不要自行创建生成器", message)
            self.assertIn("第1行", message)

    def test_protocol_calls_the_model_for_content_but_never_skips_real_verification(self):
        engine = self.engine()
        engine.decide = engine.ask
        rows = list(self.actions())
        payloads = [{"feasible": True, "method": "比较独立模拟的误差", "limitations": "模拟情景", "literature_query": ""},
                    rows[1]["args"], rows[2]["args"], {"next_action": "finish", "reason": "已完成独立复核"}, rows[4]["args"]]
        responses = [io.BytesIO(json.dumps({"message": {"content": json.dumps(item)}, "prompt_eval_count": 50, "eval_count": 30}).encode()) for item in payloads]
        base_settings = storage.settings()
        offline = {**base_settings, "web": {**base_settings["web"], "enabled": False}}
        with patch("researchloop.storage.settings", return_value=offline), \
                patch("researchloop.engine.core.urllib.request.urlopen", side_effect=responses) as calls:
            run = engine.execute()
        self.assertEqual(run["status"], "completed")
        self.assertEqual([s["tool"] for s in run["steps"]], ["inspect", "plan", "hypothesis", "experiment", "verify", "reflect", "finish"])
        self.assertEqual(calls.call_count, 5)
        self.assertEqual(run["budget"]["total_tokens"], 400)
        self.assertEqual(run["steps"][4]["source"], "controller")

    def test_experience_versions_are_separate_and_rollback_preserves_history(self):
        engine = self.engine()
        actions = self.actions()
        engine.decide = lambda: next(actions)
        run = engine.execute()
        value = core.publish(run)
        self.assertEqual(value["version"], 1)
        self.assertEqual(core.activate(0)["active"], 0)
        self.assertEqual(len(core.memory_state()["versions"]), 1)

    def test_agent_chooses_the_next_tool_itself(self):
        engine = self.engine()
        payload = {"message": {"content": json.dumps({"tool": "inspect", "reason": "先看材料", "args": {}})},
                   "prompt_eval_count": 40, "eval_count": 20}
        with patch("researchloop.engine.core.urllib.request.urlopen",
                   return_value=io.BytesIO(json.dumps(payload).encode())):
            decision = engine.decide()
        self.assertEqual(decision["tool"], "inspect")
        self.assertEqual(decision["source"], "ollama")
        self.assertEqual(engine.run["calls"][0]["purpose"], "agent_step")

    def test_agent_cannot_choose_a_tool_outside_the_available_set(self):
        engine = self.engine()
        payload = {"message": {"content": json.dumps({"tool": "verify", "reason": "跳过前提", "args": {}})},
                   "prompt_eval_count": 40, "eval_count": 20}
        with patch("researchloop.engine.core.urllib.request.urlopen",
                   side_effect=lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode())):
            with self.assertRaises(ValueError):
                engine.decide()

    def test_follow_up_continues_the_same_session(self):
        engine = self.engine()
        actions = self.actions()
        engine.decide = lambda: next(actions)
        run = engine.execute()
        first_steps = len(run["steps"])
        follow = core.resume(run, engine.folder, "继续追问：复核结果对样本量选择意味着什么？",
                             token_limit=20000, seconds_limit=120, max_steps=6)
        follow.decide = lambda: {"tool": "finish", "reason": "收束追问",
                                 "args": {"conclusion": "追问示例结论", "citations": ["x1"], "lesson": ""}}
        second = follow.execute()
        self.assertEqual(second["id"], run["id"])
        self.assertEqual(len(second["turns"]), 2)
        self.assertEqual(second["current_question"], "继续追问：复核结果对样本量选择意味着什么？")
        self.assertGreater(len(second["steps"]), first_steps)
        self.assertEqual(second["turns"][1]["status"], second["status"])
        self.assertTrue((engine.folder / "report-t2.md").exists())
        combined = (engine.folder / "report.md").read_text(encoding="utf-8")
        self.assertIn("继续追问", combined)

    def test_qualitative_finish_with_web_evidence_counts_as_completed(self):
        engine = self.engine()
        engine.run["research_kind"] = "qualitative"
        engine.run["sources"] = [{"id": "p1", "title": "媒体报道", "url": "https://news.example.com/a",
                                  "abstract": "奶龙相关内容", "evidence_scope":
                                  "网页搜索摘要（tavily），媒体/平台内容，不是同行评议"}]
        self.review(engine, "p1", "奶龙相关内容")
        result = engine.apply("finish", {"conclusion": "依据本轮网页证据只能说明平台存在相关报道，不推断传播效果。",
                                         "citations": ["p1"], "lesson": "定性问题需要平台证据"})
        self.assertEqual(result["status"], "completed")
        self.assertNotIn("evidence_gaps", engine.run["report"])

    def test_qualitative_finish_with_metadata_only_stays_inconclusive(self):
        engine = self.engine()
        engine.run["research_kind"] = "literature"
        engine.run["sources"] = [{"id": "p1", "title": "论文", "url": "https://doi.example/x",
                                  "abstract": "", "evidence_scope": "只有元数据，不是结论证据"}]
        result = engine.apply("finish", {"conclusion": "检索只有元数据，无法回答传播与影响。",
                                         "citations": ["p1"], "lesson": ""})
        self.assertEqual(result["status"], "inconclusive")
        self.assertTrue(engine.run["report"].get("evidence_gaps"))

    def test_budget_limits_are_optional_and_not_capped(self):
        engine = core.create({"question": "研究样本量对估计误差的影响", "token_limit": 300000,
                              "seconds_limit": 3600, "max_steps": 30})
        self.assertEqual(engine.run["token_limit"], 300000)
        self.assertEqual(engine.run["budget_enabled"], True)
        partial = core.create({"question": "研究样本量对估计误差的影响", "token_limit": 9000,
                               "seconds_limit": None, "max_steps": None})
        self.assertEqual(partial.run["token_limit"], 9000)
        self.assertIsNone(partial.run["seconds_limit"])
        self.assertIsNone(partial.run["max_steps"])
        unlimited = core.create({"question": "研究样本量对估计误差的影响", "budget_enabled": False})
        self.assertFalse(unlimited.run["budget_enabled"])
        self.assertIsNone(unlimited.run["token_limit"])
        self.assertIsNone(unlimited.run["seconds_limit"])
        self.assertIsNone(unlimited.run["max_steps"])
        with self.assertRaises(ValueError):
            core.create({"question": "研究样本量对估计误差的影响", "token_limit": -5})

    def test_finish_is_hidden_until_a_plan_exists_once_evidence_collected(self):
        engine = self.engine()
        engine.run["evidence_collected"] = True
        engine.run["sources"] = [{"id": "p1", "title": "x", "url": "https://example.com/a",
                                  "abstract": "a", "evidence_scope": "摘要"}]
        names = engine.available_tools()
        self.assertNotIn("finish", names)
        self.assertIn("plan", names)
        engine.run["plan"] = {"feasible": True, "method": "m", "limitations": "", "literature_query": "",
                              "research_kind": "literature"}
        self.assertIn("finish", engine.available_tools())

    def test_wrap_up_reserve_restricts_tools_to_finish(self):
        engine = self.engine(token_limit=20000)
        engine.run["plan"] = {"feasible": True, "method": "m", "limitations": "", "literature_query": "",
                              "research_kind": "literature"}
        engine.run["sources"] = [{"id": "p1", "title": "x", "url": "https://example.com/a"}]
        engine.run["steps"] = [{"id": "s1", "tool": "inspect", "status": "completed", "turn": "t1",
                                "agent": None, "elapsed_seconds": 0}]
        engine.budget.input_tokens = 15000
        self.assertEqual(engine.available_tools(), ["finish"])

    def test_kind_is_inferred_for_web_evidence_without_plan(self):
        engine = self.engine()
        engine.run["sources"] = [{"id": "p1", "title": "公开报道存在", "url": "https://news.example.com/a",
                                  "abstract": "内容", "evidence_scope":
                                  "网页搜索摘要（tavily），媒体/平台内容，不是同行评议"}]
        self.review(engine, "p1", "公开报道存在")
        engine.run["plan"] = None
        result = engine.apply("finish", {"conclusion": "依据本轮网页证据只能说明存在公开报道，不推断传播效果。",
                                         "citations": ["p1"], "lesson": ""})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(engine.run["research_kind"], "literature")

    def test_source_review_selects_by_question_and_checks_original_quote(self):
        engine = self.engine(question="核查某篇论文的标题是否存在")
        engine.run["research_kind"] = "literature"
        engine.run["sources"] = [{"id": "p1", "title": "论文标题确实存在", "abstract": "",
                                  "evidence_scope": "只有元数据，不是结论证据"}]
        with self.assertRaisesRegex(ValueError, "逐字引用"):
            self.review(engine, "p1", "未见过的论文结果")
        review = self.review(engine, "p1", "论文标题确实存在")
        self.assertEqual(review["basis"], "title")
        result = engine.apply("finish", {"conclusion": "该来源只证明检索记录中存在这一论文标题。",
                                          "citations": ["p1"], "lesson": ""})
        self.assertEqual(result["status"], "completed")
        self.assertIn("p1", engine.run["report"]["evidence_scope_note"])

    def test_rejected_or_stale_source_review_cannot_complete_report(self):
        engine = self.engine(question="核查资料是否支持当前关系")
        engine.run["research_kind"] = "qualitative"
        engine.run["sources"] = [{"id": "p1", "title": "来源", "abstract": "只有背景内容。"}]
        self.review(engine, "p1", "只有背景内容。", verdict="context")
        result = engine.apply("finish", {"conclusion": "当前资料只提供背景，未能确认所问关系。",
                                          "citations": ["p1"], "lesson": ""})
        self.assertEqual(result["status"], "inconclusive")
        self.assertNotIn("p1", engine.memory_evidence_ids())

        engine.run["sources"][0]["abstract"] = "内容已更新，仍需重新核对。"
        self.assertIsNone(core.source_assessment.current(engine, "p1"))
        self.review(engine, "p1", "内容已更新，仍需重新核对。")
        engine.run["current_question"] = "另一个问题需要独立判断来源的适用性"
        self.assertIsNone(core.source_assessment.current(engine, "p1"))

    def test_unreviewed_source_does_not_complete_even_with_citation(self):
        engine = self.engine()
        engine.run["research_kind"] = "qualitative"
        engine.run["sources"] = [{"id": "p1", "title": "相关标题", "content": "这篇正文有文字。"}]
        result = engine.apply("finish", {"conclusion": "目前不能从未经评估的材料确认结论。",
                                          "citations": ["p1"], "lesson": ""})
        self.assertEqual(result["status"], "inconclusive")
        self.assertIn("assess_source", engine.available_tools())

    def test_assessment_tool_exposes_actual_source_and_reporter_can_use_it(self):
        engine = self.engine(workflow_mode="plan")
        engine.run["workflow"] = {"stages": []}
        engine.run["sources"] = [{"id": "p1", "title": "测试标题", "abstract": "实际摘录支持有限判断。"}]
        engine.run["active_agent"] = {"id": "g2", "role": "reporter", "tools": ["finish", "assess_source"]}
        self.assertEqual(engine.available_tools(), ["assess_source", "finish"])
        catalog = engine.tool_catalog(["assess_source"])[0]
        self.assertEqual(catalog["args_schema"]["properties"]["source_id"]["enum"], ["p1"])
        captured = {}

        def complete(system, prompt, schema, **kwargs):
            captured.update(prompt)
            return {"tool": "assess_source", "reason": "核对摘录", "args": {
                "source_id": "p1", "verdict": "supports", "quote": "实际摘录支持有限判断。",
                "claim": "有限判断", "reason": "摘录直接出现", "limitation": "仅有摘要"}}, {"source": "fixture"}

        engine.complete = complete
        decision = engine.agent_step()
        self.assertEqual(decision["tool"], "assess_source")
        self.assertIn("实际摘录支持有限判断。", captured["source_candidates"][0]["excerpt"])
        engine.apply(decision["tool"], decision["args"])
        self.assertEqual(engine.available_tools(), ["finish"])
        self.assertEqual(engine.memory_evidence_ids(), ["p1"])

    def test_unfinished_evidence_run_ends_with_a_gap_report(self):
        engine = self.engine(max_steps=1)
        engine.run["sources"] = [{"id": "p1", "title": "x", "url": "https://example.com/a",
                                  "abstract": "a", "evidence_scope": "摘要"}]
        engine.decide = lambda: {"tool": "inspect", "args": {}}
        result = engine.execute()
        self.assertEqual(result["status"], "inconclusive")
        self.assertIn("未完成", result["report"]["conclusion"])
        self.assertTrue((engine.folder / "report.md").exists())

    def test_web_evidence_reaches_reporter_without_a_numeric_plan(self):
        engine = self.engine(workflow_mode="plan", use_memory=False)
        engine.run["workflow"] = {"source": "fixture", "stages": workflow.validate([
            {"role": "planner", "goal": "读取网页证据", "why": "核实原文", "max_steps": 1},
            {"role": "reporter", "goal": "引用网页并写报告", "why": "形成结论", "max_steps": 1},
        ])}
        seen = []

        def decide():
            choices = engine.available_tools()
            seen.append((engine.run["active_agent"]["role"], choices))
            if engine.run["active_agent"]["role"] == "planner":
                return {"tool": "web_fetch", "args": {"url": "https://example.org/subject"}}
            if not engine.run["source_assessments"]:
                return {"tool": "assess_source", "args": {
                    "source_id": "p1", "verdict": "supports", "quote": "测试正文说明两个名称之间的关系。",
                    "claim": "只支持该页面所述关系", "reason": "页面正文直接表述该关系",
                    "limitation": "仅有这一个网页"}}
            return {"tool": "finish", "args": {
                "conclusion": "测试网页正文仅支持有限的定性判断。", "citations": ["p1"], "lesson": ""}}

        engine.decide = decide
        base = storage.settings()
        web = {**base["web"], "enabled": True}
        page = {"url": "https://example.org/subject", "title": "测试网页",
                "content": "测试正文说明两个名称之间的关系。", "fetched_at": "fixture", "truncated": False}
        with patch("researchloop.storage.settings", return_value={**base, "web": web}), \
                patch("researchloop.webaccess.fetch", return_value=page):
            run = engine.execute()
        self.assertEqual(run["status"], "completed")
        self.assertEqual([step["tool"] for step in run["steps"]], ["web_fetch", "assess_source", "finish"])
        self.assertEqual(seen[-1], ("reporter", ["finish"]))
        self.assertEqual(run["report"]["citations"], ["p1"])
        self.assertIn("测试正文", run["steps"][0]["observation"]["content"])
        self.assertIn("[p1]", (engine.folder / "report.md").read_text(encoding="utf-8"))

    def test_fetched_article_after_navigation_reaches_next_agent_and_reporter(self):
        engine = self.engine(workflow_mode="plan", use_memory=False)
        page_text = "站点导航\n" * 260 + "正文：奶蛙是奶龙形象的二次创作。"
        page = {"url": "https://example.org/article", "title": "测试条目", "content": page_text,
                "fetched_at": "fixture", "truncated": False}
        base = storage.settings()
        web = {**base["web"], "enabled": True}
        captured = []

        def complete(system, prompt, schema, **kwargs):
            captured.append(prompt)
            return {"tool": "finish", "reason": "完成报告",
                    "args": {"conclusion": "依据条目正文作有限说明。", "citations": ["p1"], "lesson": ""}}, {"source": "fixture"}

        with patch("researchloop.storage.settings", return_value={**base, "web": web}), \
                patch("researchloop.webaccess.fetch", return_value=page):
            engine.apply("web_fetch", {"url": page["url"]})
            engine.run["steps"].append({"id": "s1", "tool": "web_fetch", "status": "completed",
                                        "observation": {"content": page_text}, "turn": "t1"})
            engine.run["plan"] = {"feasible": True, "method": "查阅正文", "research_kind": "qualitative"}
            engine.run["active_agent"] = {"id": "g1", "role": "reporter"}
            engine.complete = complete
            engine.agent_step()
        self.assertIn("正文：奶蛙是奶龙形象的二次创作", str(captured[0]["observations"]))
        self.assertIn("正文：奶蛙是奶龙形象的二次创作", str(captured[0]["report_evidence"]))

    def test_repeated_fetch_of_same_article_keeps_one_body_source(self):
        engine = self.engine()
        page = {"url": "https://example.org/article", "title": "测试条目",
                "content": "这是一段可以核对的条目正文。" * 20, "fetched_at": "fixture", "truncated": False}
        base = storage.settings()
        web = {**base["web"], "enabled": True}
        with patch("researchloop.storage.settings", return_value={**base, "web": web}), \
                patch("researchloop.webaccess.fetch", return_value=page):
            first = engine.apply("web_fetch", {"url": page["url"]})
            second = engine.apply("web_fetch", {"url": page["url"], "max_chars": 20000})
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(len(engine.run["sources"]), 1)
        self.assertTrue(second["reused"])

    def test_search_hit_is_upgraded_to_body_without_new_evidence_id(self):
        engine = self.engine()
        url = "https://example.org/article"
        engine.run["sources"] = [{"id": "p1", "title": "条目", "url": url,
                                  "abstract": "搜索摘录中的可核查线索。",
                                  "evidence_scope": "网页搜索摘要（fixture）"}]
        page = {"url": url, "title": "条目", "content": "条目正文描述了实际关系。" * 20,
                "fetched_at": "fixture", "truncated": False}
        base = storage.settings()
        web = {**base["web"], "enabled": True}
        with patch("researchloop.storage.settings", return_value={**base, "web": web}), \
                patch("researchloop.webaccess.fetch", return_value=page):
            result = engine.apply("web_fetch", {"url": url})
        self.assertEqual(result["id"], "p1")
        self.assertEqual(len(engine.run["sources"]), 1)
        self.assertEqual(engine.run["sources"][0]["abstract"], "搜索摘录中的可核查线索。")
        self.assertEqual(engine.run["sources"][0]["content"], page["content"])
        self.assertEqual(engine.run["sources"][0]["evidence_scope"], "网页正文")

    def test_reporter_includes_older_source_upgraded_with_body(self):
        engine = self.engine(workflow_mode="plan", use_memory=False)
        engine.run["plan"] = {"feasible": True, "method": "读取网页", "research_kind": "qualitative"}
        engine.run["research_kind"] = "qualitative"
        engine.run["sources"] = [
            {"id": "p1", "title": "原始条目", "url": "https://example.org/article",
             "content": "条目正文确认了待核查的关系。", "evidence_scope": "网页正文"},
            *[{"id": f"p{i}", "title": f"无关线索{i}", "url": f"https://example.org/{i}",
               "abstract": "另一条索引", "evidence_scope": "网页搜索摘要"}
              for i in range(2, 11)],
        ]
        engine.run["active_agent"] = {"id": "g1", "role": "reporter"}
        captured = []

        def complete(system, prompt, schema, **kwargs):
            captured.append(prompt)
            return {"tool": "finish", "args": {
                "conclusion": "原始条目提供了待核查的关系描述。", "citations": ["p1"], "lesson": ""}}, {"source": "fixture"}

        engine.complete = complete
        engine.agent_step()
        self.assertIn("p1", [source["id"] for source in captured[0]["report_evidence"]])
        self.assertIn("条目正文确认了待核查的关系", str(captured[0]["report_evidence"]))

    def test_longer_fetch_extends_existing_truncated_source(self):
        engine = self.engine()
        url = "https://example.org/article"
        pages = iter([
            {"url": url, "title": "条目", "content": "前段", "fetched_at": "first", "truncated": True},
            {"url": url, "title": "条目", "content": "前段与后段", "fetched_at": "second", "truncated": False},
        ])
        base = storage.settings()
        web = {**base["web"], "enabled": True}
        with patch("researchloop.storage.settings", return_value={**base, "web": web}), \
                patch("researchloop.webaccess.fetch", side_effect=lambda *a, **kw: next(pages)):
            first = engine.apply("web_fetch", {"url": url})
            second = engine.apply("web_fetch", {"url": url, "max_chars": 20000})
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(len(engine.run["sources"]), 1)
        self.assertEqual(engine.run["sources"][0]["content"], "前段与后段")
        self.assertFalse(engine.run["sources"][0]["truncated"])

    def test_uncited_article_cannot_complete_metadata_only_report(self):
        engine = self.engine()
        engine.run["research_kind"] = "literature"
        engine.run["sources"] = [
            {"id": "p1", "title": "无关正文", "url": "https://example.org/a",
             "content": "另一话题的真实内容。", "evidence_scope": "网页正文"},
            {"id": "p2", "title": "待核实条目", "url": "https://example.org/b",
             "abstract": "", "evidence_scope": "只有元数据，不是结论证据"},
        ]
        result = engine.apply("finish", {"conclusion": "该条目目前只有元数据，无法确认具体关系。",
                                          "citations": ["p2"], "lesson": ""})
        self.assertEqual(result["status"], "inconclusive")

    def test_qualitative_plan_skips_unneeded_experiment_and_reports_evidence(self):
        engine = self.engine(workflow_mode="plan", use_memory=False)
        engine.run["workflow"] = {"source": "fixture", "stages": workflow.validate([
            {"role": "experimenter", "goal": "误安排的模拟", "why": "错误规划", "max_steps": 2},
            {"role": "verifier", "goal": "复核", "why": "实验之后", "max_steps": 1},
            {"role": "reporter", "goal": "提交定性结论", "why": "结束研究", "max_steps": 1},
        ])}
        engine.run["plan"] = {"feasible": True, "method": "比较来源正文", "research_kind": "literature"}
        engine.run["research_kind"] = "literature"
        engine.run["sources"] = [{"id": "p1", "title": "测试正文", "url": "https://example.org/a",
                                  "evidence_scope": "网页正文", "text": "测试正文支持有限比较。"}]
        self.review(engine, "p1", "测试正文支持有限比较。")
        engine.decide = lambda: {"tool": "finish", "args": {
            "conclusion": "测试正文只支持有限的定性比较。", "citations": ["p1"], "lesson": ""}}
        run = engine.execute()
        self.assertEqual(run["status"], "completed")
        self.assertEqual([step["tool"] for step in run["steps"]], ["finish"])
        self.assertEqual([agent["status"] for agent in run["agents"]], ["skipped", "skipped", "completed"])

    def test_planned_local_experiment_can_start_without_retrieval(self):
        engine = self.engine(workflow_mode="plan", use_memory=False, max_steps=10)
        engine.run["workflow"] = {"source": "fixture", "stages": workflow.validate([
            {"role": "experimenter", "goal": "设计并运行本地模拟", "why": "可计算的数值问题",
             "max_steps": 2, "tools": ["experiment"]},
            {"role": "verifier", "goal": "独立重跑", "why": "检查实测结果", "max_steps": 1},
            {"role": "reporter", "goal": "报告结论", "why": "说明适用边界", "max_steps": 1},
        ])}
        decisions = iter([
            {"tool": "plan", "args": {"feasible": True, "method": "模拟标准正态分布的样本均值",
                                     "limitations": "只适用于设定的分布", "literature_query": "",
                                     "research_kind": "simulation"}},
            {"tool": "hypothesis", "args": {"statement": "样本均值误差小于0.5",
                                            "prediction": "误差较小", "controls": "固定样本量与分布",
                                            "metric": "error", "relation": "lt", "threshold": 0.5}},
            {"tool": "experiment", "args": {"hypothesis_id": "h1", "code":
                "values = rng.normal(size=200)\nemit_result({'metrics': {'error': float(abs(np.mean(values)))}})"}},
            {"tool": "verify", "args": {"experiment_id": "x1"}},
            {"tool": "finish", "args": {"conclusion": "本轮模拟的均值误差已独立复核，结论只适用于该模拟机制。",
                                      "citations": ["h1", "x1"], "lesson": "保留冻结代码与复核记录"}},
        ])
        offered = []

        def decide():
            offered.append(engine.available_tools())
            return next(decisions)

        engine.decide = decide
        run = engine.execute()
        self.assertEqual(run["status"], "completed")
        self.assertEqual([step["tool"] for step in run["steps"]],
                         ["plan", "hypothesis", "experiment", "verify", "finish"])
        self.assertEqual(run["sources"], [])
        self.assertFalse(run.get("evidence_collected"))
        self.assertIn("plan", offered[0])
        self.assertIn("hypothesis", offered[1])
        self.assertTrue(run["experiments"][0]["verification"])

    def test_mixed_report_accepts_verified_local_experiment(self):
        engine = self.engine()
        engine.apply("plan", {"feasible": True, "method": "本地模拟并说明边界",
                              "limitations": "尚无外部数据", "literature_query": "",
                              "research_kind": "mixed"})
        engine.run["experiments"] = [{"id": "x1", "hypothesis_id": "h1", "status": "completed",
                                      "verification": {"check": {"passed": False}}}]
        result = engine.apply("finish", {
            "conclusion": "当前只完成本地实验的独立复核，没有外部事实结论。",
            "citations": ["x1"], "lesson": "须进一步寻找真实数据"})
        self.assertEqual(result["status"], "completed")
        self.assertIn("本地实验", engine.run["report"]["evidence_scope_note"])

    def test_legacy_simulation_protocol_does_not_require_literature(self):
        engine = self.engine()
        engine.run["evidence_collected"] = True
        engine.apply("plan", {"feasible": True, "method": "本地模拟", "limitations": "模拟边界",
                              "literature_query": "optional", "research_kind": "simulation"})
        config = storage.settings()
        with patch("researchloop.storage.settings",
                   return_value={**config, "web": {**config["web"], "enabled": True}}):
            self.assertEqual(engine.next_protocol_action()[0], "hypothesis")

    def test_research_kind_inference_uses_records_not_source_scope_labels(self):
        engine = self.engine()
        engine.run["sources"] = [{"id": "p1", "title": "网页正文", "evidence_scope": "网页正文"}]
        self.assertEqual(engine.infer_kind(), "literature")
        engine.run["experiments"] = [{"id": "x1", "status": "completed"}]
        self.assertEqual(engine.infer_kind(), "mixed")

    def test_primary_planner_prompt_excludes_unneeded_experiment(self):
        engine = self.engine(workflow_mode="plan", use_memory=False)
        engine.run["active_agent"] = {"id": "g0", "role": "planner"}
        captured = {}

        def complete(system, prompt, schema, **kwargs):
            captured["system"] = system
            captured["instruction"] = prompt["instruction"]
            return {"tool": "design_workflow", "reason": "按问题类型规划",
                    "args": {"reason": "定性比较", "stages": [
                        {"role": "analyst", "goal": "查阅材料", "why": "需要证据"},
                        {"role": "reporter", "goal": "提交报告", "why": "需要结论"}]}}, {"source": "fixture"}

        engine.complete = complete
        self.assertEqual(engine.agent_step()["tool"], "design_workflow")
        self.assertIn("定性", captured["system"])
        self.assertIn("定性", captured["instruction"])
        self.assertIn("最后安排reporter", captured["instruction"])

    def test_experimenter_recovers_from_early_finish_and_registers_hypothesis(self):
        engine = self.engine(workflow_mode="plan", use_memory=False, max_steps=12)
        engine.run["workflow"] = {"source": "fixture", "stages": workflow.validate([
            {"role": "experimenter", "goal": "注册假设并运行模拟", "why": "数值检验",
             "max_steps": 2, "tools": ["experiment"]},
            {"role": "verifier", "goal": "独立复核", "why": "验证结果", "max_steps": 1},
            {"role": "reporter", "goal": "报告", "why": "收束", "max_steps": 1},
        ])}
        engine.run["plan"] = {"feasible": True, "method": "用固定机制进行模拟", "research_kind": "simulation"}
        engine.run["research_kind"] = "simulation"
        engine.run["evidence_collected"] = True
        decisions = iter([
            {"tool": "finish", "args": {"conclusion": "提前结束", "citations": []}},
            {"tool": "finish", "args": {"conclusion": "提前结束", "citations": []}},
            {"tool": "hypothesis", "args": {
                "statement": "均值误差小于0.5", "prediction": "误差较小", "controls": "固定采样规模",
                "metric": "error", "relation": "lt", "threshold": 0.5}},
            {"tool": "experiment", "args": {"hypothesis_id": "h1", "code":
                "values = rng.normal(size=200)\nemit_result({'metrics': {'error': float(abs(np.mean(values)))}})"}},
            {"tool": "verify", "args": {"experiment_id": "x1"}},
            {"tool": "finish", "args": {"conclusion": "仅在所设模拟机制中完成了独立复核。",
                                      "citations": ["h1", "x1"], "lesson": ""}},
        ])
        engine.decide = lambda: next(decisions)
        run = engine.execute()
        self.assertEqual(run["status"], "completed")
        self.assertEqual([step["tool"] for step in run["steps"]],
                         ["finish", "finish", "hypothesis", "experiment", "verify", "finish"])
        self.assertTrue(run["experiments"][0]["verification"])

    def test_resumed_hypothesis_runs_in_stage_restricted_to_inspect(self):
        engine = self.engine(workflow_mode="plan", use_memory=False, max_steps=12)
        engine.run["evidence_collected"] = True
        engine.apply("plan", {"feasible": True, "method": "数值模拟", "limitations": "",
                              "literature_query": "", "research_kind": "simulation"})
        engine.apply("hypothesis", {"statement": "均值误差小于0.5", "prediction": "误差较小",
                                    "controls": "固定采样规模", "metric": "error", "relation": "lt",
                                    "threshold": 0.5})
        engine.run["plan"], engine.run["research_kind"] = None, None
        engine.run["workflow"] = {"source": "fixture", "stages": workflow.validate([
            {"role": "experimenter", "goal": "运行模拟", "why": "数值检验", "max_steps": 1, "tools": ["inspect"]},
            {"role": "verifier", "goal": "复核", "why": "独立重跑", "max_steps": 1},
            {"role": "reporter", "goal": "报告", "why": "收束", "max_steps": 1},
        ])}
        offered = []

        def decide(choices):
            offered.append(engine.available_tools())
            return next(choices)

        decisions = iter([
            {"tool": "experiment", "args": {"hypothesis_id": "h1", "code":
                "values = rng.normal(size=200)\nemit_result({'metrics': {'error': float(abs(np.mean(values)))}})"}},
            {"tool": "verify", "args": {"experiment_id": "x1"}},
            {"tool": "finish", "args": {"conclusion": "仅在所设模拟机制中完成了独立复核。",
                                      "citations": ["h1", "x1"], "lesson": ""}},
        ])
        engine.decide = lambda: decide(decisions)
        run = engine.execute()
        self.assertIn("experiment", offered[0])
        self.assertEqual([step["tool"] for step in run["steps"]], ["experiment", "verify", "finish"])
        self.assertEqual(run["status"], "completed")

    def test_stage_without_available_tools_does_not_call_the_model(self):
        engine = self.engine(workflow_mode="plan", use_memory=False)
        engine.run["evidence_collected"] = True
        engine.apply("plan", {"feasible": True, "method": "比较来源", "limitations": "",
                              "literature_query": "", "research_kind": "literature"})
        engine.run["workflow"] = {"source": "fixture", "stages": workflow.validate([
            {"role": "generalist", "goal": "检查材料", "why": "误限定工具", "max_steps": 3, "tools": ["inspect"]},
            {"role": "reporter", "goal": "报告缺口", "why": "收束", "max_steps": 1},
        ])}
        roles_asked = []

        def decide():
            roles_asked.append(engine.run["active_agent"]["role"])
            return {"tool": "finish", "args": {"conclusion": "没有取得证据，研究未完成。", "citations": [], "lesson": ""}}

        engine.decide = decide
        run = engine.execute()
        self.assertEqual(roles_asked, ["reporter"])
        self.assertEqual([step["tool"] for step in run["steps"]], ["finish"])
        self.assertEqual(run["agents"][0]["status"], "skipped")

    def test_workflow_designer_sees_role_tool_ceilings(self):
        engine = self.engine(workflow_mode="plan", use_memory=False)
        engine.run["active_agent"] = {"id": "g0", "role": "planner"}
        captured = {}

        def complete(system, prompt, schema, **kwargs):
            captured["prompt"] = prompt
            return {"tool": "inspect", "reason": "先看材料", "args": {}}, {"source": "fixture"}

        engine.complete = complete
        engine.agent_step()
        ceilings = captured["prompt"]["role_tools"]
        self.assertIn("experiment", ceilings["experimenter"])
        self.assertEqual(ceilings["verifier"][0], "verify")
        self.assertNotIn("inspect", ceilings["verifier"])
        with self.assertRaises(ValueError) as raised:
            workflow.validate([{"role": "verifier", "goal": "复核", "why": "重跑", "tools": ["inspect"]}])
        self.assertIn("verify", str(raised.exception))

    def test_wrap_up_reached_mid_stage_accepts_finish(self):
        engine = self.engine(workflow_mode="plan", use_memory=False)
        engine.run["evidence_collected"] = True
        engine.apply("plan", {"feasible": True, "method": "比较来源", "limitations": "",
                              "literature_query": "", "research_kind": "literature"})
        engine.run["workflow"] = {"source": "fixture", "stages": workflow.validate([
            {"role": "analyst", "goal": "检索", "why": "取证", "max_steps": 3},
            {"role": "reporter", "goal": "报告", "why": "收束", "max_steps": 1},
        ])}
        engine.in_wrap_up = lambda: len(engine.run["steps"]) >= 1
        decisions = iter([
            {"tool": "literature", "args": {"query": "测试检索"}},
            {"tool": "finish", "args": {"conclusion": "预算不足，未取得可核验证据，研究未完成。", "citations": [], "lesson": ""}},
        ])
        engine.decide = lambda: next(decisions)
        with patch("researchloop.engine.core.tools.literature", return_value=[]):
            run = engine.execute()
        self.assertEqual([step["tool"] for step in run["steps"]], ["literature", "finish"])
        self.assertEqual(run["steps"][1]["status"], "completed")

    def test_follow_up_resets_previous_evidence_gap(self):
        engine = self.engine(use_memory=False)
        engine.run["evidence_collected"] = True
        engine.apply("plan", {"feasible": False, "method": "先取证", "limitations": "缺正文",
                              "literature_query": "", "research_kind": "literature"})
        self.assertTrue(engine.run["gap_fill"]["required"])
        engine.run["status"] = "inconclusive"
        follow = core.resume(engine.run, engine.folder, "新的追问：换一个完全不同的问题来研究")
        self.assertFalse((follow.run.get("gap_fill") or {}).get("required"))
        self.assertIsNone(follow.run.get("feasibility"))

    def test_stage_that_used_its_tools_is_not_marked_skipped(self):
        engine = self.engine(workflow_mode="plan", use_memory=False)
        engine.run["evidence_collected"] = True
        engine.apply("plan", {"feasible": True, "method": "数值模拟", "limitations": "",
                              "literature_query": "", "research_kind": "simulation"})
        engine.apply("hypothesis", {"statement": "均值误差小于0.5", "prediction": "误差较小", "controls": "固定规模",
                                    "metric": "error", "relation": "lt", "threshold": 0.5})
        engine.apply("experiment", {"hypothesis_id": "h1", "code":
            "values = rng.normal(size=200)\nemit_result({'metrics': {'error': float(abs(np.mean(values)))}})"})
        engine.run["workflow"] = {"source": "fixture", "stages": workflow.validate([
            {"role": "verifier", "goal": "复核", "why": "独立重跑", "max_steps": 3, "tools": ["verify"]},
            {"role": "reporter", "goal": "报告", "why": "收束", "max_steps": 1},
        ])}
        decisions = iter([
            {"tool": "verify", "args": {"experiment_id": "x1"}},
            {"tool": "finish", "args": {"conclusion": "仅在所设模拟机制中完成了独立复核。",
                                      "citations": ["h1", "x1"], "lesson": ""}},
        ])
        engine.decide = lambda: next(decisions)
        run = engine.execute()
        self.assertEqual([step["tool"] for step in run["steps"]], ["verify", "finish"])
        self.assertEqual(run["agents"][0]["status"], "done")

    def test_pause_arriving_while_run_finishes_does_not_leave_pausing(self):
        from researchloop.sessions import SessionManager
        for moment in ("after_loop", "during_report"):
            engine = self.engine(use_memory=False)
            manager = SessionManager(max_parallel=1)
            manager.jobs[engine.run["id"]] = {"engine": engine}
            manager.running.add(engine.run["id"])
            replies = []
            engine.decide = lambda: {"tool": "finish", "args": {"conclusion": "没有证据，研究未完成。", "citations": [], "lesson": ""}}
            if moment == "after_loop":
                original = engine._loop
                def loop(*args, original=original, manager=manager, engine=engine, **kwargs):
                    original(*args, **kwargs)
                    replies.append(manager.pause(engine.run["id"]))
                engine._loop = loop
            else:
                original = engine.write_report
                def write(original=original, manager=manager, engine=engine):
                    replies.append(manager.pause(engine.run["id"]))
                    return original()
                engine.write_report = write
            run = engine.execute()
            self.assertNotEqual(run["status"], "pausing", moment)
            self.assertNotEqual(storage.read_json(engine.folder / "run.json")["status"], "pausing", moment)
            if moment == "during_report":
                self.assertFalse(replies[0]["requested"])

    def test_fact_memory_rejects_metadata_only_sources_and_reserved_predicates(self):
        engine = self.engine(use_memory=False)
        engine.run["sources"] = [
            {"id": "p1", "title": "只有标题", "evidence_scope": "只有元数据，不是结论证据"},
            {"id": "p2", "title": "有摘要", "abstract": "摘要内容", "evidence_scope": "摘要"}]
        fact = {"predicate": "测试关系", "object": "对象", "summary": "摘要支持的有限结论"}
        with self.assertRaises(ValueError):
            engine.apply("remember_fact", {**fact, "evidence_ids": ["p1"]})
        self.review(engine, "p2", "摘要内容")
        for predicate in ("user_preference", " User_Note ", "method_lesson"):
            with self.assertRaises(ValueError):
                engine.apply("remember_fact", {**fact, "predicate": predicate, "evidence_ids": ["p2"]})
        self.assertEqual(engine.apply("remember_fact", {**fact, "evidence_ids": ["p2"]})["status"], "pending_report")

    def test_repeated_literature_and_platform_hits_keep_one_evidence_id(self):
        engine = self.engine(use_memory=False)
        paper = {"title": "论文", "doi": "10.1/ABC", "url": "https://doi.org/10.1/abc", "abstract": "",
                 "evidence_scope": "只有元数据，不是结论证据"}
        with patch("researchloop.engine.core.tools.literature",
                   side_effect=[[dict(paper)], [{**paper, "doi": "10.1/abc", "abstract": "后来取得的摘要", "evidence_scope": "摘要"}]]):
            first = engine.apply("literature", {"query": "一次"})
            second = engine.apply("literature", {"query": "二次"})
        self.assertEqual(first[0]["id"], second[0]["id"])
        self.assertEqual(len(engine.run["sources"]), 1)
        self.assertEqual(engine.run["sources"][0]["abstract"], "后来取得的摘要")
        post = {"title": "帖子", "url": "https://tieba.example/p/1", "snippet": "内容"}
        with patch("researchloop.platforms.search", side_effect=lambda *a, **kw: {
                "label": "贴吧", "results": [dict(post)]}), \
                patch.object(engine, "available_platforms", return_value=["tieba"]):
            engine.apply("platform_search", {"platform": "tieba", "query": "一次"})
            engine.apply("platform_search", {"platform": "tieba", "query": "二次"})
        self.assertEqual(len(engine.run["sources"]), 2)

    def test_manuscript_document_citation_is_reconciled_with_real_evidence(self):
        from researchloop import reporting
        manuscript = {key: "内容" for key in reporting.FIELDS}
        manuscript["results"] = "文档正文支持该结论 [d1]。"
        paper, warnings, added = reporting.review_manuscript(manuscript, ["p1"], {"p1", "d1"})
        self.assertEqual(paper, manuscript)
        self.assertEqual(added, ["d1"])
        self.assertIn("补入 citations", warnings[0])
        self.assertEqual(reporting.review_manuscript(manuscript, ["d1"], {"d1"}), (paper, [], []))

    def test_repeated_experimenter_finish_still_reaches_gap_report(self):
        engine = self.engine(workflow_mode="plan", use_memory=False)
        engine.run["workflow"] = {"source": "fixture", "stages": workflow.validate([
            {"role": "experimenter", "goal": "运行模拟", "why": "数值检验", "max_steps": 2},
            {"role": "verifier", "goal": "复核", "why": "独立重跑", "max_steps": 1},
            {"role": "reporter", "goal": "说明实验缺口", "why": "诚实收尾", "max_steps": 1},
        ])}
        engine.run["plan"] = {"feasible": True, "method": "数值模拟", "research_kind": "simulation"}
        engine.run["research_kind"] = "simulation"
        engine.run["evidence_collected"] = True
        engine.decide = lambda: {"tool": "finish", "args": {
            "conclusion": "实验未运行，因此本次研究没有数值结论。", "citations": [], "lesson": ""}}
        run = engine.execute()
        self.assertEqual(run["status"], "incomplete")
        self.assertEqual([step["tool"] for step in run["steps"]],
                         ["finish", "finish", "finish", "finish"])
        self.assertTrue(all(step.get("rejected_by_role") for step in run["steps"][:3]))
        self.assertEqual(run["steps"][-1]["role"], "reporter")
        self.assertEqual(run["agents"][1]["status"], "skipped")
        self.assertIn("实验未运行", run["report"]["conclusion"])

    def test_reporter_has_a_finish_step_after_required_evidence_attempts(self):
        engine = self.engine(workflow_mode="plan", use_memory=False)
        engine.run["workflow"] = {"source": "fixture", "stages": workflow.validate([
            {"role": "reporter", "goal": "补查来源并报告缺口", "why": "完成取证", "max_steps": 3},
        ])}
        engine.apply("plan", {"feasible": False, "method": "尝试不同来源", "limitations": "没有正文",
                              "literature_query": "", "research_kind": "literature"})
        decisions = iter([
            {"tool": "literature", "args": {"query": f"测试检索词{i}"}} for i in range(3)
        ] + [{"tool": "finish", "args": {
            "conclusion": "已尝试三种检索词，但未取得可用于核验的正文，因此本次研究未完成。",
            "citations": [], "lesson": ""}}])
        engine.decide = lambda: next(decisions)
        with patch("researchloop.engine.core.tools.literature", return_value=[]):
            run = engine.execute()
        self.assertEqual([step["tool"] for step in run["steps"]],
                         ["literature", "literature", "literature", "finish"])
        self.assertEqual(run["status"], "inconclusive")
        self.assertFalse(run["gap_fill"]["required"])
        self.assertIn("未完成", run["report"]["conclusion"])

    def test_fact_memory_waits_for_cited_completed_report(self):
        engine = self.engine()
        engine.run["research_kind"] = "qualitative"
        engine.run["sources"] = [{"id": "p1", "title": "测试正文", "evidence_scope": "网页正文",
                                  "abstract": "测试事实", "url": "https://example.org/a"}]
        self.review(engine, "p1", "测试事实")
        args = {"predicate": "测试关系", "object": "有限事实", "summary": "仅由测试正文支持",
                "evidence_ids": ["p1"]}
        with patch("researchloop.engine.extratools.facts.remember") as remember:
            pending = engine.apply("remember_fact", args)
            self.assertEqual(pending["status"], "pending_report")
            remember.assert_not_called()
            engine.apply("finish", {"conclusion": "测试正文支持这一有限事实。", "citations": ["p1"], "lesson": ""})
            remember.assert_called_once()

        other = self.engine()
        other.run["research_kind"] = "qualitative"
        other.run["sources"] = engine.run["sources"].copy()
        self.review(other, "p1", "测试事实")
        with patch("researchloop.engine.extratools.facts.remember") as remember:
            other.apply("remember_fact", args)
            other.apply("finish", {"conclusion": "报告没有引用该测试事实。", "citations": [], "lesson": ""})
            remember.assert_not_called()

    def test_unfinished_research_does_not_publish_pending_fact_or_report_tool_error_as_conclusion(self):
        engine = self.engine(max_steps=1)
        engine.run["sources"] = [{"id": "p1", "title": "测试来源", "url": "https://example.org/a",
                                  "abstract": "测试内容", "evidence_scope": "网页正文"}]
        self.review(engine, "p1", "测试内容")
        engine.apply("remember_fact", {"predicate": "测试事实", "object": "候选说法",
                                        "summary": "尚未进入完成报告", "evidence_ids": ["p1"]})
        engine.decide = lambda: {"tool": "read_file", "args": {"path": "example.org/missing"}}
        with patch("researchloop.engine.extratools.facts.remember") as remember:
            run = engine.execute()
            remember.assert_not_called()
        self.assertEqual(run["status"], "inconclusive")
        self.assertEqual(run["pending_facts"][0]["status"], "not_published")
        self.assertIn("已收集 1 条来源", run["report"]["conclusion"])
        self.assertNotIn("example.org/missing", run["report"]["conclusion"])
        self.assertIn("missing", str(run["report"]["recovery"]["execution_errors"]))

    def test_metric_must_be_a_short_identifier(self):
        engine = self.engine()
        engine.run["evidence_collected"] = True
        with self.assertRaises(ValueError):
            engine.apply("hypothesis", {"statement": "s", "prediction": "p", "controls": "c",
                                        "metric": "过滤相对未过滤的下游线性探测精度绝对增益",
                                        "relation": "gt", "threshold": 1.0})
        item = engine.apply("hypothesis", {"statement": "s", "prediction": "p", "controls": "c",
                                           "metric": "delta_accuracy", "relation": "gt", "threshold": 1.0})
        self.assertEqual(item["metric"], "delta_accuracy")

    def test_infeasible_plan_keeps_collecting_evidence_before_finishing(self):
        engine = self.engine()
        engine.apply("plan", {"feasible": False, "method": "先取证再评估", "limitations": "缺少平台数据",
                              "literature_query": "", "research_kind": "qualitative"})
        self.assertNotIn(engine.run["status"], ("inconclusive", "completed", "incomplete"))
        self.assertTrue(engine.run["gap_fill"]["required"])
        self.assertIn("finish", engine.available_tools())
        early = engine.apply("finish", {"conclusion": "证据不足，无法回答。", "citations": [], "lesson": ""})
        self.assertEqual(early["status"], "inconclusive")
        self.assertIn("证据缺口", str(early["warnings"]))
        engine.note_evidence_attempt("web_search", {"query": "奶龙"}, "completed")
        engine.note_evidence_attempt("web_search", {"query": "奶龙"}, "completed")  # duplicate marker
        self.assertEqual(engine.run["gap_fill"]["attempts"], 1)
        engine.note_evidence_attempt("platform_search", {"platform": "bilibili", "query": "奶龙"}, "completed")
        engine.note_evidence_attempt("literature", {"query": "Nailong"}, "completed")
        self.assertFalse(engine.run["gap_fill"]["required"])
        self.assertIn("finish", engine.available_tools())
        result = engine.apply("finish", {"conclusion": "已尝试多种来源仍缺少可引用证据，报告未完成。",
                                         "citations": [], "lesson": ""})
        self.assertEqual(result["status"], "inconclusive")

    def test_search_tool_does_not_require_a_second_reason_in_args(self):
        engine = self.engine()
        base = storage.settings()
        web = {**base["web"], "enabled": True, "search_provider": "searxng",
               "search_endpoint": "https://search.example.com/search", "search_api_key": ""}
        payload = {"message": {"content": json.dumps({"tool": "web_search", "reason": "查平台",
                                                      "args": {"query": "奶龙"}})},
                   "prompt_eval_count": 30, "eval_count": 20}
        with patch("researchloop.storage.settings", return_value={**base, "web": web}), \
                patch("researchloop.engine.core.urllib.request.urlopen",
                      side_effect=lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode())):
            decision = engine.agent_step()
        self.assertEqual(decision["tool"], "web_search")
        self.assertEqual(decision["args"]["query"], "奶龙")

    def test_missing_selection_reason_is_normalized(self):
        engine = self.engine()
        payload = {"message": {"content": json.dumps({"tool": "inspect", "args": {}})},
                   "prompt_eval_count": 30, "eval_count": 20}
        with patch("researchloop.engine.core.urllib.request.urlopen",
                   return_value=io.BytesIO(json.dumps(payload).encode())):
            decision = engine.agent_step()
        self.assertEqual(decision["reason"], "模型未说明选择依据")

    def test_wrap_up_finish_is_not_vetoed_by_role(self):
        engine = self.engine()
        engine.run["workflow_mode"] = "plan"
        engine.run["active_agent"] = {"id": "g1", "role": "planner"}
        engine.run["plan"] = {"feasible": True, "method": "m", "limitations": "",
                              "literature_query": "", "research_kind": "literature"}
        engine.run["sources"] = [{"id": "p1", "title": "x", "url": "https://example.com/a"}]
        engine.run["steps"] = [{"id": "s1", "tool": "inspect", "status": "completed", "turn": "t1",
                                "agent": "g1", "elapsed_seconds": 0}]
        engine.budget.input_tokens = engine.run["token_limit"] - 500
        self.assertEqual(engine.available_tools(), ["finish"])
        self.assertIsNone(engine.allowed_tools())

    def test_reflect_is_only_offered_after_a_verified_experiment(self):
        engine = self.engine()
        engine.run["plan"] = {"feasible": True, "method": "m", "limitations": "",
                              "literature_query": "", "research_kind": "quantitative"}
        engine.run["hypotheses"] = [{"id": "h1", "metric": "error", "relation": "lt", "threshold": 1.0}]
        self.assertNotIn("reflect", engine.available_tools())
        engine.run["experiments"] = [{"id": "x1", "hypothesis_id": "h1", "status": "completed",
                                      "verification": {"result": {"metrics": {"error": 0.1}}, "check": {"passed": True}}}]
        self.assertIn("reflect", engine.available_tools())

    def test_reflect_without_verified_experiment_explains_what_to_do(self):
        engine = self.engine()
        engine.run["hypotheses"] = [{"id": "h1", "metric": "error", "relation": "lt", "threshold": 1.0}]
        with self.assertRaises(ValueError) as caught:
            engine.apply("reflect", {"next_action": "finish", "reason": "没有实验"})
        message = str(caught.exception)
        self.assertIn("没有已复核实验", message)
        self.assertIn("finish", message)

    def test_reflect_accepts_common_synonyms(self):
        engine = self.engine()
        engine.run["hypotheses"] = [{"id": "h1", "metric": "error", "relation": "lt", "threshold": 1.0}]
        engine.run["experiments"] = [{"id": "x1", "hypothesis_id": "h1", "status": "completed",
                                      "verification": {"result": {"metrics": {"error": 0.1}}, "check": {"passed": True}}}]
        item = engine.apply("reflect", {"next_action": "done", "reason": "复核通过，收束"})
        self.assertEqual(item["next_action"], "finish")
        self.assertEqual(engine.run["reflections"][-1]["after_experiment"], "x1")

    def test_gap_fill_keeps_evidence_tools_open_in_later_stages(self):
        engine = self.engine()
        engine.apply("plan", {"feasible": False, "method": "先取证", "limitations": "缺平台数据",
                              "literature_query": "", "research_kind": "qualitative"})
        engine.run["workflow_mode"] = "plan"
        engine.run["active_agent"] = {"id": "g3", "role": "reporter"}
        base = storage.settings()
        web = {**base["web"], "enabled": True, "search_provider": "searxng",
               "search_endpoint": "https://search.example.com/search", "search_api_key": ""}
        with patch("researchloop.storage.settings", return_value={**base, "web": web}):
            names = engine.available_tools()
        self.assertIn("finish", names)
        for tool in ("web_search", "platform_search", "literature"):
            self.assertIn(tool, names)
        allowed = engine.allowed_tools()
        self.assertIn("web_search", allowed)

    def test_custom_stage_tool_subset_limits_available_tools(self):
        engine = self.engine()
        engine.run["plan"] = {"feasible": True, "method": "m", "limitations": "",
                              "literature_query": "", "research_kind": "literature"}
        engine.run["active_agent"] = {"id": "g1", "role": "generalist", "tools": ["literature"]}
        self.assertEqual(engine.available_tools(), ["literature"])
        self.assertEqual(engine.allowed_tools(), {"literature"})
