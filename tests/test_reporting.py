"""Export fidelity and the final writing contract; no model or network research."""
import copy
import io
import hashlib
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from researchloop import reporting, storage, server
from researchloop.engine import core


def manuscript():
    return {"title": "排版测试：仅供验收的合成示例", "abstract": "本合成样例检验报告结构，不构成科研结果。",
            "introduction": "此处的来源仅为测试夹具，不代表真实文献。[p1]",
            "methods": "直接使用固定测试夹具验证导出；没有调用语言模型。",
            "results": "测试摘录中的陈述仅用于验证引用链。[p1]",
            "discussion": "未做实际实验；不主张统计或现实适用性。", "conclusion": "仅完成测试夹具的排版与导出。"}


class ReportingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=storage.ROOT / '.cache')
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        for target, key, value in [(storage, 'RUNS', root/'runs'), (server, 'RUNS', root/'runs'),
                                    (storage, 'MEMORY', root/'memory')]:
            scope = patch.object(target, key, value)
            scope.start()
            self.addCleanup(scope.stop)
        self.engine = core.create({'question': '报告结构导出测试，不进行真实研究', 'max_steps': 6})
        self.engine.run.update(research_kind='qualitative', plan={'method': '固定夹具验证', 'limitations': '不是真实研究'},
                               sources=[{'id': 'p1', 'title': '测试夹具来源', 'url': 'https://example.org/fixture',
                                         'abstract': '仅供测试的来源摘录。', 'evidence_scope': '网页正文（测试夹具）'}])
        self.engine.apply('assess_source', {'source_id': 'p1', 'verdict': 'supports',
                          'quote': '仅供测试的来源摘录。', 'claim': '测试来源确有该摘录',
                          'reason': '原文可核对', 'limitation': '仅适用于测试夹具'})

    def finish(self, paper=True):
        args = {'conclusion': '测试证据只适用于本测试夹具，不代表真实研究结论。', 'citations': ['p1'], 'lesson': ''}
        if paper:
            args['manuscript'] = manuscript()
        self.engine.decide = lambda: {'tool': 'finish', 'args': args}
        return self.engine.execute()

    def test_structured_paper_saved_in_turn_without_truncation(self):
        paper = manuscript()
        text = '仅用于验证长结论保存。' * 220
        self.engine.decide = lambda: {'tool': 'finish', 'args': {'conclusion': text, 'citations': ['p1'],
                                                               'lesson': '', 'manuscript': paper}}
        result = self.engine.execute()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['turns'][-1]['conclusion'], text)
        self.assertEqual(result['turns'][-1]['report']['manuscript'], paper)
        md = (self.engine.folder/'report.md').read_text(encoding='utf-8')
        self.assertIn('## 摘要', md)
        self.assertIn(paper['methods'], md)
        self.assertNotIn('## Agent分析', md)

    def test_invented_inline_citation_is_rejected_before_completion(self):
        paper = manuscript()
        paper['results'] = '不存在的来源不能支持这句话。[p999]'
        before = self.engine.run['status']
        with self.assertRaisesRegex(ValueError, 'p999'):
            self.engine.apply('finish', {'conclusion': '必须核查实际引用，不能凭空完成。', 'citations': ['p1'],
                                         'lesson': '', 'manuscript': paper})
        self.assertEqual(self.engine.run['status'], before)
        self.assertNotIn('report', self.engine.run)

    def test_final_prompt_includes_content_and_actual_measurements(self):
        action = {'tool': 'finish', 'reason': '按证据收尾', 'args': {'conclusion': '只报告测试来源所支持的有限陈述。',
                                                                'citations': ['p1'], 'lesson': ''}}
        self.engine.run['experiments'] = [{'id': 'x1', 'result': {'metrics': {'test_value': .125}}}]
        with patch.object(self.engine, 'available_tools', return_value=['finish']), \
                patch.object(self.engine, 'complete', return_value=(action, {'source': 'fixture'})) as complete:
            self.engine.agent_step()
        system, prompt, schema = complete.call_args.args
        self.assertIn('writing-cs-report', system)
        self.assertEqual(prompt['report_evidence'][0]['excerpt'], '仅供测试的来源摘录。')
        self.assertEqual(prompt['report_measurements'][0]['result']['metrics']['test_value'], .125)
        self.assertIn('manuscript', schema['oneOf'][0]['properties']['args']['properties'])

    def test_legacy_report_falls_back_honestly_and_does_not_mutate_run(self):
        run = self.finish(paper=False)
        before = copy.deepcopy(run)
        doc = reporting.document(run)
        self.assertIn('未保存结构化论文正文', '\n'.join(doc['appendix']))
        self.assertEqual(run, before)
        run['report'] = {'conclusion': '没有证据，尚未完成。', 'citations': [], 'evidence_gaps': ['正文不可用']}
        run['status'] = 'inconclusive'
        doc = reporting.document(run)
        self.assertEqual(doc['references'], [])
        self.assertIn('证据不足', '\n'.join(doc['appendix']))
        self.assertIn('正文不可用', str(doc['sections']))

    def test_results_table_distinguishes_negative_unverified_and_auxiliary_metrics(self):
        run = self.finish()
        run['hypotheses'] = [{'id':'h1','statement':'固定测试判据','controls':'夹具','metric':'score','relation':'gt','threshold':.5}]
        run['experiments'] = [
            {'id':'x1','hypothesis_id':'h1','result':{'metrics':{'score':.25,'auxiliary':10}},
             'verification':{'result':{'metrics':{'score':.2}},'check':{'passed':False}}},
            {'id':'x2','hypothesis_id':'h1','result':{'metrics':{'score':.75}}}]
        rows = reporting.document(run)['measurements']
        self.assertEqual(rows[0][-1], '未满足判据')
        self.assertEqual(rows[1][-1], '非判据指标')
        self.assertEqual(rows[2][3:], ['未记录', '未复核'])

    def test_latex_escapes_untrusted_content(self):
        run = self.finish()
        run['report']['manuscript']['title'] = r'测试 50% & x_y {z} \input{secret}'
        text = reporting.latex(reporting.document(run))
        self.assertIn(r'50\% \& x\_y \{z\}', text)
        self.assertIn(r'\textbackslash{}input\{secret\}', text)
        self.assertNotIn(r'\input{secret}', text)
        self.assertIn(r'\documentclass[UTF8,a4paper,11pt]{ctexart}', text)

    def test_http_pdf_tex_md_downloads_preserve_history_and_do_not_call_model(self):
        run = self.finish()
        original = (self.engine.folder/'report.md').read_bytes()
        run_hash = hashlib.sha256((self.engine.folder/'run.json').read_bytes()).hexdigest()
        api = ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        threading.Thread(target=api.serve_forever, daemon=True).start()
        self.addCleanup(api.server_close)
        self.addCleanup(api.shutdown)
        url = f'http://127.0.0.1:{api.server_port}/api/runs/{run["id"]}/report'
        with patch.object(core.ResearchEngine, 'complete', side_effect=AssertionError('Export must not call a model')):
            for suffix, kind in [('md','text/markdown'), ('tex','application/x-tex'), ('pdf','application/pdf')]:
                with urllib.request.urlopen(url+'?format='+suffix) as response:
                    body = response.read()
                    self.assertIn(kind, response.headers['Content-Type'])
                    self.assertIn('.'+suffix, response.headers['Content-Disposition'])
                if suffix == 'pdf':
                    self.assertTrue(body.startswith(b'%PDF-'))
                    self.assertTrue(body.rstrip().endswith(b'%%EOF'))
                elif suffix == 'md':
                    self.assertEqual(body, original)
                else:
                    self.assertIn('研究方法', body.decode('utf-8'))
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url+'?format=exe')
        self.assertEqual(error.exception.code, 400)
        self.assertEqual((self.engine.folder/'report.md').read_bytes(), original)
        self.assertEqual(hashlib.sha256((self.engine.folder/'run.json').read_bytes()).hexdigest(), run_hash)
        package=self.engine.folder/'workspace/.packages';package.mkdir(parents=True,exist_ok=True)
        (package/'large-dependency-fixture.txt').write_text('excluded dependency fixture')
        (package.parent/'program.py').write_text('print(42)')
        with urllib.request.urlopen(url.removesuffix('/report')+'/export') as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
                self.assertIn('workspace/program.py',archive.namelist())
                self.assertIn('EXPORT-NOTES.txt',archive.namelist())
                self.assertFalse(any('/.packages/' in name for name in archive.namelist()))


if __name__ == '__main__':
    unittest.main()
