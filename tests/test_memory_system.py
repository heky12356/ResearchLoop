import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from researchloop import storage, memory, facts
from researchloop.engine import core


class MemorySystemTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(dir=storage.ROOT/'.cache')
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for key in ('RUNS', 'MEMORY'):
            scope = patch.object(storage, key, root/key.lower())
            scope.start()
            self.addCleanup(scope.stop)

    def test_new_research_automatically_receives_user_preferences(self):
        item = memory.add({'kind':'preference','text':'请用中文撰写结论并附来源'})
        run = core.create({'question':'研究采样均值误差随样本量的变化','model_source':'local'}).run
        self.assertEqual(run['memory_mode'], 'auto')
        self.assertIn(item['id'], [v['id'] for v in run['memory_context']['facts']])
        self.assertIn('中文', run['memory_context']['text'])

    def test_disabled_memory_is_not_injected_and_can_be_restored(self):
        item = memory.add({'kind':'preference','text':'报告优先给出局限'})
        memory.toggle(item['id'], False)
        self.assertEqual(memory.recall('采样误差')['facts'], [])
        memory.toggle(item['id'], True)
        self.assertEqual(memory.recall('采样误差')['facts'][0]['id'], item['id'])

    def test_recall_filters_unrelated_facts_and_retrieves_chinese_terms(self):
        facts.remember('s1','data_constraint','随机采样','随机采样仅使用合成数据')
        facts.remember('s2','data_constraint','建筑','建筑资料来源是书籍')
        rows = memory.recall('比较随机采样均值的误差')['facts']
        self.assertEqual([row['session_id'] for row in rows], ['s1'])

    def test_methods_keep_versions_but_only_relevant_enabled_method_is_used(self):
        folder = storage.MEMORY/'research'
        storage.write_json(folder/'v1.json', {'version':1,'question':'随机采样均值误差','lesson':'用独立种子重跑验证',
                           'source_run':'source','limitations':'合成数据'})
        core.activate(1)
        context = memory.recall('随机采样的均值误差如何验证')
        self.assertEqual(context['method']['version'], 1)
        self.assertIsNone(memory.recall('研究古建筑的榫卯连接历史')['method'])
        core.activate(0)
        self.assertIsNone(memory.recall('随机采样均值误差')['method'])
        self.assertEqual(len(memory.state()['methods']['versions']), 1)

    def test_user_note_cannot_be_presented_as_a_verified_method(self):
        with self.assertRaises(ValueError):
            memory.add({'kind':'method','text':'未经验证的方法'})
        self.assertEqual(memory.state()['methods']['versions'], [])

    def test_explicit_api_opt_out_remains_available_for_controlled_evaluation(self):
        memory.add({'kind':'preference','text':'结论用中文'})
        run = core.create({'question':'研究采样误差是否可以缩小','model_source':'local','use_memory':False}).run
        self.assertEqual(run['memory_mode'], 'disabled')
        self.assertEqual(run['memory_context']['facts'], [])

    def test_completed_evidence_report_saves_a_provisional_method_automatically(self):
        engine = core.create({'question':'研究公开报道是否支持当前的定性判断','model_source':'local','max_steps':1})
        engine.run['research_kind'] = 'qualitative'
        engine.run['sources'] = [{'id':'p1','title':'测试来源','abstract':'测试使用的网页证据',
                                 'evidence_scope':'网页搜索摘要（测试），不是同行评议'}]
        engine.apply('assess_source', {'source_id':'p1','verdict':'supports','quote':'测试使用的网页证据',
                     'claim':'仅支持测试判断','reason':'引句来自已存摘要','limitation':'仅为测试摘录'})
        engine.decide = lambda: {'tool':'finish','args':{'conclusion':'结论仅限于该测试来源，不作任何范围外推。',
                    'citations':['p1'],'lesson':'定性判断需要逐条标明来源范围'}}
        run = engine.execute()
        self.assertEqual(run['status'], 'completed')
        self.assertEqual(run['memory_saved_version'], 1)
        self.assertEqual(memory.state()['methods']['versions'][0]['status'], 'provisional')
        self.assertEqual(core.publish(run)['version'], 1)
