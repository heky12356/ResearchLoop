"""Failure types control retries; evidence and budget gates survive recovery."""
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from researchloop import storage, reliability, db, memory, webaccess, llmjson
from researchloop.engine import core, history


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(dir=storage.ROOT / '.cache')
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for key in ('RUNS', 'MEMORY'):
            scope = patch.object(storage, key, root/key.lower())
            scope.start()
            self.addCleanup(scope.stop)
        self.cfg = storage.settings()
        self.cfg.update(model='test-local', revision='v1')
        self.cfg['web'] = {'enabled': True, 'search_provider': 'tavily',
                           'search_endpoint': 'https://search.example.com', 'search_api_key': 'test-key'}
        scope = patch.object(storage, 'settings', return_value=self.cfg)
        scope.start()
        self.addCleanup(scope.stop)

    def engine(self, **kwargs):
        return core.create({'question':'核实当前问题的实际前提与证据出处', 'model_source':'local',
                            'token_limit':40000, 'max_steps':6, **kwargs})

    def error(self, code, after=None):
        return urllib.error.HTTPError('https://search.example.com', code, 'provider failure',
                                      {'Retry-After': str(after)} if after is not None else {}, None)

    def results(self):
        return {'provider':'tavily','query':'证据出处','results':[{'title':'测试来源','url':'https://example.com/p', 'snippet':'真实工具返回的测试片段'}]}

    def search(self):
        return {'tool':'web_search','args':{'query':'证据出处','reason':'核实'},'reason':'收集证据'}

    def test_temporary_tool_failure_retries_once_and_keeps_both_records(self):
        engine = self.engine()
        with patch('researchloop.webaccess._search', side_effect=[self.error(503),self.results()]) as call:
            engine._perform_tool(self.search())
        self.assertEqual(call.call_count, 2)
        self.assertEqual([s['status'] for s in engine.run['steps']], ['failed','completed'])
        self.assertEqual([s['attempt'] for s in engine.run['trace']], [1,2])
        self.assertEqual(engine.run['trace'][1]['retry_of'], engine.run['trace'][0]['id'])
        self.assertEqual(engine.budget.used, 0)
        self.assertEqual(len(engine.run['sources']), 1)

    def test_authentication_failure_blocks_shared_search_provider_across_tools(self):
        engine = self.engine()
        decision = {'tool':'platform_search','args':{'platform':'zhihu','query':'出处','reason':'核实'}}
        with patch('researchloop.webaccess._search', side_effect=self.error(401)) as call:
            engine._perform_tool(decision)
            engine._perform_tool({**decision,'args':{**decision['args'],'query':'换关键词'}})
            engine._perform_tool(self.search())
        self.assertEqual(call.call_count, 1)
        self.assertTrue(reliability.blocked(engine.run, 'search:tavily', self.cfg))
        self.assertNotIn('web_search', engine.available_tools())
        self.assertIn('literature', engine.available_tools())
        self.assertIn('bilibili', engine.available_platforms())
        self.assertNotIn('zhihu', engine.available_platforms())

    def test_second_temporary_failure_opens_circuit_instead_of_repeating_forever(self):
        engine = self.engine()
        with patch('researchloop.webaccess._search', side_effect=self.error(500)) as call:
            engine._perform_tool(self.search())
        self.assertEqual(call.call_count, 2)
        self.assertNotIn('web_search', engine.available_tools())
        self.assertEqual(engine.run['steps'][-1]['observation']['recovery']['action'], 'switch_source')

    def test_revised_settings_reopen_previously_failed_provider(self):
        engine = self.engine()
        with patch('researchloop.webaccess._search', side_effect=self.error(401)):
            engine._perform_tool(self.search())
        self.cfg['revision'] = 'v2'
        self.assertIn('web_search', engine.available_tools())
        with patch('researchloop.webaccess._search', return_value=self.results()) as call:
            engine._perform_tool(self.search())
        self.assertEqual(call.call_count, 1)
        self.assertEqual(engine.run['steps'][-1]['status'], 'completed')

    def test_retry_after_exceeding_budget_is_not_ignored_or_slept(self):
        engine = self.engine(seconds_limit=10)
        with patch('researchloop.webaccess._search', side_effect=self.error(429,3600)) as call, patch.object(engine,'wait_for_recovery') as wait:
            engine._perform_tool(self.search())
        self.assertEqual(call.call_count, 1)
        wait.assert_not_called()
        self.assertEqual(engine.run['steps'][0]['observation']['recovery']['action'], 'defer')

    def test_global_step_limit_does_not_give_free_transport_retries(self):
        engine = self.engine(max_steps=1)
        with patch('researchloop.webaccess._search', side_effect=self.error(500)) as call:
            engine._perform_tool(self.search())
        self.assertEqual(call.call_count, 1)
        self.assertEqual(len(engine.run['steps']), 1)

    def test_provider_endpoint_mismatch_fails_before_any_network_request(self):
        with patch('urllib.request.urlopen') as request:
            with self.assertRaisesRegex(ValueError, '提供方'):
                webaccess.search('test',provider='searxng',endpoint='https://api.tavily.com/search',api_key='test-key')
        request.assert_not_called()
        self.cfg['web'].update(search_provider='searxng',search_endpoint='https://api.tavily.com/search',search_api_key='')
        engine=self.engine()
        self.assertNotIn('web_search',engine.available_tools())
        self.assertNotIn('zhihu',engine.available_platforms())
        self.assertEqual(engine.run['recovery_feedback']['kind'],'configuration')
        self.assertEqual(engine.run['calls'],[])

    def test_reasoning_model_final_json_is_parsed_without_executing_reasoning(self):
        content='<think>{"tool":"finish","args":{}} is only reasoning</think>\n```json\n{"tool":"inspect","reason":"检查","args":{}}\n```'
        self.assertEqual(llmjson.parse(content)['tool'],'inspect')
        with self.assertRaises(ValueError):
            llmjson.parse('<think>{"tool":"finish"}')
        with self.assertRaises(ValueError):
            llmjson.parse('<think>{"tool":"finish"}</think>')

    def test_pause_during_backoff_resumes_pending_tool_without_new_model_choice(self):
        engine = self.engine(max_steps=4)
        engine.decide = self.search
        def pause(_):
            engine.pause_requested.set()
            engine.guard_pause()
        with patch('researchloop.webaccess._search', side_effect=[self.error(503),self.results()]) as call:
            with patch.object(engine,'wait_for_recovery',side_effect=pause):
                self.assertEqual(engine.execute()['status'], 'paused')
            resumed = core.resume_paused(storage.read_json(engine.folder/'run.json'), engine.folder)
            resumed.decide = lambda: {'tool':'finish','args':{'conclusion':'该测试没有独立复核，不形成数值结论。','citations':[],'lesson':''}}
            result = resumed.execute()
        self.assertEqual(call.call_count, 2)
        self.assertEqual([s['tool'] for s in result['steps']], ['web_search','web_search','finish'])
        self.assertEqual(result['steps'][1]['attempt'], 2)

    def test_final_report_validation_gets_one_repair_before_stage_ends(self):
        engine = self.engine(workflow_mode='plan')
        engine.run['workflow'] = {'stages':[{'role':'reporter','goal':'如实收尾','max_steps':1}]}
        actions = iter([{'tool':'finish','args':{'conclusion':'仅有记忆而没有可引用的研究证据。','citations':['memory:3'],'lesson':''}},
                        {'tool':'finish','args':{'conclusion':'本轮未取得外部证据，问题的前提和出处尚未核实。','citations':[],'lesson':''}}])
        engine.decide = lambda: next(actions)
        result = engine.execute()
        self.assertEqual(len(result['steps']), 2)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['report']['citations'], [])
        self.assertTrue(result.get('draft_report'))
        self.assertEqual(result['recovery_extensions']['t1:g1'], 1)

    def test_exhausted_recovery_keeps_invalid_draft_separate_from_report(self):
        engine = self.engine(max_steps=1)
        engine.decide = lambda: {'tool':'finish','args':{'conclusion':'该断言没有可引用的研究证据。','citations':['memory:3'],'lesson':'不可验证'}}
        result = engine.execute()
        self.assertEqual(result['report']['citations'], [])
        self.assertIn('记忆不是证据', result['report']['conclusion'])
        self.assertFalse(result['draft_report']['verified'])
        self.assertEqual(memory.state()['methods']['versions'], [])
        # A rejected finish is a failed operation, not a completed turn boundary.
        state, _ = history.fork(result, engine.folder, result['trace'][-1]['id'])
        self.assertEqual(len(state['turns']), 1)

    def test_observed_question_falls_back_and_reports_unverified_premise(self):
        engine = self.engine(workflow_mode='plan')
        engine.run['question'] = engine.run['current_question'] = '意大利面为什么要拌45号混凝土'
        engine.run['workflow'] = {'stages':[
            {'role':'planner','goal':'核实提问前提与出处','max_steps':2},
            {'role':'reporter','goal':'如实说明未核实的部分','max_steps':1}]}
        def decide():
            if engine.run['active_agent']['role']=='planner':
                if not engine.run.get('resource_health'):
                    return {'tool':'platform_search','args':{'platform':'zhihu','query':engine.question_text(),'reason':'核实出处'}}
                self.assertNotIn('web_search',engine.available_tools())
                return {'tool':'literature','args':{'query':'question premise'}}
            if not engine.run.get('draft_report'):
                return {'tool':'finish','args':{'conclusion':'未取得可用于核验的来源，不能确定这个提问的具体语境。','citations':['memory:3'],'lesson':''}}
            return {'tool':'finish','args':{'conclusion':'原搜索服务鉴权失败，备用来源未取得证据，问题前提和出处尚未核实。','citations':[],'lesson':''}}
        engine.decide=decide
        with patch('researchloop.webaccess._search',side_effect=self.error(401)) as search, patch('researchloop.engine.tools.literature',return_value=[]) as papers:
            result=engine.execute()
        self.assertEqual(search.call_count,1)
        self.assertEqual(papers.call_count,1)
        self.assertEqual([s['tool'] for s in result['steps']],['platform_search','literature','finish','finish'])
        self.assertEqual(result['status'],'incomplete')
        self.assertIn('尚未核实',result['report']['conclusion'])
        self.assertEqual(result['calls'],[])
        self.assertEqual(db.facts(scope='research'),[])

    def test_model_schema_repairs_nonexistent_citation_without_executing_invalid_finish(self):
        engine = self.engine()
        actions = iter([{'tool':'finish','reason':'错误引用','args':{'conclusion':'没有来源的暂定判断，不应作为结论。','citations':['remember_fact:3'],'lesson':''}},
                        {'tool':'finish','reason':'修正引用','args':{'conclusion':'没有证据，问题前提尚未核实，当前未完成。','citations':[],'lesson':''}}])
        def respond(*args, **kwargs):
            return io.BytesIO(json.dumps({'message':{'content':json.dumps(next(actions))},'prompt_eval_count':40,'eval_count':20}).encode())
        with patch('urllib.request.urlopen', side_effect=respond):
            decision = engine.agent_step()
        self.assertEqual(decision['args']['citations'], [])
        self.assertEqual(engine.budget.used, 120)
        self.assertEqual(len(engine.run['calls']), 2)
        self.assertEqual(engine.run['steps'], [])

    def test_unverified_claim_cannot_be_promoted_into_fact_memory(self):
        engine = self.engine()
        self.assertNotIn('remember_fact', engine.available_tools())
        with self.assertRaises(ValueError):
            engine.apply('remember_fact',{'predicate':'classification','object':'unsupported','summary':'未核实的判断','evidence_ids':[]})
        self.assertEqual(db.facts(scope='research'), [])
        engine.run['sources'] = [{'id':'p1','title':'测试来源','evidence_scope':'网页正文',
                                  'content':'测试网页正文支持有限的观察。'}]
        pending = engine.apply('remember_fact',{'predicate':'observation','object':'test','summary':'有来源的测试观察','evidence_ids':['p1']})
        self.assertEqual(pending['status'], 'pending_report')
        self.assertEqual(db.facts(scope='research'), [])
        engine.run['research_kind'] = 'qualitative'
        engine.apply('finish',{'conclusion':'测试网页正文支持有限的观察。','citations':['p1'],'lesson':''})
        self.assertIn('p1', db.facts(scope='research')[0]['source_message'])

    def test_cloud_auth_failure_falls_back_to_configured_local_with_honest_usage(self):
        engine = self.engine()
        engine.run['model_source'] = 'cloud'
        self.cfg['cloud'] = {'enabled':True,'base_url':'https://model.example.com/v1','api_key':'test-key','model':'test-cloud'}
        def respond(request, **kwargs):
            if request.full_url.endswith('/chat/completions'):
                raise self.error(401)
            return io.BytesIO(json.dumps({'message':{'content':json.dumps({'tool':'inspect','reason':'回退检查','args':{}})},'prompt_eval_count':40,'eval_count':20}).encode())
        with patch('urllib.request.urlopen', side_effect=respond) as call:
            decision = engine.agent_step()
        self.assertEqual(decision['tool'],'inspect')
        self.assertEqual(call.call_count, 2)
        self.assertEqual([c['source'] for c in engine.run['calls']], ['cloud','ollama'])
        self.assertEqual(engine.run['calls'][-1]['route']['tier'], 'fallback')
        # Dynamic reservation: the failed cloud attempt consumes a conservative
        # prompt-sized reservation instead of a fixed 7644; local usage is added once.
        self.assertGreaterEqual(engine.budget.unknown_reserved, 2048)
        self.assertEqual(engine.budget.used, engine.budget.unknown_reserved + 60)
        self.assertEqual(engine.budget.cloud_calls, 1)


if __name__ == '__main__':
    unittest.main()
