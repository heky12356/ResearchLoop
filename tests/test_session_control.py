"""Durable invocation boundaries, honest accounting and model routing contracts."""
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from researchloop import storage
from researchloop.engine import core, history, protocol
from researchloop.ollama import Budget


class SessionControlTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(dir=storage.ROOT / '.cache')
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for name in ('RUNS', 'MEMORY'):
            scope = patch.object(storage, name, self.root / name.lower())
            scope.start()
            self.addCleanup(scope.stop)

    def engine(self, **kwargs):
        return core.create({'question': '检查随机样本均值的误差和证据是否可靠', 'max_steps': 5,
                            'model_source': 'local', 'model': 'test-light', **kwargs})

    def response(self, action, **kw):
        payload = {'message': {'content': json.dumps(action)}, 'prompt_eval_count': 60, 'eval_count': 20, **kw}
        return io.BytesIO(json.dumps(payload).encode())

    def test_pause_during_model_preserves_decision_and_resume_does_not_repay(self):
        engine = self.engine()
        def request(*a, **kw):
            engine.pause_requested.set()
            return self.response({'tool': 'inspect', 'reason': '检查输入', 'args': {}})
        with patch('urllib.request.urlopen', side_effect=request):
            run = engine.execute()
        self.assertEqual(run['status'], 'paused')
        self.assertEqual(run['steps'], [])
        self.assertEqual(len(run['calls']), 1)
        self.assertEqual(run['budget']['total_tokens'], 80)
        self.assertEqual(run['pending_decision']['tool'], 'inspect')
        before = run['budget']['elapsed_seconds']
        time.sleep(.02)
        resumed = core.resume_paused(storage.read_json(engine.folder / 'run.json'), engine.folder)
        # Pause at the next safe tool boundary, before any new model request.
        original = resumed.apply
        def apply(*args):
            result = original(*args)
            resumed.pause_requested.set()
            return result
        resumed.apply = apply
        with patch('urllib.request.urlopen', side_effect=AssertionError('must not request again')):
            result = resumed.execute()
        self.assertEqual(result['status'], 'paused')
        self.assertEqual(result['steps'][0]['tool'], 'inspect')
        self.assertEqual(len(result['calls']), 1)
        self.assertEqual(result['budget']['total_tokens'], 80)
        self.assertGreaterEqual(result['budget']['elapsed_seconds'], before)
        self.assertEqual(len(result['turns']), 1)

    def test_fork_model_boundary_reuses_response_without_future_evidence(self):
        engine = self.engine()
        with patch('urllib.request.urlopen', return_value=self.response({'tool': 'inspect', 'reason': '检查', 'args': {}})):
            engine.agent_step()
        node = engine.run['trace'][0]
        engine.run['sources'].append({'id': 'future', 'title': '未来证据'})
        state, folder = history.fork(engine.run, engine.folder, node['id'])
        self.assertEqual(state['sources'], [])
        self.assertEqual(state['lineage']['precision'], 'checkpoint')
        self.assertNotEqual(state['id'], engine.run['id'])
        self.assertEqual(state['budget'], {})
        branch = core.resume_paused(state, folder)
        with patch('urllib.request.urlopen', side_effect=AssertionError('must reuse model response')):
            decision = branch.agent_step()
        self.assertEqual(decision['tool'], 'inspect')
        self.assertEqual(branch.budget.used, 0)
        self.assertEqual(engine.run['sources'][0]['id'], 'future')
        changed, _ = history.fork(engine.run, engine.folder, node['id'], '换一种方法重新分析')
        self.assertNotIn('pending_response', changed)
        self.assertNotIn('pending_decision', changed)

    def test_tool_checkpoint_is_after_settlement_and_excludes_future_results(self):
        engine = self.engine(assets=[{'name': 'data.txt', 'content': 'frozen input'}])
        engine.controller_step('inspect', {})
        node = engine.run['trace'][-1]
        (engine.folder / 'inputs' / 'a1').write_text('changed later', encoding='utf-8')
        engine.run['hypotheses'].append({'id': 'future-hypothesis'})
        state, folder = history.fork(engine.run, engine.folder, node['id'])
        self.assertEqual(len(state['steps']), 1)
        self.assertEqual(state['hypotheses'], [])
        self.assertEqual((folder / 'inputs' / 'a1').read_text(), 'frozen input')
        self.assertEqual(state['trace'][-1]['inherited_from'], engine.run['id'])
        # A second branch from an inherited record remains independent of parent files.
        child, _ = history.fork(state, folder, node['id'])
        self.assertEqual(child['lineage']['run_id'], state['id'])
        self.assertEqual(child['hypotheses'], [])

    def test_trace_inputs_are_frozen_when_live_prompt_is_changed(self):
        engine = self.engine()
        prompt = {'observations': []}
        with patch('urllib.request.urlopen', return_value=self.response({'ok': True})):
            engine.complete('system', prompt, {'type': 'object'})
        prompt['observations'].append('future')
        self.assertEqual(engine.run['trace'][0]['input']['prompt']['observations'], [])

    def test_budget_restore_excludes_paused_time(self):
        with patch('researchloop.ollama.time.monotonic', return_value=1000):
            budget = Budget(10000, 60)
            budget.restore({'elapsed_seconds': 12, 'input_tokens': 20, 'output_tokens': 5}, [{}])
            self.assertEqual(budget.elapsed, 12)
            self.assertEqual(budget.used, 25)

    def test_legacy_fork_clearly_replans_without_latest_state(self):
        engine = self.engine()
        engine.run['steps'] = [{'id': 's1', 'tool': 'inspect', 'observation': {'seen': 'old'}},
                               {'id': 's2', 'tool': 'inspect', 'observation': {'seen': 'future'}}]
        engine.run['sources'] = [{'id': 'future'}]
        state, _ = history.fork(engine.run, engine.folder, 's1')
        self.assertEqual(state['lineage']['precision'], 'legacy_context')
        self.assertEqual(state['steps'], [])
        self.assertEqual(state['sources'], [])
        self.assertEqual(len(state['branch_context']), 1)
        self.assertNotIn('future', json.dumps(state['branch_context']))

    def test_invalid_json_keeps_known_usage_instead_of_double_charging(self):
        engine = self.engine()
        with patch('urllib.request.urlopen', return_value=self.response({}, message={'content': '{broken'})):
            with self.assertRaises(ValueError):
                engine.complete('system', {}, {})
        self.assertEqual(engine.budget.used, 80)
        self.assertEqual(engine.run['calls'][0].get('unknown_reserved', 0), 0)
        self.assertEqual(engine.run['trace'][0]['status'], 'failed')

    def test_cloud_invalid_json_is_recorded_with_actual_usage(self):
        engine = self.engine()
        engine.run['model_source'] = 'cloud'
        cfg = storage.settings()
        cfg['cloud'] = {'enabled': True, 'base_url': 'https://example.com/v1', 'model': 'test', 'api_key': 'test-secret'}
        response = {'usage': {'prompt_tokens': 10, 'completion_tokens': 5},
                    'choices': [{'message': {'content': 'broken'}}]}
        with patch.object(storage, 'settings', return_value=cfg), patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
            with self.assertRaises(RuntimeError):
                engine.complete('system', {}, {'type': 'object'})
        self.assertEqual(engine.budget.used, 15)
        self.assertEqual(len(engine.run['calls']), 1)
        self.assertEqual(engine.run['trace'][0]['status'], 'failed')
        self.assertNotIn('test-secret', json.dumps(engine.run))

    def test_cloud_truncated_report_retries_with_complete_short_answer(self):
        engine = self.engine()
        engine.run['model_source'] = 'cloud'
        engine.run['active_agent'] = {'id': 'g1', 'role': 'reporter',
                                       'model': 'test-cloud', 'model_source': 'cloud'}
        cfg = storage.settings()
        cfg['cloud'] = {'enabled': True, 'base_url': 'https://example.com/v1',
                        'model': 'test-cloud', 'api_key': 'test-secret'}
        truncated = {'usage': {'prompt_tokens': 100, 'completion_tokens': 3135,
                               'completion_tokens_details': {'reasoning_tokens': 1635}},
                     'choices': [{'finish_reason': 'length', 'message': {
                         'content': '{"tool":"finish","args":{"manuscript":{"discussion":"unfinished'}}]}
        answer = {'tool': 'finish', 'reason': '根据现有证据收尾',
                  'args': {'conclusion': '本轮资料不足，尚不能形成已核验的结论。',
                           'citations': [], 'lesson': ''}}
        complete = {'usage': {'prompt_tokens': 120, 'completion_tokens': 70},
                    'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer)}}]}
        requests = []
        responses = iter([truncated, complete])

        def respond(request, **kwargs):
            requests.append(json.loads(request.data))
            return io.BytesIO(json.dumps(next(responses)).encode())

        with patch.object(storage, 'settings', return_value=cfg), \
                patch('urllib.request.urlopen', side_effect=respond):
            decision = engine.agent_step()
        self.assertEqual(decision['tool'], 'finish')
        self.assertEqual(decision['args']['citations'], [])
        self.assertEqual([item['status'] for item in engine.run['calls']], ['error', 'ok'])
        self.assertEqual(engine.budget.used, 100 + 3135 + 120 + 70)
        self.assertEqual(engine.budget.cloud_calls, 2)
        self.assertEqual([request['max_tokens'] for request in requests], [8192, 8192])
        self.assertIn('省略可选manuscript', engine.run['trace'][1]['input']['prompt']['repair']['instruction'])
        self.assertEqual(engine.run['trace'][0]['status'], 'failed')
        self.assertNotIn('test-secret', json.dumps(engine.run))

    def test_cloud_repeated_truncation_does_not_execute_partial_json(self):
        engine = self.engine()
        engine.run['model_source'] = 'cloud'
        engine.run['active_agent'] = {'id': 'g1', 'role': 'reporter',
                                       'model': 'test-cloud', 'model_source': 'cloud'}
        cfg = storage.settings()
        cfg['cloud'] = {'enabled': True, 'base_url': 'https://example.com/v1',
                        'model': 'test-cloud', 'api_key': 'test-secret'}
        response = {'usage': {'prompt_tokens': 10, 'completion_tokens': 40},
                    'choices': [{'finish_reason': 'length',
                                 'message': {'content': '{"tool":"finish","args":{"conclusion":"partial'}}]}
        with patch.object(storage, 'settings', return_value=cfg), \
                patch('urllib.request.urlopen', side_effect=lambda *a, **kw: io.BytesIO(json.dumps(response).encode())):
            with self.assertRaisesRegex(RuntimeError, 'finish_reason=length'):
                engine.agent_step()
        self.assertEqual(engine.run['report'] if 'report' in engine.run else None, None)
        self.assertEqual(engine.budget.used, 100)
        self.assertEqual(len(engine.run['calls']), 2)

    def test_retry_routes_to_stronger_model_and_has_exact_repair_feedback(self):
        engine = self.engine(routing={'enabled': True, 'profiles': {
            'light': {'model_source': 'local', 'model': 'test-light'},
            'strong': {'model_source': 'local', 'model': 'test-strong'}}})
        outputs = iter([{'tool': '', 'args': {}}, {'tool': 'inspect', 'reason': '已修复', 'args': {}}])
        with patch('urllib.request.urlopen', side_effect=lambda *a, **kw: self.response(next(outputs))):
            decision = engine.agent_step()
        self.assertEqual(decision['tool'], 'inspect')
        self.assertEqual([c['model'] for c in engine.run['calls']], ['test-light', 'test-strong'])
        self.assertIn('invalid_response', engine.run['trace'][2]['input']['prompt']['repair'])
        self.assertNotIn('只按本次指定的工具参数schema返回JSON', engine.run['trace'][0]['input']['system'])

    def test_recursive_validation_rejects_wrong_nested_types_and_missing_fields(self):
        engine = self.engine()
        catalog = engine.tool_catalog(['finish', 'design_workflow', 'hypothesis'])
        with self.assertRaises(ValueError):
            protocol.validate({'conclusion': 'x', 'citations': 'p1', 'lesson': ''}, catalog[0]['args_schema'])
        with self.assertRaises(ValueError):
            protocol.validate({'reason': 'x', 'stages': [{'role': 'planner', 'goal': 'x'}]}, catalog[1]['args_schema'])
        self.assertNotIn('experiment', engine.available_tools())
        self.assertNotIn('verify', engine.available_tools())

    def test_resuming_workflow_keeps_stage_and_completed_work(self):
        engine = self.engine(workflow_mode='plan')
        engine.run['workflow'] = {'stages': [{'role': 'planner', 'goal': '检查', 'max_steps': 2}]}
        def decide():
            engine.pause_requested.set()
            return {'tool': 'inspect', 'args': {}}
        engine.decide = decide
        paused = engine.execute()
        self.assertEqual(paused['status'], 'paused')
        resumed = core.resume_paused(storage.read_json(engine.folder / 'run.json'), engine.folder)
        def finish():
            return {'tool': 'finish', 'args': {'conclusion': '当前材料不足，无法形成可核验的结论。', 'citations': [], 'lesson': ''}}
        resumed.decide = finish
        result = resumed.execute()
        self.assertEqual([s['tool'] for s in result['steps']], ['inspect', 'finish'])
        self.assertEqual(len(result['agents']), 1)
        self.assertEqual(result['agents'][0]['id'], 'g1')


if __name__ == '__main__':
    unittest.main()
