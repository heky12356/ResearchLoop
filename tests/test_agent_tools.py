"""Real local tool execution + deterministic delegated decisions, no model calls."""
import copy
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from researchloop import storage
from researchloop.engine import core, engineering, history, workflow, routing
from researchloop.engine.permissions import PermissionDenied
from researchloop.ollama import BudgetError


class AgentToolTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(dir=storage.ROOT/'.cache')
        self.addCleanup(temp.cleanup)
        root=Path(temp.name)
        for key in ('RUNS','MEMORY'):
            scope=patch.object(storage,key,root/key.lower());scope.start();self.addCleanup(scope.stop)
        self.engine=core.create({'question':'工具测试夹具，不进行真实科研或模型调用', 'budget_enabled':False})
        self.engine.run['routing']={'enabled':False,'profiles':{'light':{'model':'fixture-light','model_source':'local'},
                                                               'strong':{'model':'fixture-strong','model_source':'local'}}}

    def wait(self,predicate):
        until=time.monotonic()+10
        while time.monotonic()<until:
            if predicate():return
            time.sleep(.02)
        self.fail('Timed out waiting for fixture')

    def background(self,fn):
        result={}
        def run():
            try:result['value']=fn()
            except Exception as error:result['error']=error
        thread=threading.Thread(target=run);thread.start()
        self.addCleanup(lambda: self.engine.cancelled.set())
        self.addCleanup(lambda: thread.join(5))
        return thread,result

    def pending(self):
        return [r for r in self.engine.run.get('approval_requests',[]) if r['status']=='pending']

    def test_write_requires_approval_and_once_does_not_grant_future_writes(self):
        thread,result=self.background(lambda:self.engine.apply('write_file',{'path':'case.txt','content':'first'}))
        self.wait(lambda:self.pending())
        path=engineering.workspace(self.engine)/'case.txt'
        self.assertFalse(path.exists())
        self.engine.permission_gate.resolve(self.pending()[0]['id'],'once')
        thread.join(5);self.assertNotIn('error',result);self.assertEqual(path.read_text(),'first')
        thread,result=self.background(lambda:self.engine.apply('write_file',{'path':'case.txt','content':'second'}))
        self.wait(lambda:self.pending());self.assertEqual(path.read_text(),'first')
        self.engine.permission_gate.resolve(self.pending()[0]['id'],'deny')
        thread.join(5);self.assertIsInstance(result.get('error'),PermissionDenied)
        self.assertEqual(path.read_text(),'first')

    def test_category_grant_covers_waiting_and_future_writes_in_same_session(self):
        a,ra=self.background(lambda:self.engine.apply('write_file',{'path':'a.txt','content':'A'}))
        b,rb=self.background(lambda:self.engine.apply('write_file',{'path':'b.txt','content':'B'}))
        self.wait(lambda:len(self.pending())==2)
        self.engine.permission_gate.resolve(self.pending()[0]['id'],'category')
        a.join(5);b.join(5)
        self.assertNotIn('error',ra);self.assertNotIn('error',rb)
        self.engine.apply('write_file',{'path':'c.txt','content':'C'})
        self.assertEqual(self.pending(),[])
        self.assertIn('workspace_write',self.engine.run['permission_grants'])

    def test_pause_cancels_pending_write_without_side_effect(self):
        thread,result=self.background(lambda:self.engine.apply('write_file',{'path':'never.txt','content':'x'}))
        self.wait(lambda:self.pending());self.engine.pause_requested.set();thread.join(5)
        self.assertIsInstance(result.get('error'),history.Paused)
        self.assertFalse((engineering.workspace(self.engine)/'never.txt').exists())
        self.assertEqual(self.engine.run['approval_requests'][0]['status'],'cancelled')

    def test_waiting_write_does_not_overwrite_a_concurrent_user_edit(self):
        path=engineering.workspace(self.engine)/'edit.txt';path.write_text('initial')
        thread,result=self.background(lambda:self.engine.apply('write_file',{'path':'edit.txt','content':'model edit'}))
        self.wait(lambda:self.pending());path.write_text('user edit')
        self.engine.permission_gate.resolve(self.pending()[0]['id'],'once');thread.join(5)
        self.assertIn('文件已改变',str(result.get('error')));self.assertEqual(path.read_text(),'user edit')

    def test_scope_rejects_escape_and_full_mode_needs_no_requests(self):
        self.engine.run['access_mode']='full'
        for path in ('../run.json',str(self.engine.folder/'run.json')):
            with self.assertRaises(ValueError):self.engine.apply('read_file',{'path':path})
            with self.assertRaises(ValueError):self.engine.apply('write_file',{'path':path,'content':'no'})
        self.engine.apply('write_file',{'path':'code/test.py','content':'print(2)'})
        self.assertIn('print(2)',self.engine.apply('read_file',{'path':'code/test.py'})['content'])
        self.assertFalse(self.engine.run['approval_requests'])

    def test_uploaded_source_is_available_in_isolated_workspace(self):
        engine=core.create({'question':'上传源文件的工具夹具检查，无模型调用','assets':[
            {'name':'sample.py','content':'print(1)'},{'name':'sample.py','content':'print(2)'}]})
        self.assertEqual((engine.folder/'workspace/sample.py').read_text(),'print(1)')
        self.assertEqual((engine.folder/'workspace/sample-a2.py').read_text(),'print(2)')
        self.assertEqual(engine.run['approval_requests'],[])

    @unittest.skipUnless(engineering.shells().get('powershell'),'PowerShell unavailable')
    def test_powershell_actual_output_and_nonzero_exit_are_preserved(self):
        self.engine.run['access_mode']='full'
        result=self.engine.apply('run_command',{'shell':'powershell','command':'Write-Output (6 * 7)'})
        self.assertEqual(result['exit_code'],0);self.assertIn('42',result['output'])
        step=self.engine._perform_tool({'tool':'run_command','args':{'shell':'powershell','command':'Write-Output "fixture failure"; exit 7'}})
        self.assertEqual(step['status'],'failed');self.assertEqual(step['observation']['exit_code'],7)
        self.assertIn('fixture failure',step['observation']['output'])

    @unittest.skipUnless(engineering.shells().get('cmd'),'CMD unavailable')
    def test_command_approval_precedes_actual_process(self):
        thread,result=self.background(lambda:self.engine.apply('run_command',{'shell':'cmd','command':'echo approved>approved.txt'}))
        self.wait(lambda:self.pending());path=engineering.workspace(self.engine)/'approved.txt'
        self.assertFalse(path.exists());self.engine.permission_gate.resolve(self.pending()[0]['id'],'once')
        thread.join(8);self.assertNotIn('error',result);self.assertTrue(path.exists())
        self.assertNotIn('command_cmd',self.engine.run['permission_grants'])

    @unittest.skipUnless(engineering.shells().get('bash'),'Bash unavailable')
    def test_bash_actual_local_execution(self):
        self.engine.run['access_mode']='full'
        result=self.engine.apply('run_command',{'shell':'bash','command':'printf "bash fixture"'})
        self.assertEqual(result['exit_code'],0);self.assertIn('bash fixture',result['output'])

    @unittest.skipUnless(engineering.shells().get('powershell'),'PowerShell unavailable')
    def test_engineering_experiment_freezes_code_and_verifies_actual_output(self):
        self.engine.run['access_mode']='full'
        self.engine.apply('inspect',{})
        self.engine.apply('hypothesis',{'statement':'测试夹具平方和达到固定判据','prediction':'仅供集成测试','controls':'同一范围',
                                         'metric':'sum_squares','relation':'ge','threshold':280})
        self.engine.apply('write_file',{'path':'.packages/fixture_helper.py','content':'def total():\n    return sum(i*i for i in range(10))\n'})
        code='import json,fixture_helper\nprint(json.dumps({"metrics":{"sum_squares":fixture_helper.total()},"correctness_passed":True}))\n'
        self.engine.apply('write_file',{'path':'benchmark.py','content':code})
        command=f"& '{sys.executable}' benchmark.py"
        result=self.engine.apply('command_experiment',{'hypothesis_id':'h1','shell':'powershell','command':command})
        self.assertEqual(result['result']['metrics']['sum_squares'],285)
        self.engine.apply('write_file',{'path':'benchmark.py','content':'raise RuntimeError("mutated live file")'})
        repeat=self.engine.apply('verify',{'experiment_id':result['id']})
        self.assertEqual(repeat['result']['metrics']['sum_squares'],285)
        self.assertTrue(repeat['check']['passed'])
        self.assertIn('冻结工作区',repeat['protocol'])

    def test_unconfigured_strong_tier_uses_session_model_when_routing_disabled(self):
        self.engine.run['routing'] = {'enabled': False, 'profiles': {
            'light': {'model': '', 'model_source': 'local'},
            'strong': {'model': '', 'model_source': 'local'}}}
        self.engine.apply('design_workflow', {
            'reason': '未启用按调用选择模型时，未配置的 strong 档应被忽略',
            'stages': [{'role': 'analyst', 'goal': '检查现有材料并记录证据缺口',
                        'why': '确认关闭路由后不会因空的 strong 档失败', 'model_tier': 'strong'}]})
        self.engine.run['active_agent'] = {'model_tier': 'strong', 'model': self.engine.run['model'],
                                           'model_source': self.engine.run.get('model_source', 'local')}
        chosen = routing.choose(self.engine, 'agent_step', {})
        self.assertEqual(chosen['model'], self.engine.run['model'])
        self.assertEqual(chosen['tier'], 'default')

    def test_tier_can_be_selected_even_when_automatic_routing_disabled(self):
        self.engine.apply('select_model',{'model_tier':'strong','reason':'测试编排'})
        self.assertEqual(routing.choose(self.engine,'agent_step',{})['model'],'fixture-strong')
        self.engine.run['routing']['enabled']=True
        self.engine.apply('select_model',{'model_tier':'auto','reason':'测试不限预算'})
        self.assertIn(routing.choose(self.engine,'agent_step',{})['tier'],('light','strong'))
        with self.assertRaises(ValueError):routing.profile(self.engine,'nonexistent')

    def test_parallel_children_overlap_and_are_charged_once(self):
        barrier=threading.Barrier(2)
        seen=[]
        def factory(child,item):
            def decide():
                seen.append((item['id'],routing.choose(child,'agent_step',{})['model']))
                barrier.wait(5)
                call={'input_tokens':11,'output_tokens':7,'source':'fixture','node_id':item['id']+'-fixture'}
                child.budget.record(call);child.run['calls'].append(call)
                return {'tool':'inspect','args':{}}
            return decide
        self.engine.child_decision_factory=factory
        tasks=[{'goal':'并行测试任务 A，不调用模型','role':'generalist','model_tier':'light','max_steps':1},
               {'goal':'并行测试任务 B，不调用模型','role':'generalist','model_tier':'strong','max_steps':1}]
        result=self.engine.apply('delegate_research',{'tasks':tasks,'batch_key':'fixture'})
        self.assertEqual({model for _,model in seen},{'fixture-light','fixture-strong'})
        self.assertEqual(len(result['subagents']),2);self.assertEqual(self.engine.budget.used,36)
        self.assertEqual(len(self.engine.run['calls']),2)
        self.engine.apply('delegate_research',{'tasks':tasks,'batch_key':'fixture'})
        self.assertEqual(self.engine.budget.used,36)
        self.assertEqual(len([n for n in self.engine.run['trace'] if n['kind']=='subagent']),2)
        self.assertTrue(all((self.engine.folder/'subagents'/i/'run.json').exists() for i in ('sa1','sa2')))

    def test_delegation_refuses_insufficient_budget_before_launch(self):
        self.engine.budget.token_limit=10000
        with self.assertRaises(BudgetError):
            self.engine.apply('delegate_research',{'tasks':[{'goal':'测试不应启动的子任务','role':'analyst','model_tier':'light'}]})
        self.assertFalse(self.engine.run.get('subagents'))

    def test_plan_automatically_runs_a_parallel_group_then_main_reporter(self):
        barrier=threading.Barrier(2)
        def factory(child,item):
            def decide():
                barrier.wait(5)
                return {'tool':'inspect','args':{}}
            return decide
        self.engine.child_decision_factory=factory
        self.engine.run.update(workflow_mode='plan',workflow={'source':'fixture','stages':workflow.validate([
            {'role':'analyst','goal':'并行计划任务 A 的独立材料检查','why':'测试','parallel_group':'evidence','max_steps':1,'model_tier':'light'},
            {'role':'analyst','goal':'并行计划任务 B 的独立材料检查','why':'测试','parallel_group':'evidence','max_steps':1,'model_tier':'strong'},
            {'role':'reporter','goal':'主 Agent 如实汇总测试','why':'子任务结束后','max_steps':1}])})
        self.engine.decide=lambda:{'tool':'finish','args':{'conclusion':'测试没有真实研究证据，明确报告未完成。','citations':[],'lesson':''}}
        result=self.engine.execute()
        self.assertEqual(len(result['subagents']),2)
        self.assertTrue(any(a['role']=='reporter' and a['status']=='done' for a in result['agents']))
        self.assertEqual([s['tool'] for s in result['steps']],['delegate_research','finish'])

    def test_paused_children_resume_without_double_charging(self):
        entered=threading.Barrier(3)
        def factory(child,item):
            def decide():
                call={'input_tokens':8,'output_tokens':2,'source':'fixture'}
                child.budget.record(call);child.run['calls'].append(call)
                entered.wait(5)
                while not child.pause_requested.is_set():time.sleep(.01)
                child.guard_pause()
            return decide
        self.engine.child_decision_factory=factory
        tasks=[{'goal':'暂停恢复测试子任务 '+label,'role':'generalist','model_tier':'light','max_steps':2} for label in ('A','B')]
        thread,result=self.background(lambda:self.engine.apply('delegate_research',{'tasks':tasks,'batch_key':'pause'}))
        entered.wait(5);self.engine.pause_requested.set();thread.join(8)
        self.assertNotIn('error',result)
        self.assertEqual(self.engine.budget.used,20)
        self.engine.run['status']='paused';self.engine.save()
        resumed=core.resume_paused(self.engine.run,self.engine.folder)
        resumed.child_decision_factory=lambda child,item: lambda:{'tool':'inspect','args':{}}
        resumed.apply('delegate_research',{'tasks':tasks,'batch_key':'pause'})
        self.assertEqual(resumed.budget.used,20)
        self.assertEqual(len(resumed.run['calls']),2)
        self.assertEqual(len([n for n in resumed.run['trace'] if n['kind']=='subagent']),4)

    def test_workflow_validates_model_tiers_and_parallel_groups(self):
        stage={'role':'analyst','goal':'相互独立的证据分析任务','why':'测试并行与模型编排','model_tier':'light','parallel_group':'evidence'}
        cleaned=workflow.validate([stage,{**stage,'model_tier':'strong'},{'role':'reporter','goal':'汇总实际结果','why':'等待两个子任务'}])
        self.assertEqual(cleaned[0]['parallel_group'],'evidence');self.assertEqual(cleaned[1]['model_tier'],'strong')
        with self.assertRaises(ValueError):workflow.validate([{**stage,'role':'reporter'}])
        with self.assertRaises(ValueError):workflow.validate([stage,{'role':'generalist','goal':'中间的串行阶段','why':'隔断'},stage])

    def test_retry_restores_before_target_without_future_evidence_or_grants(self):
        self.engine.run['access_mode']='full'
        self.engine.apply('write_file',{'path':'version.txt','content':'before'})
        self.engine._perform_tool({'tool':'inspect','args':{}})
        failed=self.engine._perform_tool({'tool':'read','args':{'asset_id':'missing'}})
        self.engine.apply('write_file',{'path':'version.txt','content':'after'})
        self.engine.run['sources'].append({'id':'p1','title':'future evidence','evidence_scope':'网页正文'})
        self.engine.save()
        before=copy.deepcopy(self.engine.run)
        state,folder=history.retry(self.engine.run,self.engine.folder,failed['node_id'])
        self.assertEqual(state['sources'],[])
        self.assertEqual((folder/'workspace/version.txt').read_text(),'before')
        self.assertEqual(state['access_mode'],'approval');self.assertEqual(state['permission_grants'],[])
        self.assertEqual(state['lineage']['retry_of'],failed['node_id'])
        self.assertEqual(self.engine.run,before)

    def test_retry_first_node_uses_initial_inputs(self):
        node=history.begin(self.engine,'model_call',input={'fixture':True})
        history.settle(self.engine,node,status='failed',output={'error':'fixture failure'})
        self.engine.run['sources']=[{'id':'p1','title':'future'}]
        state,folder=history.retry(self.engine.run,self.engine.folder,node['id'])
        self.assertEqual(state['trace'],[]);self.assertEqual(state['sources'],[])
        self.assertEqual(state['lineage']['precision'],'initial_inputs')


if __name__=='__main__':unittest.main()
