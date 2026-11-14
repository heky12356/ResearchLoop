"""Workspace file and shell tools, decoupled from the NumPy experiment worker."""
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid
from types import SimpleNamespace
from researchloop import storage
from . import registry


class CommandError(ValueError):
    def __init__(self,result):
        self.result=result
        super().__init__(f"命令未成功：exit_code={result['exit_code']}，{result.get('termination') or '请检查实际输出'}")

ARGUMENTS = {
    'list_files': ({'path':{'type':'string'}}, []),
    'read_file': ({'path':{'type':'string'},'max_chars':{'type':'integer'}}, ['path']),
    'write_file': ({'path':{'type':'string'},'content':{'type':'string'}}, ['path','content']),
    'run_command': ({'shell':{'type':'string','enum':['powershell','cmd','bash']},'command':{'type':'string'},
                     'timeout_seconds':{'type':'integer'}}, ['shell','command']),
    'command_experiment': ({'hypothesis_id':{'type':'string'},'shell':{'type':'string','enum':['powershell','cmd','bash']},
                            'command':{'type':'string'},'timeout_seconds':{'type':'integer'}},['hypothesis_id','shell','command']),
}


def workspace(engine):
    root = (engine.folder/'workspace').resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def target(engine, value):
    root = workspace(engine)
    path = (root/str(value or '.')).resolve()
    if not path.is_relative_to(root):
        raise ValueError('文件路径必须位于本会话 workspace 内；不能通过绝对路径、.. 或符号链接越界')
    return path


def shells():
    return {name: (fallback if fallback and Path(fallback).is_file() else None) or shutil.which(exe)
            for name, exe, fallback in [('powershell','powershell',''),('cmd','cmd',''),
                                        ('bash','bash','C:/Program Files/Git/bin/bash.exe')]}


def list_files(engine, args):
    base = target(engine, args.get('path', '.'))
    if not base.is_dir():
        raise ValueError('目录不存在')
    rows=[]
    for path in base.iterdir():
        if not path.resolve().is_relative_to(workspace(engine)):
            continue
        rows.append({'path':path.relative_to(workspace(engine)).as_posix(), 'directory':path.is_dir(),
                     'bytes':path.stat().st_size if path.is_file() else None})
        if len(rows) >= 500:
            break
    return {'workspace':str(workspace(engine)), 'entries':rows, 'limit':500}


def read_file(engine, args):
    path = target(engine, args['path'])
    maximum = max(1,min(40000,int(args.get('max_chars',16000))))
    with path.open('rb') as handle:
        raw = handle.read(maximum*4+1)
    if b'\x00' in raw:
        raise ValueError('该工具只读文本文件；二进制文件请使用文档或实验工具')
    text = raw.decode('utf-8-sig', errors='replace')
    return {'path':args['path'], 'content':text[:maximum], 'truncated':len(text)>maximum or path.stat().st_size>len(raw)}


def write_file(engine, args):
    path = target(engine, args['path'])
    content = str(args['content'])
    if len(content.encode('utf-8')) > 1024*1024:
        raise ValueError('单次文本写入最多 1MB')
    def digest(file):
        if not file.exists():return None
        hasher=hashlib.sha256()
        with file.open('rb') as handle:
            for block in iter(lambda:handle.read(1024*1024),b''):hasher.update(block)
        return hasher.hexdigest()
    previous=digest(path)
    details = {'operation':'write_file','path':str(path),'content':content,
               'sha256':hashlib.sha256(content.encode()).hexdigest(), 'overwrites':path.exists(),'previous_sha256':previous}
    engine.permission_gate.require(engine,'workspace_write',details)
    engine.guard_pause()
    if engine.cancelled.is_set() or engine.seconds_left()<=0:
        raise ValueError('研究停止或预算耗尽，未写入文件')
    path = target(engine, args['path'])
    if digest(path)!=previous:
        raise ValueError('等待批准期间文件已改变，未写入。请重新读取后再决定修改。')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(content,encoding='utf-8')
    os.replace(temporary,path)
    return {'path':args['path'],'bytes':len(content.encode()),'sha256':details['sha256']}


def run_command(engine, args):
    shell = args['shell']
    executable = shells().get(shell)
    if not executable:
        raise ValueError(f'{shell} 在本机不可用，请选择其他 shell')
    command = str(args['command']).strip()
    if not command or len(command)>20000:
        raise ValueError('命令为空或超过 20000 字符')
    timeout = max(1,min(600,int(args.get('timeout_seconds',60)),engine.seconds_left()))
    root = workspace(engine)
    engine.permission_gate.require(engine,'command_'+shell,{'operation':'run_command','shell':shell,
                                   'command':command,'cwd':str(root),'timeout_seconds':timeout})
    engine.guard_pause()
    if engine.cancelled.is_set():
        raise ValueError('研究已停止，未启动命令')
    timeout=min(timeout,engine.seconds_left())
    if timeout<=0:
        from researchloop.ollama import BudgetError
        raise BudgetError('批准后预算已耗尽，未启动命令')
    flags = {'powershell':['-NoProfile','-NonInteractive','-Command'], 'cmd':['/d','/s','/c'], 'bash':['-c']}[shell]
    log = engine.folder/'command-output'/(uuid.uuid4().hex+'.log')
    log.parent.mkdir(exist_ok=True)
    allowed = ('PATH','PATHEXT','SystemRoot','WINDIR','COMSPEC','TEMP','TMP','USERPROFILE','APPDATA','LOCALAPPDATA','PROGRAMFILES','PROGRAMFILES(X86)')
    env = {key:value for key,value in os.environ.items() if key.upper() in {s.upper() for s in allowed}}
    env.update(PYTHONIOENCODING='utf-8', PYTHONUTF8='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               PIP_CACHE_DIR=str(root/'.cache/pip'), XDG_CACHE_HOME=str(root/'.cache'),
               PIP_TARGET=str(root/'.packages'),PYTHONPATH=str(root/'.packages'),
               UV_CACHE_DIR=str(root/'.cache/uv'),npm_config_cache=str(root/'.cache/npm'),
               npm_config_prefix=str(root/'.packages/npm'),HF_HOME=str(root/'.cache/huggingface'),
               TMP=str(root/'.tmp'),TEMP=str(root/'.tmp'))
    env['PYTHONPATH']=os.pathsep.join([str(root/'.packages'),*map(str,getattr(engine,'dependency_paths',[]))])
    if getattr(engine,'execution_seed',None) is not None:
        env['RESEARCHLOOP_SEED']=str(engine.execution_seed)
    (root/'.tmp').mkdir(exist_ok=True)
    started = time.monotonic()
    with log.open('wb') as output:
        child = subprocess.Popen([executable,*flags,command],cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,
                                 start_new_session=os.name!='nt')
        reason = None
        while child.poll() is None:
            if engine.cancelled.is_set() or time.monotonic()-started >= timeout or engine.seconds_left()<=0:
                reason = 'cancelled' if engine.cancelled.is_set() else 'timeout'
                if os.name=='nt':
                    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
                else:
                    import signal
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
                break
            time.sleep(.05)
    with log.open('rb') as handle:
        raw = handle.read(64000)
    result = {'shell':shell,'command':command,'cwd':str(root),'exit_code':child.returncode,'termination':reason,
            'output':raw.decode('utf-8-sig',errors='replace'),'truncated':log.stat().st_size>64000,
            'output_file':str(log.relative_to(engine.folder)),'elapsed_seconds':round(time.monotonic()-started,3),
            'note':'命令输出是执行证据；工程结论仍需设计对照、检查正确性和复核。'}
    if child.returncode!=0 or reason:
        raise CommandError(result)
    return result


def _project_manifest(root):
    rows={}
    total=0
    excluded={'.git','.venv','node_modules','.cache','.tmp','.packages','__pycache__'}
    for directory,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=[name for name in dirs if name not in excluded and not (Path(directory)/name).is_symlink()]
        for name in files:
            path=Path(directory)/name
            if path.is_symlink():
                raise ValueError('冻结工程实验不接受符号链接，请使用工作区内真实文件')
            size=path.stat().st_size
            total+=size
            if size>20*1024*1024 or total>100*1024*1024:
                raise ValueError('冻结实验源文件超过单文件20MB/合计100MB，请将大型数据缩减为可复现样本')
            rows[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return rows


def _copy_project(source,target):
    manifest=_project_manifest(source)
    target.mkdir(parents=True,exist_ok=True)
    for relative in manifest:
        dest=target/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/relative,dest)
    return manifest


def _command_result(engine,item,phase,seed):
    base=engine.folder/'experiments'/item['id']
    if _project_manifest(base/'project')!=item['project_manifest']:
        raise ValueError('冻结实验文件已改变，拒绝复核')
    location=base/phase
    _copy_project(base/'project',location/'workspace')
    dependencies=[engine.folder/'workspace/.packages',*getattr(engine,'dependency_paths',[])]
    origin=(item.get('subagent_origin') or {}).get('id')
    if origin and origin.startswith('sa') and origin[2:].isdigit():
        study=engine.folder.parent.parent if engine.run.get('is_subagent') else engine.folder
        dependencies.append(study/'subagents'/origin/'workspace/.packages')
    context=SimpleNamespace(folder=location,run=engine.run,permission_gate=engine.permission_gate,
                            guard_pause=engine.guard_pause,cancelled=engine.cancelled,seconds_left=engine.seconds_left,
                            pause_requested=engine.pause_requested,
                            execution_seed=seed,dependency_paths=dependencies)
    output=run_command(context,item['command_spec'])
    from researchloop import llmjson
    result=llmjson.parse(output['output'])
    metrics=result.get('metrics') if isinstance(result,dict) else None
    if not isinstance(metrics,dict) or not metrics or any(
            not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) for value in metrics.values()):
        raise ValueError('工程实验必须实际输出 JSON metrics，所有指标须为有限数值')
    if not isinstance(result.get('correctness_passed'),bool):
        raise ValueError('工程实验须输出 correctness_passed=true/false，说明正确性检查是否通过')
    payload={'metrics':metrics,'correctness_passed':result['correctness_passed'],'notes':str(result.get('notes') or '')[:2000],
             'command_result':output}
    storage.write_json(location/'result.json',payload)
    return payload


def command_experiment(engine,args):
    from . import tools
    hypothesis=next((h for h in engine.run['hypotheses'] if h['id']==args['hypothesis_id']),None)
    if hypothesis is None:
        raise ValueError('先注册可证伪假设，再进行工程实验')
    identifier='x'+str(len(engine.run['experiments'])+1)
    base=engine.folder/'experiments'/identifier
    manifest=_copy_project(workspace(engine),base/'project')
    item={'id':identifier,'kind':'command','hypothesis_id':hypothesis['id'],'status':'running',
          'code':f"[{args['shell']}] {args['command']}",'project_manifest':manifest,
          'command_spec':{key:args[key] for key in ('shell','command','timeout_seconds') if key in args}}
    engine.run['experiments'].append(item)
    try:
        item['result']=_command_result(engine,item,'explore',engine.run['seed'])
        item['check']=tools.check(hypothesis,item['result'])
        if not item['result']['correctness_passed']:
            item['check']['passed']=False
        item['status']='completed'
    except Exception as error:
        item.update(status='failed',error=str(error))
        raise
    return {'id':identifier,'result':item['result'],'check':item['check']}


def verify_command(engine,item,seed):
    return _command_result(engine,item,'verify',seed)


for name, handler, description, sample in [
    ('list_files',list_files,'列出本会话工作目录中的文件与子目录。',{'path':'.'}),
    ('read_file',read_file,'读取本会话工作目录内的文本文件。',{'path':'experiment.py'}),
    ('write_file',write_file,'写入本会话工作目录；请求批准模式须等待用户批准，不能绕过或伪造授权。',{'path':'experiment.py','content':'print(1+1)'}),
    ('run_command',run_command,'使用可用的 Bash/CMD/PowerShell 执行命令。请求批准模式须等待用户批准；返回真实输出、退出码和超时情况。',{'shell':'powershell','command':'python experiment.py','timeout_seconds':60}),
    ('command_experiment',command_experiment,'为已注册假设冻结工作区代码，实际运行工程实验；命令须输出 JSON {metrics:{指标:实测值},correctness_passed:true/false}。随后用 verify 复核同一代码。',{'hypothesis_id':'h1','shell':'powershell','command':'python benchmark.py','timeout_seconds':60})]:
    registry.register_tool(name,handler,description=description,arguments=sample,required=ARGUMENTS[name][1])
