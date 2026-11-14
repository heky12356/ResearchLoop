let researchSubmitting=false;
const sessionActionIds=new Set();
/* Open research questions, evidence and invocation history. */
statuses.incomplete='证据不足 / 未完成';
const previousTop=renderTop, previousMetrics=renderMetrics, previousRun=renderRun;

renderTop=function(){
  previousTop();
  $('#dataset-summary').textContent=`${state.research_tools?.length||0} 项研究工具 · 当前会话 ${run?.assets?.length||0} 份材料`;
  if(page==='data')$('#page-label').textContent='材料与工具';
  $('#draft-brief').disabled=researchSubmitting;
  const modelSelect=$('#research-model');
  const cloud=state.settings?.cloud||{};
  if(modelStatus?.models?.length||(cloud.enabled&&cloud.model)){
    const names=(modelStatus?.models||[]).filter(m=>!m.name.toLowerCase().includes('cloud')).map(m=>m.name);
    const options=[...names.map(name=>`<option value="${esc(name)}">${esc(name)} · 本地</option>`),(cloud.enabled&&cloud.model)?`<option value="cloud:${esc(cloud.model)}">${esc(cloud.model)} · 云端</option>`:''].join('');
    if(modelSelect.innerHTML!==options){const selected=modelSelect.value;modelSelect.innerHTML=options;const fallback=cloud.enabled&&cloud.model?`cloud:${cloud.model}`:(names.includes('deepseek-r1:14b')?'deepseek-r1:14b':state.settings.model);modelSelect.value=modelSelect.dataset.ready&&options.includes(`value="${selected}"`)?selected:(options.includes(`value="${fallback}"`)?fallback:(names[0]||''));modelSelect.dataset.ready='1';}
  }
};
renderMetrics=function(){
  if(run&&run.domain!=='research'){previousMetrics();return;}
  const experiments=run?.experiments||[], verified=experiments.filter(e=>e.verification);
  $('#metrics').innerHTML=metric('已复核实验',String(verified.length),'',`${verified.filter(e=>e.verification.check.passed).length} 项满足判据，否定结果同样保留`,'target')+
    metric('本轮模型 token',run?.budget?number(run.budget.total_tokens):'0','','规划、修正与报告决策均计入','coin')+
    metric('真实实验',String(experiments.length),'',`${run?.steps?.length||0} 步工具行动 · 不预设实验顺序`,'flask')+
    metric('记忆',`${(run?.memory_context?.facts?.length||0)+(run?.experience?1:0)} 条`,'',run?.memory_saved_version?`已保存方法 v${run.memory_saved_version}`:run?.memory_version?`参考方法 v${run.memory_version}`:'按当前研究问题自动召回','spark');
};
renderRun=function(){
  if(run&&run.domain!=='research'){previousRun();return;}
  if(!run){
    $('#session-title').textContent='研究对话';
    $('#session-question').textContent='从工作台开始一个新的研究，或从左侧研究记录里打开已有会话。';
    $('#session-status').textContent='未选择';
  }else{
    $('#session-title').textContent=(run.question||run.task||'研究对话').slice(0,60);
    $('#session-question').textContent=run.question||'';
    $('#session-status').textContent=statuses[run.status]||run.status;
  }
  $('#diagnostics-panel').classList.add('hidden');
  $('#evidence-experiment').innerHTML='';$('#evidence-claim').innerHTML='';
  if(!run){
    $('#run-meta').innerHTML='';$('#experiment-counter').textContent='等待研究问题';
    $('#chart').innerHTML='<div class="research-guide"><span>研究问题</span> → <span>证据与假设</span> → <span>实验与分析</span> → <span>复核与修正</span></div>';
    $('#experiments').innerHTML=empty('从一个可检验的问题开始','Agent会根据工具的实际返回决定下一步，报告包含执行代码、测量结果、成本与局限。');
    $('#hypothesis').innerHTML='<div class="hypothesis-empty">等待问题与材料。<br>假设由Agent提出，判据在实验之前登记。</div>';
    $('#evidence').innerHTML=empty('真实证据会汇集到这里','文献来源、上传材料与实验ID均可追溯。没有验证时不会显示完成。');return;
  }
  const active=['running','queued'].includes(run.status);
  $('#experiment-counter').textContent=`${run.steps.length} 步行动 · ${run.experiments.length} 项实验`;
  $('#run-meta').innerHTML=`<span class="status-dot ${active?'running':''}"></span><span>${esc(statuses[run.status]||run.status)}</span><span>· 科研 Agent</span><span class="spacer"></span><button id="research-raw">完整记录</button>${active?'<button id="research-cancel">停止</button>':`<a href="/api/runs/${run.id}/report">下载报告</a>`}<a href="/api/runs/${run.id}/export">复现包 ↗</a>${run.status==='completed'&&run.report?.lesson?'<button id="research-publish">发布方法经验</button>':''}`;
  $('#research-raw').onclick=()=>showDetail('研究问题、决策和原始工具结果',run);
  $('#research-cancel')?.addEventListener('click',async()=>{await api(`/api/runs/${run.id}/cancel`,{});toast('当前调用结束后停止，已有证据保留。');});
  $('#research-publish')?.addEventListener('click',async()=>{try{const value=await api('/api/memory/promote',{run_id:run.id});toast(`已发布科研经验 v${value.version}，迁移收益仍待验证。`);await refresh(true);}catch(e){toast(e.message);}});
  const h=run.hypotheses.at(-1);
  $('#hypothesis').innerHTML=`<span class="hypothesis-tag">${h?esc(h.id):'收集证据中'}</span><div class="hypothesis-text">${esc(h?.statement||run.question)}</div><div class="hypothesis-source">${h?`预测：${esc(h.prediction)}<br>控制：${esc(h.controls)}<br>冻结判据：${esc(h.metric)} ${esc(h.relation)} ${esc(h.threshold)}`:'模型将先检查条件、材料和可用工具，不预设结论。'}</div>`;
  $('#chart').innerHTML=`<div class="research-guide">${esc(run.question)}</div>`;
  $('#experiments').innerHTML=run.steps.length?run.steps.map(s=>`<details class="research-step"><summary><span class="pill subtle">${esc(s.id)} · ${esc(s.tool)}</span><strong>${esc(s.reason)}</strong><span class="${s.status==='failed'?'negative':''}">${s.status==='failed'?'失败，等待修正':'已执行'}</span></summary><pre>${esc(JSON.stringify({arguments:s.args,observation:s.observation},null,2))}</pre></details>`).join(''):empty('正在规划第一步','模型正在结合问题、预算与工具拟定研究行动。');
  renderResearchEvidence(run);
};
function renderResearchEvidence(value){
  const box=$('#evidence'),stamp=JSON.stringify([value.id,value.experiments,value.sources,value.assets]);
  if(box._evidenceStamp===stamp)return;box._evidenceStamp=stamp;
  const opened=new Set($$('#evidence details[open]').map(el=>el.dataset.evidenceId));
  const experiments=(value.experiments||[]).map(e=>`<article class="evidence-card"><div class="evidence-card-heading"><span class="source-id">${esc(e.id)}</span><h3>实验 · ${esc(e.hypothesis_id)}</h3><span class="evidence-kind">${e.verification?'已复核':'未复核'}</span></div><p>${e.result?`探索结果：${esc(JSON.stringify(e.result.metrics))}`:esc(e.error||'执行中')}</p>${e.verification?`<p>复核结果：${esc(JSON.stringify(e.verification.result.metrics))}</p><p>${e.verification.check.passed?'满足预注册判据':'未满足预注册判据'}</p>`:''}<details data-evidence-id="${esc(e.id)}" ${opened.has(e.id)?'open':''}><summary>查看实验代码</summary><pre>${esc(e.code||'未记录代码')}</pre></details></article>`);
  const sources=(value.sources||[]).map(p=>{
    const text=String(p.abstract||p.text||p.content||''),scope=p.evidence_scope||'证据范围未标注';let url='',host='';
    try{const parsed=new URL(p.url);if(['https:','http:'].includes(parsed.protocol)){url=parsed.href;host=parsed.hostname.replace(/^www\./,'');}}catch(_){}
    return `<article class="evidence-card"><div class="evidence-card-heading"><span class="source-id">${esc(p.id)}</span><span class="evidence-kind" title="${esc(scope)}">${esc(scope)}</span></div><h3>${esc(p.title||'未命名来源')}</h3><p class="source-excerpt">${esc(text?text.slice(0,260)+(text.length>260?'…':''):'未获取摘要或正文，当前只有元数据。')}</p>${text.length>260?`<details data-evidence-id="${esc(p.id)}" ${opened.has(p.id)?'open':''}><summary>展开完整摘录</summary><p>${esc(text)}</p></details>`:''}<div class="source-footer"><span>${esc(host||'材料来源')}</span>${url?`<a href="${esc(url)}" target="_blank" rel="noreferrer">打开来源 ↗</a>`:''}</div></article>`;
  });
  const assets=(value.assets||[]).map(a=>`<article class="evidence-card"><div class="evidence-card-heading"><span class="source-id">${esc(a.id)}</span><span class="evidence-kind">上传材料</span></div><h3>${esc(a.name)}</h3><p>${number(a.bytes)} 字节</p><details><summary>文件校验值</summary><pre>${esc(a.sha256||'未记录')}</pre></details></article>`);
  const cards=[...experiments,...sources,...assets];box.innerHTML=cards.length?`<div class="research-evidence">${cards.join('')}</div>`:empty('尚无可展示的证据','可在执行记录中查看取证过程与失败原因。');
}
renderData=function(){$('#research-tool-list').innerHTML=(state.research_tools||[]).map(t=>`<article class="panel memory-card"><h3>${esc(t.name)}</h3><p>${esc(t.description)}</p></article>`).join('');};
let pendingFiles=[];
const TEXT_SUFFIX=/\.(csv|txt|md|json|py|js|ts|jsx|tsx|c|cpp|h|rs|go|java|toml|yaml|yml|sh|ps1|bat|sql|html|css)$/i, DOC_SUFFIX=/\.(pdf|docx|pptx)$/i;
const MAX_TEXT_BYTES=500000, MAX_DOC_BYTES=20*1024*1024;
function base64Bytes(buffer){const bytes=new Uint8Array(buffer);let binary='';const chunk=0x8000;for(let i=0;i<bytes.length;i+=chunk){binary+=String.fromCharCode.apply(null,bytes.subarray(i,i+chunk));}return btoa(binary);}
function renderAssetChips(){
  $('#research-asset-list').innerHTML=pendingFiles.map((file,index)=>`<span class="asset-chip"><b>${esc(file.name)}</b><small>${Math.ceil(file.size/1024)}KB</small><button type="button" data-remove-file="${index}" aria-label="移除 ${esc(file.name)}">×</button></span>`).join('');
  $$('[data-remove-file]').forEach(button=>button.onclick=()=>{pendingFiles.splice(Number(button.dataset.removeFile),1);renderAssetChips();});
}
async function assetPayload(file){
  if(TEXT_SUFFIX.test(file.name)){if(file.size>MAX_TEXT_BYTES)throw new Error(`${file.name} 超过 500KB`);return {name:file.name,content:await file.text()};}
  if(DOC_SUFFIX.test(file.name)){if(file.size>MAX_DOC_BYTES)throw new Error(`${file.name} 超过 20MB`);return {name:file.name,content_base64:base64Bytes(await file.arrayBuffer())};}
  throw new Error(`不支持的材料类型：${file.name}`);
}
async function startGeneralResearch(){
  const question=$('#brief-input').value.trim();if(question.length<8){toast('请先描述一个具体研究问题。');return;}
  if(researchSubmitting)return;researchSubmitting=true;const button=$('#draft-brief');button.disabled=true;
  try{
    if(pendingFiles.length>8)throw new Error('最多8份材料');
    const assets=[];for(const file of pendingFiles)assets.push(await assetPayload(file));
    const pick=$('#research-model').value;
    const isCloud=pick.startsWith('cloud:');
    const budgetOn=$('#budget-enabled').checked;
    const tokensK=Number($('#research-tokens-k').value||0);
    const secondsRaw=$('#research-seconds').value.trim();
    const stepsRaw=$('#research-steps').value.trim();
    if(budgetOn&&(!tokensK||tokensK<=0))throw new Error('启用预算时请填写 token 预算（k）');
    const created=await api('/api/research',{question,assets,model:isCloud?pick.slice(6):pick,model_source:isCloud?'cloud':'local',workflow_mode:$('#research-workflow').value,access_mode:$('#research-access').value,roles:state.settings?.roles||{},budget_enabled:budgetOn,token_limit:budgetOn?Math.round(tokensK*1000):null,seconds_limit:budgetOn&&secondsRaw?Number(secondsRaw):null,max_steps:budgetOn&&stepsRaw?Number(stepsRaw):null});
    pendingFiles=[];renderAssetChips();
    await selectRun(created.id);toast('研究已启动；材料已随本次运行保存。');
  }catch(error){toast(error.message);}finally{researchSubmitting=false;button.disabled=false;}
}
$('#attach-button').addEventListener('click',()=>$('#research-files').click());
$('#research-files').addEventListener('change',()=>{
  for(const file of $('#research-files').files){
    if(!TEXT_SUFFIX.test(file.name)&&!DOC_SUFFIX.test(file.name)){toast(`不支持的材料类型：${file.name}`);continue;}
    if(pendingFiles.some(item=>item.name===file.name&&item.size===file.size))continue;
    if(pendingFiles.length>=8){toast('最多8份材料');break;}
    pendingFiles.push(file);
  }
  $('#research-files').value='';renderAssetChips();
});

/* Random topic generator: the three fixed examples are gone; every click composes
   a testable simulation question that fits the current tool surface. */
const TOPIC_PARTS={
  units:['总体均值','中位数','分位数','回归系数','相关系数','分类准确率'],
  settings:['离群值比例 0.02 至 0.10','样本量 30 与 300','缺失值比例 5% 至 20%','异方差噪声','非正态重尾分布','重复测量误差'],
  methods:['均值填补与中位数填补','普通最小二乘与稳健回归','简单随机抽样与分层抽样','固定阈值与自适应阈值'],
  metrics:['平均绝对误差','均方误差','置信区间覆盖率','排序稳定性'],
  templates:[
    '在模拟情景中比较{method}在{setting}下的{metric}差异。先注册可证伪假设与数值判据，重复模拟至少200次，实际运行并用新的随机种子独立复核，说明结论的适用条件。',
    '研究{setting}如何影响{unit}的{metric}。没有真实数据时明确采用模拟，控制生成过程，设计对照实验并由固定代码复核。',
    '比较两种估计{unit}的方法（{method}）在{setting}下的{metric}，给出预注册判据并独立重跑复核，不允许用同一批随机数当作验证。'
  ]
};
function randomTopic(){
  const pick=list=>list[Math.floor(Math.random()*list.length)];
  const template=pick(TOPIC_PARTS.templates);
  return template.replace('{method}',pick(TOPIC_PARTS.methods)).replace('{setting}',pick(TOPIC_PARTS.settings)).replace('{unit}',pick(TOPIC_PARTS.units)).replace('{metric}',pick(TOPIC_PARTS.metrics));
}
$('#random-topic').addEventListener('click',()=>{const input=$('#brief-input');input.value=randomTopic();input.dispatchEvent(new Event('input'));input.focus();});

/* --- workflow orchestration, decision chain and session metrics --- */
statuses.done='阶段完成';
const ROLE_LABELS={planner:'规划',analyst:'分析',generalist:'通用',experimenter:'实验',verifier:'复核',reporter:'报告'};
function roleLabel(role){return ROLE_LABELS[role]||role||'—';}
let sessionTab='overview', sessionViewId='', chainView={runId:'',turn:'all',status:'all',query:''};
const TOOL_LABELS={get_memory_context:'读取研究经验',inspect:'检查材料',design_workflow:'设计研究流程',plan:'制定研究方案',literature:'检索文献',platform_search:'检索平台',web_search:'搜索网页',web_fetch:'读取网页',extract_document:'提取文档',read:'读取材料',hypothesis:'注册假设',experiment:'运行实验',verify:'复核实验',reflect:'评估结果',finish:'形成结论',invalid:'决策失败',remember_fact:'保存事实',recall_facts:'查找事实'};
function setSessionTab(name){
  sessionTab=['overview','execution','evidence'].includes(name)?name:'overview';
  $$('[data-session-tab]').forEach(button=>{const selected=button.dataset.sessionTab===sessionTab;button.setAttribute('aria-selected',String(selected));button.tabIndex=selected?0:-1;});
  $$('#page-session .session-view').forEach(panel=>panel.classList.toggle('hidden',panel.id!==`session-view-${sessionTab}`));
}
function renderSessionLayout(value){
  if(!value)return;
  if(sessionViewId!==value.id){sessionViewId=value.id;setSessionTab(value.domain==='research'?'overview':'execution');$('.session-raw-records').open=value.domain!=='research';}
  $('#session-title').textContent=value.question||value.task||'研究会话';
  const created=new Date(value.created_at);
  $('#session-question').textContent=[Number.isNaN(created.getTime())?'':created.toLocaleString('zh-CN',{month:'long',day:'numeric',hour:'2-digit',minute:'2-digit',hour12:false}),value.turns?.length?`${value.turns.length} 轮对话`:'历史运行',value.id].filter(Boolean).join(' · ');
  $('#session-step-count').textContent=value.steps?.length||0;
  $('#session-evidence-count').textContent=(value.sources?.length||0)+(value.assets?.length||0)+(value.experiments?.length||0);
  $('#session-report').href=`/api/runs/${value.id}/report?format=${$('#session-report-format').value}`;$('#session-export').href=`/api/runs/${value.id}/export`;
  const pending=['running','queued','pausing'].includes(value.status);$('#report-export').classList.toggle('hidden',pending||value.domain!=='research'||!value.report);$('#session-archive').disabled=pending;
  const isResearch=value.domain==='research',paused=value.status==='paused';
  $('#session-pause').classList.toggle('hidden',!isResearch||!pending);$('#session-pause').disabled=value.status==='pausing';$('#session-pause').textContent=value.status==='pausing'?'正在暂停…':value.status==='queued'?'暂停排队':'暂停研究';
  $('#session-resume').classList.toggle('hidden',!isResearch||!paused);$('#session-resume').disabled=sessionActionIds.has(value.id);
  $('#followup-send').disabled=pending||paused||sessionActionIds.has(value.id);
  const note=$('#session-control-note');note.classList.toggle('hidden',!paused&&value.status!=='pausing'&&value.status!=='queued'&&!value.lineage);
  note.textContent=value.status==='queued'?'排队等待运行名额，等待期间不计研究时间预算。':value.status==='pausing'?'正在等待当前调用结束。完成后保存结果与实际用量，再暂停。':paused?'研究已暂停。继续时沿用当前轮次和剩余预算，暂停期间不计研究用时。':value.lineage?`从 ${value.lineage.run_id} 的 ${value.lineage.node_id} 创建的分支${value.lineage.precision==='legacy_context'?' · 历史记录上下文，重新规划':' · 已恢复该调用结束时的状态'}`:'';
  $('#session-step-count').textContent=value.trace?.length||(value.steps?.length||0)+(value.calls?.length||0);
  if(isResearch)$('#experiments').innerHTML='';
}
function renderWorkflow(run){
  const panel=$('#workflow-panel');
  if(!run||run.domain!=='research'||run.workflow_mode!=='plan'){panel.classList.add('hidden');panel.innerHTML='';return;}
  panel.classList.remove('hidden');
  const stages=run.workflow?.stages||[],turn=run.turns?.at(-1)?.id,agents=(run.agents||[]).filter(a=>!a.turn||!turn||a.turn===turn),calls=(run.calls||[]).filter(c=>!c.turn||!turn||c.turn===turn);
  const byAgent={};calls.forEach(c=>{if(c.agent)byAgent[c.agent]=(byAgent[c.agent]||0)+1;});
  panel.innerHTML=`<div class="panel-header"><h2>本轮研究流程</h2><span class="workflow-source">${esc(run.workflow?.source==='agent_tool'?'Agent 设计':run.workflow?.source==='model'?'模型规划':run.workflow?.source==='model_retry'?'模型重试':'默认规划')} · ${stages.length} 阶段</span></div>${run.workflow?.reason?`<p class="workflow-reason">${esc(run.workflow.reason)}</p>`:''}<div class="workflow-stages">${stages.map((s,i)=>{let a=agents.filter(x=>x.id==='g'+(i+1)).at(-1)||{};if(s.parallel_group){const child=(run.subagents||[]).find(c=>c.goal===s.goal&&c.turn===turn);if(child){const live=(run.subagent_live||[]).find(c=>c.subagent_id===child.id)||{};a={...child,status:live.status||child.status,model:(run.routing?.profiles||{})[s.model_tier]?.model||run.model};}}return `<article class="workflow-stage${a.status==='running'?' active':''}"><div class="workflow-stage-heading"><span class="workflow-index">${String(i+1).padStart(2,'0')}</span><h3>${esc(s.label||roleLabel(s.role))}</h3><span class="workflow-stage-status">${esc(statuses[a.status]||'未开始')}</span></div><p>${esc(s.goal)}</p>${s.why?`<details class="stage-why"><summary>设计依据</summary><p>${esc(s.why)}</p></details>`:''}<small>${s.parallel_group?'并行 · '+esc(s.parallel_group)+' · ':''}${esc(({light:'轻量',strong:'高能力',auto:'默认'}[s.model_tier]||'默认'))} · ${esc(a.model||'待执行')}</small></article>`;}).join('')}</div>`;
}
function invocationRecords(value){
  if(value.trace?.length)return value.trace;
  // Legacy model requests cannot be placed between tool calls reliably.
  return [...(value.steps||[]).map(s=>({...s,kind:s.tool==='invalid'?'decision_error':'tool_call',input:s.args,output:s.observation,legacy:true})),
    ...(value.calls||[]).map((c,i)=>({...c,id:`call:${i}`,kind:'model_call',status:c.status==='error'?'failed':'completed',input:c.request,output:c.response,usage:c,elapsed_seconds:c.wall_seconds,legacy:true}))];
}
function renderChain(run){
  const panel=$('#chain-panel');
  if(!run||run.domain!=='research'){panel.classList.add('hidden');return;}
  panel.classList.remove('hidden');
  const records=invocationRecords(run),box=$('#chain');
  if(chainView.runId!==run.id){chainView={runId:run.id,turn:'all',status:'all',kind:'all',query:''};$('#chain-status').value='all';$('#chain-kind').value='all';$('#chain-search').value='';box._chainStamp='';}
  const turnNames=[...new Set([...(run.turns||[]).map(t=>t.id),...records.map(s=>s.turn).filter(Boolean)])];
  const options='<option value="all">全部轮次</option>'+turnNames.map(t=>`<option value="${esc(t)}">第 ${esc(t.replace(/^t/,''))} 轮</option>`).join('');
  if($('#chain-turn').innerHTML!==options){$('#chain-turn').innerHTML=options;$('#chain-turn').value=chainView.turn;}
  const scoped=records.filter(s=>chainView.turn==='all'||s.turn===chainView.turn);
  const filtered=scoped.filter(s=>(chainView.status==='all'||s.status===chainView.status)&&(!chainView.kind||chainView.kind==='all'||s.kind===chainView.kind)&&(!chainView.query||JSON.stringify(s).toLowerCase().includes(chainView.query.toLowerCase())));
  const children=(run.subagents||[]).filter(c=>chainView.turn==='all'||c.turn===chainView.turn);
  const liveCalls=children.reduce((sum,child)=>sum+Math.max(0,((run.subagent_live||[]).find(c=>c.subagent_id===child.id)?.budget?.model_calls||0)-(child.accounted_calls||0)),0);
  const modelCalls=(run.calls||[]).filter(c=>chainView.turn==='all'||c.turn===chainView.turn).length+liveCalls;
  $('#chain-meta').textContent=children.length?`${scoped.length} 条主链记录 · ${modelCalls} 次模型请求 · ${children.length} 个子 Agent`:`${scoped.length} 条调用 · ${modelCalls} 次模型请求 · ${scoped.filter(s=>s.kind==='tool_call').length} 次工具执行`;
  $('#chain-usage-note').textContent=run.trace?.length?'每次请求独立计量；工具执行单独标记。展开记录可查看输入输出或从此处创建分支。':'历史记录未保存调用顺序与状态快照；模型请求单独列出，分支会基于所选记录重新规划。';
  const stamp=JSON.stringify([run.id,chainView,filtered,run.subagent_live]);if(box._chainStamp===stamp)return;
  const open=new Set($$('details.chain-step[open]').map(el=>el.dataset.stepKey));box._chainStamp=stamp;
  box.innerHTML=filtered.length?filtered.map((s,index)=>{
    if(s.kind==='subagent'&&typeof renderSubagentNode==='function')return renderSubagentNode(run,s,index,open);
    const key=`${s.turn||'legacy'}:${s.id}`,failed=s.status==='failed',model=s.kind==='model_call',seconds=s.elapsed_seconds;
    const title=model?(s.model||'模型请求'):(TOOL_LABELS[s.tool]||s.tool||'决策校验');
    const source=s.source==='cloud'?'云端':s.source==='controller'?'控制器':'本地';
    const usage=s.usage||{},tokens=(usage.input_tokens||0)+(usage.output_tokens||0)+(usage.unknown_reserved||0);
    const duration=`${typeof seconds==='number'?seconds.toFixed(2)+'s':'等待结果'}${model?' · '+number(tokens)+' token':''}`;
    const error=s.output?.error||'',recovery=s.output?.recovery;
    const forkable=s.status!=='running'&&(s.checkpoint||s.legacy);
    return `<details class="chain-step ${esc(s.status||'')} ${model?'chain-model':'chain-tool-call'}" data-step-key="${esc(key)}" ${open.has(key)?'open':''}><summary class="chain-summary"><span class="chain-index">${index+1}</span><span class="chain-content"><span class="chain-heading"><strong class="chain-tool">${esc(title)}</strong><span class="invocation-kind ${model?'model':''}">${model?'模型请求':s.kind==='decision_error'?'决策错误':'工具'}</span><span class="chain-status ${failed?'failed':''}">${esc(statuses[s.status]||s.status)}</span><span class="chain-agent">${esc(source)}${s.role?' · '+esc(roleLabel(s.role)):''}</span></span><span class="chain-reason">${esc(s.route?.reason||s.reason||s.purpose||'')}${s.attempt>1?' · 第 '+s.attempt+' 次尝试':''}</span>${recovery?`<span class="call-recovery">${esc(recovery.action==='retry'?'已安排有限重试':recovery.action==='switch_source'?'已停用当前服务配置，选择替代来源':recovery.action==='defer'?'剩余预算不允许立即重试':'等待修正决策')} · ${esc(recovery.message)}</span>`:''}${error?`<span class="chain-error">${esc(String(error).slice(0,220))}</span>`:''}</span><span class="chain-duration">${esc(duration)}</span><span class="chain-chevron" aria-hidden="true">⌄</span></summary><div class="chain-body">${s.route?`<p class="call-routing">模型路由：${esc(s.route.tier==='fallback'?'回退模型':s.route.tier==='strong'?'高能力档':s.route.tier==='light'?'轻量档':'默认模型')} · ${esc(s.route.reason)}</p>`:''}<div class="chain-io"><section><h3>输入</h3><pre>${esc(JSON.stringify(s.input??{},null,2))}</pre></section><section><h3>输出</h3><pre>${esc(JSON.stringify(s.output??{},null,2))}</pre></section></div><div class="chain-detail-meta"><span>${esc(s.id)}${s.turn?' · '+esc(s.turn):''}${s.parent_id?' · 由 '+esc(s.parent_id)+' 决策':''}${s.inherited_from?' · 继承自原会话':''}</span><div><button type="button" class="small-link" data-chain-raw="${esc(s.id)}">原始记录</button>${forkable?`<button type="button" class="secondary-button fork-record" data-fork-node="${esc(s.id)}">从此处分支</button>`:''}</div></div></div></details>`;
  }).join(''):empty('暂无匹配的调用','开始研究或调整筛选条件。');
  $$('[data-chain-raw]').forEach(button=>button.onclick=()=>showDetail('调用记录',records.find(s=>s.id===button.dataset.chainRaw)));
  $$('[data-fork-node]').forEach(button=>button.onclick=()=>openFork(records.find(s=>s.id===button.dataset.forkNode)));
  if(typeof bindAgentChain==='function')bindAgentChain(run,records);
}
function renderBudgetResearch(){
  if(!run||run.domain!=='research')return;
  const b=run.budget,box=$('#budget');
  if(!b){if(!box.querySelector('.budget-pending'))box.innerHTML='<div class="budget-item"><div class="budget-label">等待首个模型调用</div></div>';return;}
  if(!box.querySelector('#budget-calls')){
    box.innerHTML=`<div class="budget-item"><div class="budget-label">模型 token <strong id="budget-token-text">—</strong></div><div class="progress-track"><div class="progress-fill" id="budget-token-bar" style="width:0%"></div></div></div><div class="budget-item"><div class="budget-label">墙钟时间 <strong id="budget-time-text">—</strong></div><div class="progress-track"><div class="progress-fill purple" id="budget-time-bar" style="width:0%"></div></div></div><div class="budget-metrics"><div class="budget-row"><span>模型调用</span><strong id="budget-calls">—</strong></div><div class="budget-row"><span>生成速度</span><strong id="budget-speed">—</strong></div><div class="budget-row"><span>缓存命中</span><strong id="budget-cached">—</strong></div><div class="budget-row"><span>上下文用量</span><strong id="budget-context">—</strong></div><div class="budget-row"><span>云端调用</span><strong id="budget-cloud">—</strong></div><div class="budget-row"><span>工具耗时</span><strong id="budget-tools">—</strong></div></div>`;
  }
  const turn=run.turns?.at(-1)?.id;
  const calls=(run.calls||[]).slice(run.budget_call_start||0).filter(call=>!turn||call.turn===turn);
  const out=calls.reduce((sum,call)=>sum+(call.output_tokens||0),0);
  const wall=calls.reduce((sum,call)=>sum+(call.wall_seconds||0),0);
  const cached=b.cached_tokens||0;
  const context=(calls[calls.length-1]||{}).context_tokens||0;
  const speed=wall?out/wall:0;
  const set=(id,value)=>{const el=document.getElementById(id);if(el&&el.textContent!==value)el.textContent=value;};
  const bar=(id,percent)=>{const el=document.getElementById(id);if(!el)return;const width=Math.min(100,Math.max(0,percent))+'%';if(el.style.width!==width)el.style.width=width;};
  set('budget-token-text',`${number(b.total_tokens)} / ${b.token_limit===null||b.token_limit===undefined?'不限':number(b.token_limit)}`);
  bar('budget-token-bar',b.token_limit?b.total_tokens/b.token_limit*100:0);
  set('budget-time-text',`${time(Math.max(b.elapsed_seconds||0,wall))} / ${b.seconds_limit===null||b.seconds_limit===undefined?'不限':time(b.seconds_limit)}`);
  bar('budget-time-bar',b.seconds_limit?(b.elapsed_seconds||0)/b.seconds_limit*100:0);
  set('budget-calls',`${b.model_calls||0} 次`);
  set('budget-speed',`${speed.toFixed(1)} tok/s`);
  set('budget-cached',`${number(cached)} token`);
  set('budget-context',calls.at(-1)?.context_tokens==null?'未返回':`${number(context)} token`);
  set('budget-cloud',`${b.cloud_calls||0} 次`);
  const toolSeconds=(run.steps||[]).slice(run.branch_step_start||0).filter(step=>!turn||step.turn===turn).reduce((sum,step)=>sum+(step.elapsed_seconds||0),0);
  set('budget-tools',`${toolSeconds.toFixed(2)} s`);
}
const baseResearchRun=renderRun;
renderRun=function(){baseResearchRun();renderWorkflow(run);renderChain(run);};
const baseBudget=renderBudget;
renderBudget=function(){if(run&&run.domain==='research'){renderBudgetResearch();return;}baseBudget();};

/* Multi-turn: the session page shows every turn and lets the user continue. */
function renderSessionTurns(run){
  const box=$('#session-turns');
  if(!box)return;
  renderSessionLayout(run);
  if(!run||run.domain!=='research'){box.innerHTML='';return;}
  const turns=run.turns?.length?run.turns:[{question:run.question,status:run.status,conclusion:run.report?.conclusion||run.stop_reason||run.error||''}];
  const stamp=JSON.stringify([run.id,run.status,turns,run.report]);if(box._turnStamp===stamp)return;
  const historyOpen=box.querySelector('.session-turn-history')?.open;
  const paperOpen=new Set([...box.querySelectorAll('.paper-preview[open]')].map(el=>el.dataset.turn));box._turnStamp=stamp;
  const card=(turn,index)=>{
    const latest=index===turns.length-1,statusKey=latest?run.status:turn.status,status=statuses[statusKey]||statusKey||'',question=index>0||turn.question!==run.question;
    const report=latest?run.report:turn.report,paper=report?.manuscript,conclusion=report?.conclusion||turn.conclusion;
    const manuscript=paper?`<details class="paper-preview" data-turn="${index}" ${paperOpen.has(String(index))?'open':''}><summary>阅读全文</summary>${[['摘要',paper.abstract],['研究问题与背景',paper.introduction],['研究方法',paper.methods],['研究结果',paper.results],['讨论与局限',paper.discussion],['结论',paper.conclusion]].filter(([,text])=>text).map(([title,text])=>`<section><h3>${esc(title)}</h3><p>${esc(text)}</p></section>`).join('')}</details>`:'';
    const warnings=report?.warnings?.length?`<section class="report-warnings" role="note"><strong>报告检查提醒</strong><ul>${report.warnings.map(item=>`<li>${esc(item)}</li>`).join('')}</ul></section>`:'';
    return `<article class="turn-block">${question?`<div class="turn-question"><span class="turn-role">追问</span><p>${esc(turn.question||'')}</p></div>`:''}<div class="turn-answer"><div class="turn-answer-heading"><strong>${latest?'研究结果':'历史结果'}</strong><span>第 ${index+1} 轮</span><span class="pill subtle">${esc(status)}</span></div><p class="${conclusion?'':'muted'}">${esc(conclusion||(statusKey==='queued'?'排队等待运行名额。':statusKey==='running'?'研究正在进行，执行记录中会显示最新步骤。':'本轮尚未形成结论。'))}</p>${warnings}${manuscript}</div></article>`;
  };
  box.innerHTML=(turns.length>1?`<details class="session-turn-history" ${historyOpen?'open':''}><summary>此前 ${turns.length-1} 轮对话</summary>${turns.slice(0,-1).map(card).join('')}</details>`:'')+card(turns.at(-1),turns.length-1);
}
const baseTurnsRun=renderRun;
$('#session-report-format').addEventListener('change',()=>{if(run)$('#session-report').href=`/api/runs/${run.id}/report?format=${$('#session-report-format').value}`;});
renderRun=function(){baseTurnsRun();renderSessionTurns(run);};
$$('[data-session-tab]').forEach(button=>{
  button.addEventListener('click',()=>setSessionTab(button.dataset.sessionTab));
  button.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();const tabs=$$('[data-session-tab]');let index=tabs.indexOf(button);index=event.key==='Home'?0:event.key==='End'?tabs.length-1:(index+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;setSessionTab(tabs[index].dataset.sessionTab);tabs[index].focus();});
});
$('#chain-turn').addEventListener('change',event=>{chainView.turn=event.target.value;renderChain(run);});
$('#chain-status').addEventListener('change',event=>{chainView.status=event.target.value;renderChain(run);});
$('#chain-search').addEventListener('input',event=>{chainView.query=event.target.value.trim();renderChain(run);});
$('#followup-send')?.addEventListener('click',async()=>{
  const question=$('#followup-input').value.trim();
  if(!run){toast('先打开一个研究会话');return;}
  if(question.length<8){toast('追问至少 8 个字符');return;}
  const actionId=run.id;sessionActionIds.add(actionId);const button=$('#followup-send');button.disabled=true;
  try{
    await api(`/api/runs/${actionId}/continue`,{question});
    $('#followup-input').value='';
    toast('已在本会话中开始下一轮研究。');
    await refresh(true);
  }catch(error){toast(error.message);}finally{sessionActionIds.delete(actionId);button.disabled=false;}
});

Object.assign(statuses,{pausing:'正在暂停',paused:'已暂停',incomplete:'未完成',inconclusive:'证据不足',done:'阶段结束'});
$('#chain-kind').addEventListener('change',event=>{chainView.kind=event.target.value;renderChain(run);});
$('#session-pause').addEventListener('click',async()=>{try{await api(`/api/runs/${run.id}/pause`,{});toast('已请求暂停，等待当前调用结束。');await refresh(true);}catch(error){toast(error.message);}});
$('#session-resume').addEventListener('click',async()=>{const actionId=run.id,button=$('#session-resume');sessionActionIds.add(actionId);button.disabled=true;try{await api(`/api/runs/${actionId}/resume`,{});toast('已提交继续研究；运行名额用满时排队。');await refresh(true);}catch(error){toast(error.message);}finally{sessionActionIds.delete(actionId);button.disabled=false;}});
let forkSelection=null;
function openFork(node){forkSelection={runId:run.id,nodeId:node.id};$('#fork-question').value='';$('#fork-routing').checked=false;$('#fork-description').textContent=node.legacy?'该历史记录没有检查点。新分支只继承提交材料与截至此处的记录作为上下文，重新收集证据并规划研究。':'恢复到该调用结束时的检查点，后续结果不会带入分支。';$('#fork-dialog').showModal();}
$('#fork-form').addEventListener('submit',async event=>{event.preventDefault();if(!forkSelection)return;const button=$('#fork-submit');button.disabled=true;try{
  const created=await api(`/api/runs/${forkSelection.runId}/fork`,{node_id:forkSelection.nodeId,question:$('#fork-question').value,routing:$('#fork-routing').checked?(state.settings.routing||{}):undefined});$('#fork-dialog').close();await refresh(true);await selectRun(created.id);toast('已创建分支并继续研究。');
}catch(error){toast(error.message);}finally{button.disabled=false;button.textContent='创建分支并继续';}});
