/* Vue 3 app — Meituan Demo */
const { createApp, ref, reactive, computed, onMounted } = Vue;

const app = createApp({
  setup() {
    const currentTab = ref('demo');
    const demoTasks = ref([]);
    const demoDetail = ref(null);
    const showEvalReport = ref(false);
    const showOptReport = ref(false);
    const historyTasks = ref([]);
    const historyDetail = ref(null);

    const customStep = ref(1);
    const caseText = ref('');
    const selectedPreset = ref('');
    const parseResult = ref(null);
    const presets = ref([]);
    const slots = reactive([
      { key: 'assistant', label: '被评测模型', role: '对话中的 AI 客服', api_key: '', base_url: '', model: '', server_side: false, test_result: '', test_ok: false },
      { key: 'simulator', label: '用户模拟器', role: '画像生成 + 模拟用户', api_key: '', base_url: '', model: '', server_side: false, test_result: '', test_ok: false },
      { key: 'evaluator', label: '评测引擎', role: '9 维度 Judge', api_key: '', base_url: '', model: '', server_side: false, test_result: '', test_ok: false },
      { key: 'optimizer', label: '优化引擎', role: '生成优化建议', api_key: '', base_url: '', model: '', server_side: false, test_result: '', test_ok: false },
    ]);
    const runEval = ref(true);
    const runOptimize = ref(false);
    const canProceed = computed(() => caseText.value.trim());

    // ——— real-time ———
    const currentPhase = ref('');
    const progressPct = ref(0);
    const logs = ref([]);
    const dialogueCards = ref([]);
    const taskResult = ref(null);
    const currentTaskId = ref(null);
    const running = ref(false);
    let ws = null;

    // ——— helpers ———
    function switchTab(tab) { currentTab.value = tab; if (tab === 'history') loadHistory(); if (tab === 'demo') loadDemoConfig(); }
    function formatTime(ts) { if (!ts) return ''; try { return new Date(ts).toLocaleString('zh-CN'); } catch (e) { return ts; } }
    function statusLabel(s) { const m = { completed: 'completed', failed: 'failed', timeout: 'timeout', cancelled: 'cancelled', queued: 'queued' }; return m[s] || s; }
    function addLog(msg, level) { logs.value.push({ time: new Date().toISOString(), message: msg, level: level || 'info' }); }

    // ——— API ———
    async function apiGet(url) {
      const r = await fetch(url);
      if (!r.ok) throw new Error(await r.text());
      return await r.json();
    }
    async function apiPost(url, body) {
      const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      if (!r.ok) throw new Error(await r.text());
      return await r.json();
    }

    // ——— markdown render ———
    function renderMarkdown(text) {
      if (!text) return '';
      let h = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      h = h.replace(/```(\w*)\n([\s\S]*?)```/g,(m,l,c)=>'<pre><code>'+c.trim()+'</code></pre>');
      h = h.replace(/`([^`]+)`/g,'<code>$1</code>');
      h = h.replace(/^#### (.+)$/gm,'<h5>$1</h5>'); h = h.replace(/^### (.+)$/gm,'<h5>$1</h5>');
      h = h.replace(/^## (.+)$/gm,'<h4>$1</h4>'); h = h.replace(/^# (.+)$/gm,'<h3>$1</h3>');
      h = h.replace(/^---+/gm,'<hr>'); h = h.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
      h = h.replace(/(\|.+\|\n\|[-| :]+\|\n((?:\|.+\|\n?)*))/g,(m)=>{
        const l=m.trim().split('\n').filter(l=>l.trim().startsWith('|'));
        if(l.length<2)return m;
        const mr=(line)=>{const c=line.split('|').map(c=>c.trim()).filter(c=>c&&!c.match(/^[-: ]+$/));return'<tr>'+c.map(c=>'<td>'+c+'</td>').join('')+'</tr>';};
        const hc=l[0].split('|').map(c=>c.trim()).filter(c=>c);
        let t='<table class="md-table"><thead><tr>'+hc.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>';
        for(let i=2;i<l.length;i++) t+=mr(l[i]); t+='</tbody></table>'; return t;
      });
      h = h.replace(/^- (.+)$/gm,'<li>$1</li>'); h = h.replace(/((?:<li>.*<\/li>\n?)+)/g,'<ul>$1</ul>');
      h = h.replace(/【(.+?)】/g,'<strong>$1</strong>'); return h;
    }

    // ——— demo ———
    async function loadDemoConfig() {
      try {
        const d = await apiGet('/api/demo/config');
        demoTasks.value = d.demo_tasks || [];
      } catch (e) { console.error('loadDemoConfig:', e); }
    }
    async function loadDemoDetail(taskId) {
      try {
        const d = await apiGet('/api/tasks/' + taskId + '/detail');
        demoDetail.value = d;
      } catch (e) { console.error('loadDemoDetail:', e); }
    }

    // ——— presets ———
    async function loadPresets() { try { presets.value = await apiGet('/api/presets'); } catch (e) { console.error('loadPresets:', e); } }
    async function loadPresetCase() { if (!selectedPreset.value) return; try { const d = await apiGet('/api/presets/' + selectedPreset.value); caseText.value = d.instruction; } catch (e) { console.error('loadPresetCase:', e); } }
    async function parsePreview() { try { parseResult.value = await apiPost('/api/cases/parse', { case_text: caseText.value }); } catch (e) { console.error('parsePreview:', e); } }
    async function testSlot(slot) { slot.test_result = 'testing...'; try { const d = await apiPost('/api/test-connection', { slot: slot.key, api_key: slot.api_key || undefined, base_url: slot.base_url || undefined, model: slot.model || undefined }); slot.test_ok = d.ok; slot.test_result = d.message; } catch (e) { slot.test_ok = false; slot.test_result = 'failed'; } }

    // ——— start task ———
    async function startTask() {
      if (!slots[0].api_key) { alert('请先在 Step 2 填写被评测模型的 API Key'); return; }
      running.value = true; customStep.value = 3; progressPct.value = 0; logs.value = []; dialogueCards.value = []; taskResult.value = null;
      const bs = s => ({ api_key: s.api_key || null, base_url: s.base_url || null, model: s.model || null });
      try {
        const d = await apiPost('/api/tasks', { case_text: caseText.value, demo_mode: false, llm_config: { assistant: bs(slots[0]), simulator: bs(slots[1]), evaluator: bs(slots[2]), optimizer: bs(slots[3]) }, run_eval: runEval.value, run_optimize: runOptimize.value });
        currentTaskId.value = d.task_id; addLog('Task: ' + d.task_id); connectWS(d.task_id);
      } catch (e) { running.value = false; addLog('Failed: ' + e.message, 'error'); }
    }
    function connectWS(taskId) {
      const u = (location.protocol === 'https:' ? 'wss' : 'ws') + '://' + location.host + '/ws/task/' + taskId;
      ws = new WebSocket(u); ws.onopen = () => addLog('WS connected');
      ws.onmessage = e => { try { const m = JSON.parse(e.data); if (m.type === 'queued') addLog('Queue #' + m.position); else if (m.type === 'phase') { currentPhase.value = m.phase + ' ' + m.status; const pm = { parsing: 5, profiles: 20, dialogues: 60, eval: 85, optimize: 95 }; progressPct.value = pm[m.phase] || 10; } else if (m.type === 'log') addLog(m.message, m.level); else if (m.type === 'progress' && m.phase === 'dialogues') { if (m.total > 0) progressPct.value = 20 + Math.round((m.completed / m.total) * 40); } else if (m.type === 'dialogue_card') { dialogueCards.value.push(m); progressPct.value = Math.min(80, progressPct.value + 3); } else if (m.type === 'completed') { running.value = false; progressPct.value = 100; addLog('Done!'); apiGet('/api/tasks/' + currentTaskId.value + '/detail').then(d => { if (d) taskResult.value = d; }).catch(e => { addLog('Failed to load detail: ' + e.message, 'error'); }); if (ws) ws.close(); } else if (m.type === 'error') { addLog(m.message, 'error'); if (!m.recoverable) { running.value = false; if (ws) ws.close(); } } } catch (err) { console.error('WS message:', err); } };
      ws.onclose = () => { if (running.value) setTimeout(() => { if (running.value) connectWS(taskId); }, 2000); };
    }

    // ——— download ———
    const EXT_MAP = { report_md: '.md', report_json: '.json', conversations_json: '.json', batch_summary: '.json', all_zip: '.zip' };
    async function downloadFile(taskId, type) {
      try {
        const r = await fetch('/api/tasks/' + taskId + '/download/' + type);
        if (!r.ok) throw new Error('fail');
        const blob = await r.blob();
        const ext = EXT_MAP[type] || '';
        const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = type + '_' + taskId + ext; document.body.appendChild(a); a.click(); document.body.removeChild(a); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
      } catch (e) { addLog('Download fail: ' + e.message, 'error'); }
    }

    // ——— history ———
    async function loadHistory() { try { historyTasks.value = await apiGet('/api/tasks?limit=50'); } catch (e) { console.error('loadHistory:', e); } }
    async function loadHistoryDetail(taskId) { try { historyDetail.value = await apiGet('/api/tasks/' + taskId + '/detail'); } catch (e) { console.error('loadHistoryDetail:', e); } }
    async function deleteTask(taskId) { if (!confirm('Delete?')) return; try { await fetch('/api/tasks/' + taskId, { method: 'DELETE' }); historyTasks.value = historyTasks.value.filter(t => t.task_id !== taskId); } catch (e) { console.error('deleteTask:', e); } }

    // ——— init ———
    onMounted(async () => {
        await loadPresets(); await loadDemoConfig();
        // 检查是否有运行中的任务，自动重连
        try {
            const tasks = await apiGet('/api/tasks?limit=10');
            const terminalStates = ['completed', 'failed', 'timeout', 'cancelled'];
            const runningTask = tasks.find(t => !terminalStates.includes(t.status));
            if (runningTask) {
                currentTaskId.value = runningTask.task_id;
                running.value = true;
                customStep.value = 3;
                connectWS(runningTask.task_id);
                addLog('Reconnected to running task: ' + runningTask.task_id);
            }
        } catch(e) { console.error('onMounted reconnect:', e); }
    });

    return {
      currentTab, switchTab, formatTime, statusLabel,
      demoTasks, demoDetail, showEvalReport, showOptReport, loadDemoConfig, loadDemoDetail,
      historyTasks, historyDetail, loadHistory, loadHistoryDetail, deleteTask,
      customStep, caseText, selectedPreset, parseResult, parsePreview, presets, loadPresetCase,
      slots, testSlot, canProceed, runEval, runOptimize,
      currentPhase, progressPct, logs, dialogueCards, taskResult, currentTaskId, running,
      startTask, downloadFile,
    };
  },
});

// —— detail-block component ——
app.component('detail-block', {
  props: ['d', 'score', 'tid'],
  template: `
<div style="padding:1rem" v-if="d">
  <h5>案例信息</h5>
  <table><tbody>
    <tr><td width="120"><strong>案例标题</strong></td><td>{{ d.case_title }}</td></tr>
    <tr v-if="d.conversation_summary?.complexity_score != null"><td><strong>指令复杂度</strong></td><td>{{ d.conversation_summary.complexity_score }}</td></tr>
    <tr v-if="d.conversation_summary?.profile_label"><td><strong>画像标签</strong></td><td>{{ d.conversation_summary.profile_label }}</td></tr>
    <tr v-if="d.conversation_summary?.adversarial_strategies?.length"><td><strong>对抗策略</strong></td><td>{{ d.conversation_summary.adversarial_strategies.join(', ') }}</td></tr>
  </tbody></table>

  <h5>评测总览</h5>
  <div class="score-big">{{ score || d.eval_result?.total_score_100 || '-' }}<small> 分</small></div>
  <p style="text-align:center;font-size:1.5rem">{{ d.eval_result?.rating_label || 'N/A' }}</p>
  <div v-if="d.eval_result?.total_indicative_score" style="text-align:center">加权综合分: {{ d.eval_result.total_indicative_score }}</div>
  <div v-if="d.eval_result?.confidence" style="margin:0.5rem 0">置信度: {{ d.eval_result.confidence.level || (d.eval_result.confidence.overall ? (d.eval_result.confidence.overall*100).toFixed(0)+'%' : 'N/A') }}</div>
  <div v-if="d.eval_result?.summary" style="padding:0.8rem;background:var(--pico-card-sectioning-background-color);border-radius:4px;margin:0.5rem 0;font-family:monospace;font-size:0.85rem;white-space:pre-wrap;max-height:300px;overflow:auto">{{ d.eval_result.summary }}</div>

  <h5>9 维度评分</h5>
  <table><thead><tr><th>维度</th><th>评级</th><th>分数</th></tr></thead><tbody>
    <tr v-for="(score, dim) in (d.eval_result?.indicative_scores || {})" :key="dim"><td><strong>{{ dim }}</strong></td><td>{{ d.eval_result?.ratings?.[dim] || '-' }}</td><td>{{ score }}</td></tr>
  </tbody></table>

  <h5>模型配置</h5>
  <table><tbody><tr v-for="(m, slot) in (d.effective_models || {})" :key="slot"><td width="120"><strong>{{ slot }}</strong></td><td>{{ m }}</td></tr></tbody></table>

  <h5 v-if="getAttr(d).length">缺陷归因 ({{ getAttr(d).length }} 条)</h5>
  <div v-for="(a,i) in getAttr(d).slice(0,10)" :key="i" style="margin:0.3rem 0;padding:0.4rem 0.6rem;background:var(--pico-card-sectioning-background-color);border-radius:4px;font-size:0.85rem;border-left:3px solid var(--pico-primary)"><strong>#{{ i+1 }}</strong> [{{ a.source }}] {{ a.category }} — {{ (a.description||'').substring(0,200) }}<span v-if="a.confidence" style="color:var(--pico-muted-color)"> ({{ (a.confidence*100).toFixed(0) }}%)</span></div>

  <h5 v-if="getImp(d).length">改进建议 ({{ getImp(d).length }} 条)</h5>
  <div v-for="(s,i) in getImp(d)" :key="i" style="margin:0.3rem 0;padding:0.5rem 0.8rem;background:var(--pico-card-sectioning-background-color);border-radius:4px;border-left:3px solid var(--pico-primary)"><strong>#{{ i+1 }}.</strong> {{ typeof s === 'string' ? s.substring(0,500) : JSON.stringify(s).substring(0,500) }}</div>

  <h5>对话记录 ({{ getTurns(d).length }} 轮)</h5>
  <div v-if="getTurns(d).length" style="max-height:400px;overflow:auto;margin:0.5rem 0"><div v-for="(turn,i) in getTurns(d)" :key="i" style="display:flex;gap:0.5rem;padding:0.3rem 0;border-bottom:1px solid var(--pico-muted-border-color);align-items:flex-start"><span style="min-width:40px;font-weight:700;font-size:0.8rem;padding:0.15rem 0.4rem;border-radius:4px;text-align:center" :style="isAst(turn)?'background:#d4edda;color:#155724':'background:#cce5ff;color:#004085'">{{ isAst(turn) ? '客服' : '用户' }}</span><span style="font-size:0.9rem;line-height:1.4;flex:1">{{ (turn.content||turn.text||'').substring(0,300) }}{{ (turn.content||turn.text||'').length > 300 ? '...' : '' }}</span></div></div>

  <h5>报告</h5>
  <div class="grid"><button class="secondary" @click="showR = !showR">{{ showR ? '隐藏' : '展开' }}评测报告</button><button class="secondary" @click="showO = !showO">{{ showO ? '隐藏' : '展开' }}优化报告</button><button class="secondary" @click="dl(tid, 'all_zip')">下载 ZIP</button></div>
  <div v-if="showR && d.report_md" class="rendered-report" v-html="md(d.report_md)" style="margin-top:1rem;padding:1rem;background:var(--pico-card-background-color);border:1px solid var(--pico-muted-border-color);border-radius:6px;max-height:500px;overflow:auto;font-size:0.9rem;line-height:1.6"></div>
  <div v-if="showO && d.optimization_report_md" class="rendered-report" v-html="md(d.optimization_report_md)" style="margin-top:1rem;padding:1rem;background:var(--pico-card-background-color);border:1px solid var(--pico-muted-border-color);border-radius:6px;max-height:500px;overflow:auto;font-size:0.9rem;line-height:1.6"></div>
</div>`,
  data() { return { showR: false, showO: false }; },
  methods: {
    getAttr(d) { return d?.eval_result?.attributions || []; },
    getImp(d) { return d?.eval_result?.improvement_suggestions || []; },
    getTurns(d) { const c = d?.conversation_summary; if (!c) return []; if (Array.isArray(c.turns)) return c.turns; return c.turns && Array.isArray(c.turns) ? c.turns : []; },
    isAst(t) { return t && (t.speaker === 'assistant' || t.speaker === 'system'); },
    dl(tid, type) { const em = { report_md: '.md', report_json: '.json', conversations_json: '.json', batch_summary: '.json', all_zip: '.zip' }; fetch('/api/tasks/' + tid + '/download/' + type).then(r => { if (!r.ok) throw new Error('Download failed'); return r.blob(); }).then(b => { const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = type + '_' + tid + (em[type] || ''); document.body.appendChild(a); a.click(); document.body.removeChild(a); setTimeout(() => URL.revokeObjectURL(a.href), 1000); }).catch(e => { console.error('dl error:', e); }); },
    md(text) { if (!text) return ''; let h = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); h = h.replace(/```(\w*)\n([\s\S]*?)```/g,(m,l,c)=>'<pre><code>'+c.trim()+'</code></pre>'); h = h.replace(/`([^`]+)`/g,'<code>$1</code>'); h = h.replace(/^#### (.+)$/gm,'<h5>$1</h5>'); h = h.replace(/^### (.+)$/gm,'<h5>$1</h5>'); h = h.replace(/^## (.+)$/gm,'<h4>$1</h4>'); h = h.replace(/^# (.+)$/gm,'<h3>$1</h3>'); h = h.replace(/^---+/gm,'<hr>'); h = h.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>'); h = h.replace(/(\|.+\|\n\|[-| :]+\|\n((?:\|.+\|\n?)*))/g,(m)=>{const l=m.trim().split('\n').filter(l=>l.trim().startsWith('|'));if(l.length<2)return m;const mr=(line)=>{const c=line.split('|').map(c=>c.trim()).filter(c=>c&&!c.match(/^[-: ]+$/));return'<tr>'+c.map(c=>'<td>'+c+'</td>').join('')+'</tr>';};const hc=l[0].split('|').map(c=>c.trim()).filter(c=>c);let t='<table class="md-table"><thead><tr>'+hc.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>';for(let i=2;i<l.length;i++) t+=mr(l[i]);t+='</tbody></table>';return t;}); h = h.replace(/^- (.+)$/gm,'<li>$1</li>'); h = h.replace(/((?:<li>.*<\/li>\n?)+)/g,'<ul>$1</ul>'); h = h.replace(/【(.+?)】/g,'<strong>$1</strong>'); return h; },
  },
});

app.mount('#app');
