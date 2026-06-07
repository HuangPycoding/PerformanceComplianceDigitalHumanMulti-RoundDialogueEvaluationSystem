/* Vue 3 app — Meituan Demo */
const { createApp, ref, reactive, computed, watch, onMounted } = Vue;

const app = createApp({
  setup() {
    const currentTab = ref('demo'), currentTaskId = ref(null), running = ref(false);
    const demoTasks = ref([]), demoDetail = ref(null);
    const customStep = ref(1), caseText = ref(''), selectedPreset = ref(''), parseResult = ref(null), presets = ref([]);
    const slots = reactive([
      { key: 'assistant', label: 'be evaluated model', role: 'AI agent in dialogue', api_key: '', base_url: '', model: '', server_side: false, test_result: '', test_ok: false },
      { key: 'simulator', label: 'user simulator', role: 'profile generation + user simulation', api_key: '', base_url: '', model: '', server_side: false, test_result: '', test_ok: false },
      { key: 'evaluator', label: 'evaluation engine', role: '9-dimension Judge', api_key: '', base_url: '', model: '', server_side: false, test_result: '', test_ok: false },
      { key: 'optimizer', label: 'optimization engine', role: 'generate optimization suggestions', api_key: '', base_url: '', model: '', server_side: false, test_result: '', test_ok: false },
    ]);
    const nProfiles = ref(3), runEval = ref(true), runOptimize = ref(false);
    const currentPhase = ref(''), progressPct = ref(0), logs = ref([]), dialogueCards = ref([]), taskResult = ref(null);
    let ws = null;
    const historyTasks = ref([]), historyDetail = ref(null);
    const canProceed = computed(() => caseText.value.trim());

    function switchTab(tab) { currentTab.value = tab; if (tab === 'history') loadHistory(); if (tab === 'demo') loadDemoConfig(); }
    function formatTime(ts) { if (!ts) return ''; try { return new Date(ts).toLocaleString('zh-CN'); } catch (e) { return ts; } }
    function statusLabel(s) { const m = { completed: 'DONE', failed: 'FAIL', timeout: 'TIMEOUT', cancelled: 'CANCEL', queued: 'QUEUE' }; return (m[s] || s); }
    async function api(url, opts = {}) { try { const r = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...opts }); if (!r.ok) throw new Error(await r.text()); return await r.json(); } catch (e) { addLog(e.message, 'error'); throw e; } }
    function addLog(msg, level = 'info') { logs.value.push({ time: new Date().toISOString(), message: msg, level }); }
    async function loadDemoConfig() { try { demoTasks.value = (await api('/api/demo/config')).demo_tasks || []; } catch (e) {} }
    async function loadDemoDetail(taskId) { try { const d = await api('/api/tasks/' + taskId + '/detail'); demoDetail.value = d; } catch (e) { demoDetail.value = null; } }
    async function loadDemoCase() { if (caseText.value) return; try { const d = await api('/api/presets'); if (d.length) { presets.value = d; caseText.value = (await api('/api/presets/' + d[0].id)).instruction; } } catch (e) {} }
    async function loadPresets() { try { presets.value = await api('/api/presets'); } catch (e) {} }
    async function loadPresetCase() { if (!selectedPreset.value) return; try { caseText.value = (await api('/api/presets/' + selectedPreset.value)).instruction; } catch (e) {} }
    async function parsePreview() { try { parseResult.value = await api('/api/cases/parse', { method: 'POST', body: JSON.stringify({ case_text: caseText.value }) }); } catch (e) { parseResult.value = null; } }
    async function testSlot(slot) { slot.test_result = 'testing...'; try { const b = { slot: slot.key }; if (slot.api_key) b.api_key = slot.api_key; if (slot.base_url) b.base_url = slot.base_url; if (slot.model) b.model = slot.model; const d = await api('/api/test-connection', { method: 'POST', body: JSON.stringify(b) }); slot.test_ok = d.ok; slot.test_result = d.message; } catch (e) { slot.test_ok = false; slot.test_result = 'failed'; } }
    async function startTask() {
      running.value = true; customStep.value = 3; progressPct.value = 0; logs.value = []; dialogueCards.value = []; taskResult.value = null;
      const bs = (s) => ({ api_key: s.api_key || null, base_url: s.base_url || null, model: s.model || null });
      try {
        const b = { case_text: caseText.value, demo_mode: false, llm_config: { assistant: bs(slots[0]), simulator: bs(slots[1]), evaluator: bs(slots[2]), optimizer: bs(slots[3]) }, n_profiles: nProfiles.value, run_eval: runEval.value, run_optimize: runOptimize.value };
        currentTaskId.value = (await api('/api/tasks', { method: 'POST', body: JSON.stringify(b) })).task_id;
        addLog('Task: ' + currentTaskId.value); connectWS(currentTaskId.value);
      } catch (e) { running.value = false; addLog('Failed: ' + e.message, 'error'); }
    }
    function connectWS(taskId) {
      const u = (location.protocol === 'https:' ? 'wss' : 'ws') + '://' + location.host + '/ws/task/' + taskId;
      ws = new WebSocket(u); ws.onopen = () => addLog('WS connected');
      ws.onmessage = (e) => { try { handleWSEvent(JSON.parse(e.data)); } catch (err) {} };
      ws.onclose = () => { if (running.value) { setTimeout(() => { if (running.value) connectWS(taskId); }, 2000); } };
    }
    function handleWSEvent(msg) {
      switch (msg.type) {
        case 'queued': addLog('Queue #' + msg.position); break;
        case 'phase': currentPhase.value = msg.phase + ' ' + msg.status; progressPct.value = { parsing: 5, profiles: 20, dialogues: 60, eval: 85, optimize: 95 }[msg.phase] || 10; break;
        case 'log': addLog(msg.message, msg.level); break;
        case 'progress': if (msg.phase === 'dialogues') progressPct.value = 20 + Math.round((msg.completed / msg.total) * 40); break;
        case 'dialogue_card': dialogueCards.value.push(msg); progressPct.value = Math.min(60, progressPct.value + 2); break;
        case 'completed': running.value = false; progressPct.value = 100; addLog('Done!'); loadTaskDetail(currentTaskId.value).then(d => { if (d) taskResult.value = d; }); if (ws) ws.close(); break;
        case 'error': addLog(msg.message, 'error'); if (!msg.recoverable) { running.value = false; if (ws) ws.close(); } break;
      }
    }
    async function loadTaskDetail(taskId) { try { return await api('/api/tasks/' + taskId + '/detail'); } catch (e) { return null; } }
    async function downloadFile(taskId, type) { try { const r = await fetch('/api/tasks/' + taskId + '/download/' + type); if (!r.ok) throw new Error('fail'); const b = await r.blob(); const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = type + '_' + taskId; a.click(); URL.revokeObjectURL(a.href); } catch (e) { addLog('Download fail: ' + e.message, 'error'); } }
    async function loadHistory() { try { historyTasks.value = await api('/api/tasks?limit=50'); } catch (e) {} }
    async function loadHistoryDetail(taskId) { try { const d = await api('/api/tasks/' + taskId + '/detail'); historyDetail.value = d; } catch (e) { historyDetail.value = null; } }
    async function deleteTask(taskId) { if (!confirm('Delete?')) return; try { await api('/api/tasks/' + taskId, { method: 'DELETE' }); historyTasks.value = historyTasks.value.filter(t => t.task_id !== taskId); } catch (e) {} }
    onMounted(async () => { await loadPresets(); await loadDemoConfig(); await loadDemoCase(); });

    return {
      currentTab, switchTab, formatTime, statusLabel, demoTasks, demoDetail, loadDemoConfig, loadDemoDetail, loadDemoCase,
      customStep, caseText, selectedPreset, parseResult, parsePreview, presets, loadPresetCase, slots, testSlot, canProceed,
      nProfiles, runEval, runOptimize, progressPct, currentPhase, logs, dialogueCards, taskResult, currentTaskId, running,
      startTask, downloadFile, historyTasks, historyDetail, loadHistory, loadHistoryDetail, deleteTask,
    };
  },
});

app.component('full-detail', {
  props: ['detail', 'taskId'], emits: ['close'], data() { return { showReport: false, showOptReport: false }; },
  template: `
<div>
  <h5>Case Info</h5>
  <table><tbody>
    <tr><td width="120"><strong>Case Title</strong></td><td>{{ detail.case_title }}</td></tr>
    <tr v-if="convData().complexity_score != null"><td><strong>Complexity</strong></td><td>{{ convData().complexity_score }}</td></tr>
    <tr v-if="convData().profile_label"><td><strong>Profile Label</strong></td><td>{{ convData().profile_label }}</td></tr>
    <tr v-if="convData().adversarial_strategies?.length"><td><strong>Adversarial</strong></td><td>{{ convData().adversarial_strategies.join(', ') }}</td></tr>
  </tbody></table>

  <h5>User Profile</h5>
  <div v-if="getProfileText()" class="profile-text" v-html="renderMarkdown(getProfileText())"></div>
  <p v-else>No profile data</p>

  <h5>Model Config</h5>
  <table><tbody><tr v-for="(m, slot) in (detail.effective_models || {})" :key="slot"><td width="120"><strong>{{ slot }}</strong></td><td>{{ m }}</td></tr></tbody></table>

  <h5>Score Overview</h5>
  <div class="score-big">{{ detail.eval_result?.total_score_100 || '-' }}<small> / 100</small></div>
  <div class="rating-big">{{ detail.eval_result?.rating_label || getOverallRating(detail) }}</div>
  <div v-if="detail.eval_result?.total_indicative_score" style="text-align:center;margin-bottom:1rem">Weighted: {{ detail.eval_result.total_indicative_score?.toFixed?.(1) || detail.eval_result.total_indicative_score }}</div>
  <div v-if="detail.eval_result?.confidence" style="margin-bottom:0.5rem"><strong>Confidence:</strong> <span v-if="detail.eval_result.confidence.level">Level: {{ detail.eval_result.confidence.level }}</span><span v-if="detail.eval_result.confidence.overall"> | {{ (detail.eval_result.confidence.overall * 100).toFixed(0) }}%</span></div>
  <div v-if="detail.eval_result?.summary" style="margin-bottom:0.5rem;padding:0.5rem;background:var(--pico-card-sectioning-background-color);border-radius:4px"><strong>Summary:</strong> {{ detail.eval_result.summary }}</div>

  <h5>9 Dimensions</h5>
  <table><thead><tr><th>Dimension</th><th>Rating</th><th>Score</th></tr></thead><tbody>
    <tr v-for="(score, dim) in getDimScores(detail)" :key="dim"><td><strong>{{ dim }}</strong></td><td>{{ getDimRating(detail, dim) }}</td><td>{{ score }}</td></tr>
  </tbody></table>

  <h5>Evaluation Details</h5>
  <div v-if="hasChecklistItems()">
    <details v-for="(items, dim) in detail.eval_result.dimension_checklists" :key="dim">
      <summary>{{ dim }} ({{ (items && items.length) || 0 }} items)</summary>
      <table class="checklist-table" v-if="items && items.length"><thead><tr><th width="60">Status</th><th width="80">Source</th><th>Description</th></tr></thead><tbody>
        <tr v-for="(item, i) in items" :key="i"><td><span class="check-status" :class="'status-' + (item.status || '').toLowerCase().replace('_','-')">{{ item.status || '-' }}</span></td><td>{{ item.source || '-' }}</td><td><div>{{ item.description || JSON.stringify(item) }}</div><small v-if="item.evidence" style="color:gray">Evidence: {{ (item.evidence || '').substring(0, 150) }}</small></td></tr>
      </tbody></table>
    </details>
  </div>
  <div v-else-if="showEvalFallback(detail)">
    <table v-if="detail.eval_result?.indicative_scores"><thead><tr><th>Dim</th><th>Rating</th><th>Score</th></tr></thead><tbody>
      <tr v-for="(score, dim) in getDimScores(detail)" :key="dim"><td><strong>{{ dim }}</strong></td><td>{{ getDimRating(detail, dim) }}</td><td>{{ score }}</td></tr>
    </tbody></table>
    <div v-if="detail.eval_result?.attributions?.length" style="margin-top:0.8rem"><strong>Attributions ({{ detail.eval_result.attributions.length }}):</strong>
      <div v-for="(a, i) in detail.eval_result.attributions.slice(0,10)" :key="i" style="margin:0.3rem 0 0.3rem 1rem;font-size:0.85rem"><strong>#{{ i+1 }}</strong> [{{ a.source }}] {{ a.category }} - {{ (a.description || '').substring(0,180) }}<span v-if="a.confidence" style="color:var(--pico-muted-color)"> ({{ (a.confidence * 100).toFixed(0) }}%)</span></div>
    </div>
    <div v-if="detail.eval_result?.meta_check_alerts?.length" style="margin-top:0.8rem"><strong>Meta Check ({{ detail.eval_result.meta_check_alerts.length }}):</strong>
      <div v-for="(a, i) in detail.eval_result.meta_check_alerts.slice(0,5)" :key="i" style="margin:0.15rem 0 0.15rem 1rem;font-size:0.85rem">{{ typeof a === 'string' ? a.substring(0,200) : (a.message || a.check_type || JSON.stringify(a).substring(0,200)) }}</div>
    </div>
  </div>
  <p v-else>No detail data</p>

  <h5 v-if="detail.eval_result?.improvement_suggestions?.length">Suggestions</h5>
  <div v-if="detail.eval_result?.improvement_suggestions?.length"><div v-for="(s, i) in detail.eval_result.improvement_suggestions" :key="i" style="margin:0.3rem 0;padding:0.5rem 0.8rem;background:var(--pico-card-sectioning-background-color);border-radius:4px;border-left:3px solid var(--pico-primary)"><span style="font-weight:600">#{{ i+1 }}.</span> {{ typeof s === 'string' ? s.substring(0, 800) : JSON.stringify(s).substring(0, 800) }}</div></div>

  <h5>Conversation ({{ convData().total_turns || convData().turns?.length || 0 }} turns)</h5>
  <div class="conv-turns" v-if="getConvTurns().length"><div v-for="(turn, i) in getConvTurns()" :key="i" class="conv-turn"><span class="turn-speaker" :class="isAsst(turn) ? 'asst' : 'user'">{{ isAsst(turn) ? 'Agent' : 'User' }}</span><span class="turn-text">{{ (turn.content || turn.text || '').substring(0, 300) }}{{ (turn.content || turn.text || '').length > 300 ? '...' : '' }}</span></div></div>
  <p v-else>No conversation data</p>

  <div class="grid" style="margin-top:1rem">
    <button class="secondary" @click="showReport = !showReport">{{ showReport ? 'Hide' : 'Show' }} Report</button>
    <button class="secondary" @click="showOptReport = !showOptReport">{{ showOptReport ? 'Hide' : 'Show' }} Optimization</button>
    <button class="secondary" @click="downloadFile(taskId, 'all_zip')">Download ZIP</button>
    <button class="secondary" @click="$emit('close')">Close</button>
  </div>
  <div v-if="showReport && detail.report_md" class="rendered-report" v-html="renderMarkdown(detail.report_md)"></div>
  <div v-if="showOptReport && detail.optimization_report_md" class="rendered-report" v-html="renderMarkdown(detail.optimization_report_md)"></div>
</div>`,
  methods: {
    convData() { return this.detail?.conversation_summary || {}; },
    dimNames() { return ['O','C','E','A','N','Talk','Ask','Interrupt','Hangup','Knowledge','Number','Anxiety','Patience','Critical','Aggressive']; },
    dimCategories() { return ['Big5','Big5','Big5','Big5','Big5','Behavior','Behavior','Behavior','Behavior','Cognition','Cognition','Emotion','Emotion','Adversarial','Adversarial']; },
    renderMarkdown(text) { if(!text)return'';let h=text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');h=h.replace(/```(\w*)\n([\s\S]*?)```/g,(m,l,c)=>'<pre><code>'+c.trim()+'</code></pre>');h=h.replace(/`([^`]+)`/g,'<code>$1</code>');h=h.replace(/^#### (.+)$/gm,'<h5>$1</h5>');h=h.replace(/^### (.+)$/gm,'<h5>$1</h5>');h=h.replace(/^## (.+)$/gm,'<h4>$1</h4>');h=h.replace(/^# (.+)$/gm,'<h3>$1</h3>');h=h.replace(/^---+/gm,'<hr>');h=h.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');h=h.replace(/\*(.+?)\*/g,'<em>$1</em>');h=h.replace(/(\|.+\|\n\|[-| :]+\|\n((?:\|.+\|\n?)*))/g,(m)=>{const l=m.trim().split('\n').filter(l=>l.trim().startsWith('|'));if(l.length<2)return m;const mr=(line)=>{const c=line.split('|').map(c=>c.trim()).filter(c=>c&&!c.match(/^[-: ]+$/));return'<tr>'+c.map(c=>'<td>'+c+'</td>').join('')+'</tr>';};const hc=l[0].split('|').map(c=>c.trim()).filter(c=>c);let t='<table class="md-table"><thead><tr>'+hc.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>';for(let i=2;i<l.length;i++)t+=mr(l[i]);t+='</tbody></table>';return t;});h=h.replace(/^- (.+)$/gm,'<li>$1</li>');h=h.replace(/((?:<li>.*<\/li>\n?)+)/g,'<ul>$1</ul>');h=h.replace(/【(.+?)】/g,'<strong>$1</strong>');return h; },
    getProfileText() { const cs=this.convData(),sv=cs.sampled_vector||[],vv=cs.verified_vector||[],label=cs.profile_label||'';let p=[];if(label)p.push('Label: '+label);if(sv&&sv.length>=15){let t='| Dim | Cat | Sampled | Verified | Bias |\n|------|------|--------|----------|------|\n';for(let i=0;i<15;i++){const v=sv[i]||0,ve=vv[i]||'-';let b='';if(v>=0.85)b='VHigh';else if(v>=0.65)b='High';else if(v>=0.35)b='Mid';else if(v>=0.15)b='Low';else b='VLow';t+='| '+this.dimNames()[i]+' | '+this.dimCategories()[i]+' | '+v.toFixed(2)+' | '+(typeof ve==='number'?ve.toFixed(2):ve)+' | '+b+' |\n';}p.push(t);const cats={};for(let i=0;i<15;i++){const c=this.dimCategories()[i];if(!cats[c])cats[c]=[];cats[c].push({name:this.dimNames()[i],value:sv[i]});}let s=[];for(const[cat,dims]of Object.entries(cats)){const hi=dims.filter(d=>d.value>=0.65).map(d=>d.name);const lo=dims.filter(d=>d.value<=0.35).map(d=>d.name);if(hi.length)s.push(cat+'+: '+hi.join(','));if(lo.length)s.push(cat+'-: '+lo.join(','));}if(cs.adversarial_strategies?.length)s.push('Adv: '+cs.adversarial_strategies.join(','));if(s.length)p.push('Bias: '+s.join('; '));}return p.join('\n\n')||''; },
    hasChecklistItems(){const d=this.detail?.eval_result?.dimension_checklists;return d&&typeof d==='object'&&Object.values(d).some(v=>Array.isArray(v)&&v.length>0);},
    showEvalFallback(d){const e=d?.eval_result;return e&&(e.ratings||e.indicative_scores||e.attributions)?true:false;},
    getOverallRating(d){const r=d?.eval_result?.ratings;if(!r)return'-';const rk={'卓越':5,'良好':4,'合格':3,'需改进':2,'不合格':1};const v=Object.values(r).filter(v=>rk[v]).map(v=>rk[v]);if(!v.length)return'-';const a=v.reduce((a,b)=>a+b,0)/v.length;return a>=4.5?'卓越':a>=3.5?'良好':a>=2.5?'合格':a>=1.5?'需改进':'不合格';},
    getDimScores(d){const e=d?.eval_result;return e?.(e.indicative_scores||e.dimension_scores||e.ratings||{});},
    getDimRating(d,dim){return d?.eval_result?.ratings?.[dim]||'-';},
    getConvTurns(){const c=this.detail?.conversation_summary;if(!c)return[];if(Array.isArray(c))return c;if(c.turns&&Array.isArray(c.turns))return c.turns;return[];},
    isAsst(t){if(!t)return false;const s=t.speaker||t.role||'';return s==='assistant'||s==='system';},
    downloadFile(tid,type){const u='/api/tasks/'+tid+'/download/'+type;fetch(u).then(r=>r.blob()).then(b=>{const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=type+'_'+tid;a.click();URL.revokeObjectURL(a.href);});},
  },
});

app.mount('#app');
