"""Render the formal aligned snapshot as a standalone, offline metrics page."""
from pathlib import Path
import html
import json

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'official':'Official ProcVLM-2B','full_s1_6284':'Full S1 · 6284','full_s2_6284_2724':'Full S2 · 6284 → 2724','full_s1_3600':'Historical S1 · 3600','full_s2_3600_2724':'Historical S2 · S1 3600','subset5_s2':'5% subset S2','subset10_s2':'10% subset S2','subset20_s2':'20% subset S2'}

def render():
    snapshot = json.loads((ROOT/'results/aligned_metrics.json').read_text())
    rows = snapshot['results']; keys = list(rows[0]['metrics'])
    assert len(rows)==16 and sum(x['total'] for x in rows)==292408
    featured = {r['dataset']: r for r in rows if r['model_id']=='full_s2_6284_2724'}
    summary = [f"{featured[split]['metrics']['BF1@5']:.4f}" for split in ['test_full','valid_full']]
    prediction_count = f"{sum(r['total'] for r in rows):,}"
    head=''.join('<th scope="col">'+html.escape(k)+(' ↓' if k=='mMAE' else ' ↑')+'</th>' for k in keys)
    body=[]
    for r in rows:
        cls='featured' if r['model_id']=='full_s2_6284_2724' else ''
        body.append('<tr data-split="'+r['dataset']+'" class="'+cls+'"><th scope="row">'+html.escape(LABELS[r['model_id']])+'</th><td>'+('ID' if r['dataset']=='test_full' else 'OOD')+'</td>'+''.join('<td>'+f'{r["metrics"][k]:.4f}'+'</td>' for k in keys)+'</tr>')
    page='''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ProcVLM · Reproduction results</title>
<style>
:root{color-scheme:light;--ink:#17243a;--muted:#526077;--line:#dbe3ec;--accent:#087f8c}
*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:var(--ink);font:16px/1.6 system-ui,-apple-system,sans-serif}
main{max-width:1240px;margin:auto;padding:56px 28px}header{border-top:5px solid var(--accent);padding:24px 0}
.eyebrow{letter-spacing:.18em;text-transform:uppercase;color:var(--accent);font-size:12px;font-weight:750}h1{font-size:clamp(32px,5vw,52px);line-height:1.15;margin:12px 0}p{max-width:850px;color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:28px 0}.card{background:white;border:1px solid var(--line);border-radius:12px;padding:20px}.card strong{display:block;font-size:32px;font-variant-numeric:tabular-nums}.card span{font-size:13px;color:var(--muted)}
.toolbar{display:flex;gap:8px;align-items:center;margin:25px 0 12px;flex-wrap:wrap}button{cursor:pointer;border:1px solid var(--line);border-radius:7px;background:white;padding:9px 17px;font:inherit;color:var(--ink)}button[aria-pressed=true]{background:var(--ink);color:white}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:12px;background:white}table{border-collapse:collapse;width:100%;min-width:950px;font-size:14px}caption{text-align:left;padding:15px;font-weight:650}th,td{padding:12px 14px;border-top:1px solid var(--line);white-space:nowrap;text-align:right;font-variant-numeric:tabular-nums}thead th{background:#eef2f6}th:first-child{text-align:left}tbody th{font-weight:500}.featured{background:#e8f6f4}.featured th{font-weight:750}tr[hidden]{display:none}
.note{padding:18px 22px;border-left:4px solid #bc8634;background:#fffaee;margin:25px 0}.note p{margin:4px 0;font-size:14px}footer{font-size:13px;color:var(--muted);padding-top:18px}a{color:var(--accent)}
@media(max-width:650px){main{padding:24px 16px}.cards{grid-template-columns:1fr}.card strong{font-size:28px}}
</style></head><body><main>
<header><div class="eyebrow">ProcVLM · independent reproduction</div><h1>Results, with context.</h1><p>Two-stage ProcVQA training and the aligned original-vLLM evaluation. Historical results snapshot: 2026-10-15. Report assembled 2026-10-24; this is not a live training dashboard.</p></header>
<section class="cards" aria-label="Summary"><div class="card"><strong>'''+summary[0]+'''</strong><span>Full S2 · ID BF1@5 ↑</span></div><div class="card"><strong>'''+summary[1]+'''</strong><span>Full S2 · OOD BF1@5 ↑</span></div><div class="card"><strong>'''+prediction_count+'''</strong><span>Predictions · 8 checkpoints × 2 splits</span></div></section>
<nav class="toolbar" aria-label="Filter split"><span>Show:</span><button data-filter="all" aria-pressed="true">Both splits</button><button data-filter="test_full" aria-pressed="false">ID · 18,973 QA</button><button data-filter="valid_full" aria-pressed="false">OOD · 17,578 QA</button></nav>
<div class="table-wrap"><table><caption>Aligned metrics · four decimal places</caption><thead><tr><th scope="col">Checkpoint</th><th scope="col">Split</th>'''+head+'''</tr></thead><tbody>'''+''.join(body)+'''</tbody></table></div>
<aside class="note"><strong>Read before comparing</strong><p>mMAE is lower-is-better; the other six metrics are higher-is-better. Planning samples are included in inference but have no dedicated metric here. These scores are not overall QA accuracy.</p><p>Stage 1 excludes 1,508 unavailable-media groups. Official and reproduced checkpoints use different processor pixel budgets. Not every metric improves; this is not a strict, difference-free paper reproduction.</p><p>Historical results do not prove the publication copy has passed GPU validation. See the repository's validation report for the actual check status.</p></aside>
<footer>Source: results/aligned_metrics.json · Offline, self-contained, no external assets.<br><a href="../docs/REPRODUCTION.md">Detailed reproduction record</a> · <a href="../docs/VALIDATION.md">Release validation</a></footer>
</main><script>
for(const button of document.querySelectorAll('button[data-filter]'))button.addEventListener('click',()=>{for(const b of document.querySelectorAll('button[data-filter]'))b.setAttribute('aria-pressed',String(b===button));for(const row of document.querySelectorAll('tr[data-split]'))row.hidden=button.dataset.filter!=='all'&&row.dataset.split!==button.dataset.filter;});
</script></body></html>'''
    dest=ROOT/'visualization/index.html';dest.write_text(page);print(dest)

if __name__=='__main__':render()
