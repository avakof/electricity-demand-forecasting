// Offline DOM tests only: no network, model fits or project notebook execution.
const fs=require('fs'),assert=require('node:assert/strict');
const {JSDOM,VirtualConsole}=require('../.runtime/graph-ui-check/node_modules/jsdom');
const data=JSON.parse(fs.readFileSync(__dirname+'/architecture_overview.json','utf8'));
const errors=[],vc=new VirtualConsole();vc.on('jsdomError',e=>errors.push(e.message));
const dom=new JSDOM(fs.readFileSync(__dirname+'/architecture.html','utf8'),{runScripts:'dangerously',virtualConsole:vc});
const doc=dom.window.document,win=dom.window,svg=doc.querySelector('#canvas');
assert.equal(doc.querySelectorAll('g.node').length,data.nodes.length);
for(const mode of ['workflow','code','data','all']){
 doc.querySelector(`[data-mode="${mode}"]`).click();
 assert.equal(doc.querySelectorAll('.edge').length,data.edges.filter(e=>mode==='all'||e.layer===mode).length);
}
doc.querySelector('[data-node="feature_views"]').dispatchEvent(new win.MouseEvent('click',{bubbles:true}));
assert.equal(doc.querySelector('#details h2').textContent,'Three causal predictor views');
assert.ok(doc.querySelector('#details').textContent.includes('src/rte_feature_lab.py'));
assert.ok(doc.querySelectorAll('.edge.highlight').length>0);
const input=doc.querySelector('#search');input.value='rte_learned_features.py';input.dispatchEvent(new win.Event('input',{bubbles:true}));
assert.ok(doc.querySelector('#search-results').textContent.includes('src/rte_learned_features.py'));
assert.ok(doc.querySelector('#search-results').textContent.includes('tests/test_rte_learned_features.py'));
doc.querySelector('[data-node="contract"]').dispatchEvent(new win.KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
assert.equal(doc.querySelector('#details h2').textContent,'Forecast contract');
svg.dispatchEvent(new win.KeyboardEvent('keydown',{key:'Escape',bubbles:true}));
assert.equal(doc.querySelector('#details h2').textContent,'Trace the evidence');
const before=svg.getAttribute('viewBox');doc.querySelector('#zoom-in').click();assert.notEqual(svg.getAttribute('viewBox'),before);
doc.querySelector('#fit').click();assert.equal(svg.getAttribute('viewBox'),before);
// Explicit dimensions for deterministic DOM simulation; browser layout is tested separately.
svg.getBoundingClientRect=()=>({width:1000,height:800});
const jump=doc.querySelector('#panel-jump');jump.value='risk';jump.dispatchEvent(new win.Event('change',{bubbles:true}));
assert.notEqual(svg.getAttribute('viewBox'),before);
doc.querySelector('#reset').click();assert.equal(doc.querySelector('#details h2').textContent,'Trace the evidence');
assert.deepEqual(errors,[]);
fs.writeFileSync(__dirname+'/interaction_validation.json',JSON.stringify({method:'JSDOM simulation; no browser or network',rendered_components:data.nodes.length,all_links:data.edges.length,mode_switches:true,node_selection:true,keyboard_selection:true,reset:true,file_search:true,zoom_and_fit:true,panel_focus:true,javascript_errors:errors},null,2)+'\n');
console.log(`DOM verification passed: ${data.nodes.length} components, ${data.edges.length} relationships, all layers, selection, search, keyboard, zoom/reset and panel focus.`);dom.window.close();
