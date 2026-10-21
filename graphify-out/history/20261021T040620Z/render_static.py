"""Render source-derived architecture with Matplotlib, without browser automation."""
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MPath

OUT=Path(__file__).resolve().parent
d=json.loads((OUT/'architecture_overview.json').read_text())
colors={'notebook':'#385e85','module':'#137d68','config':'#88694c','data':'#6c7ca3','gate':'#b16a16','test':'#7b668d','tool':'#71838b','output':'#6c7ca3'}
fig=plt.figure(figsize=(25,15),facecolor='#f4f6f4')
ax=fig.add_axes((.025,.105,.95,.77));ax.set_xlim(0,1516);ax.set_ylim(798,0);ax.axis('off')
byid={n['id']:n for n in d['nodes']}
for e in d['edges']:
    primary={('n1','data'),('n2','eda'),('n3','features'),('n4','backtest'),('n5','uncertainty'),('n6','pipeline')}
    if e['layer']!='workflow' and (e['source'],e['target']) not in primary:continue
    a,b=byid[e['source']],byid[e['target']]
    if a['y']==b['y']:
        start=(a['x']+a['w']+2,a['y']+a['h']/2);end=(b['x']-2,b['y']+b['h']/2)
        patch=FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,lw=1.6,color='#6a8587',zorder=1)
    elif abs(a['x']-b['x'])<1 and b['y']-a['y']>250:
        right=a['x']+a['w'];mid=right+11
        verts=[(right,a['y']+a['h']/2),(mid,a['y']+a['h']/2),(mid,b['y']+b['h']/2),(right,b['y']+b['h']/2)]
        path=MPath(verts,[MPath.MOVETO,MPath.LINETO,MPath.LINETO,MPath.LINETO])
        patch=FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=12,lw=1.4,color='#91a4a2',zorder=1)
    else:
        start=(a['x']+a['w']/2,a['y']+a['h']);end=(b['x']+b['w']/2,b['y'])
        if b['y']<a['y']:start=(a['x']+a['w']/2,a['y']);end=(b['x']+b['w']/2,b['y']+b['h'])
        patch=FancyArrowPatch(start,end,connectionstyle='arc3,rad=0.08',arrowstyle='-|>',mutation_scale=12,lw=1.3,color='#91a4a2',zorder=1)
    ax.add_patch(patch)
for y,label in [(38,'RESEARCH SPECIFICATION / INPUTS / EVIDENCE'),(193,'NOTEBOOKS / CHRONOLOGICAL WORKFLOW'),(348,'SHARED PYTHON IMPLEMENTATION'),(649,'SAVED ARTIFACTS / MANAGER DELIVERABLES')]:
    ax.text(44,y,label,fontsize=13,fontweight='bold',color='#61777d')
for n in d['nodes']:
    x,y=n['x'],n['y'];c=colors[n['kind']]
    box=FancyBboxPatch((x,y),n['w'],n['h'],boxstyle='round,pad=0,rounding_size=7',linewidth=1,edgecolor='#d4dfdd',facecolor='#fff6e7' if n['kind']=='gate' else 'white',zorder=3)
    ax.add_patch(box)
    ax.plot([x+1,x+1],[y+15,y+n['h']-15],color=c,lw=3,zorder=4,solid_capstyle='round')
    ax.text(x+13,y+24,n['title'],fontsize=13.5,fontweight='bold',color='#142b37',zorder=4)
    ax.text(x+13,y+44,'\n'.join(textwrap.wrap(n['subtitle'],30)),fontsize=12.5,color='#61737d',va='top',linespacing=1.35,zorder=4)
    ax.text(x+13,y+87,f"{n['kind'].upper()}  /  {n['count']} {'FILE' if n['count']==1 else 'FILES'}",fontsize=10.5,color=c,zorder=4)
fig.text(.052,.959,'PROJECT ARCHITECTURE / SOURCE SNAPSHOT 20 OCT 2026',fontsize=13,fontweight='bold',color='#137d68')
fig.text(.052,.921,'French electricity demand forecasting',fontsize=30,fontweight='bold',color='#142b37')
fig.text(.052,.889,'Six notebook stages. Shared origin replay. Archived predictions. Explicit evidence limits.',fontsize=16,color='#61737d')
fig.text(.052,.083,'Operational gate CLOSED: authentic vintage archives are empty; interval lineage is unverified. Current comparisons are retrospective.',fontsize=14,color='#865c22')
fig.text(.052,.052,'29 components / 199 files inventoried. Arrows show selected workflow links; use architecture.html for code/data views and source evidence.',fontsize=13,color='#61737d')
fig.text(.052,.028,'Full Graphify network: 379 nodes / 877 edges. Source-level architecture, not a runtime execution trace.',fontsize=12,color='#61737d')
fig.savefig(OUT/'architecture.png',dpi=170,facecolor=fig.get_facecolor())
fig.savefig(OUT/'architecture.svg',facecolor=fig.get_facecolor())
plt.close(fig)
print('Saved architecture.png and architecture.svg')
