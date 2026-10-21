"""Render exportable architecture figures from the evidence-backed overview payload."""
from pathlib import Path
import json
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath

OUT=Path(__file__).resolve().parent
DATA=json.loads((OUT/'architecture_overview.json').read_text())
width,height=DATA['canvas']['width'],DATA['canvas']['height']
colors={'notebook':'#147d87','module':'#4467a6','config':'#8c609e','data':'#33826c','gate':'#a66c16','test':'#977126','tool':'#6c7484','output':'#6b8060'}
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
fig=plt.figure(figsize=(25,25*height/width+2.0),facecolor='#f6f8fa')
ax=fig.add_axes([.025,.06,.95,.88]);ax.set_xlim(0,width);ax.set_ylim(height,0);ax.axis('off')
for p in DATA['panels']:
    ax.add_patch(FancyBboxPatch((p['x'],p['y']),p['w'],p['h'],boxstyle='round,pad=0,rounding_size=12',facecolor='#edf3f6',edgecolor='#d5e1e7',zorder=0))
    ax.text(p['x']+20,p['y']+25,p['title'],fontsize=17,fontweight='bold',color='#203d48',va='center')
    ax.text(p['x']+20,p['y']+49,p['subtitle'],fontsize=10.5,color='#5d727c',va='center')
by_id={n['id']:n for n in DATA['nodes']}
for edge in DATA['edges']:
    # The static export shows a deliberately sparse set of local relationships.
    # The interactive map exposes every sourced workflow/code/data edge.
    if not edge.get('overview'):continue
    a,b=by_id[edge['source']],by_id[edge['target']]
    if abs(a['y']-b['y'])<8:
        right=b['x']>a['x'];x1=a['x']+a['w'] if right else a['x'];x2=b['x'] if right else b['x']+b['w'];y1=a['y']+a['h']/2;y2=b['y']+b['h']/2
        vertices=[(x1,y1),((x1+x2)/2,y1),((x1+x2)/2,y2),(x2,y2)]
    else:
        down=b['y']>a['y'];x1=a['x']+a['w']/2;x2=b['x']+b['w']/2;y1=a['y']+a['h'] if down else a['y'];y2=b['y'] if down else b['y']+b['h'];dy=max(25,abs(y2-y1)*.45)
        vertices=[(x1,y1),(x1,y1+(dy if down else -dy)),(x2,y2+(-dy if down else dy)),(x2,y2)]
    path=MplPath(vertices,[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4])
    ax.add_patch(PathPatch(path,facecolor='none',edgecolor='#748f9c',lw=1.2,zorder=1))
    prev=vertices[-2];end=vertices[-1]
    ax.annotate('',xy=end,xytext=(end[0]+(prev[0]-end[0])*.1,end[1]+(prev[1]-end[1])*.1),arrowprops=dict(arrowstyle='-|>',color='#748f9c',lw=1.2),zorder=1)
for n in DATA['nodes']:
    fill='#fff8ec' if n['kind']=='gate' else '#fff'
    ax.add_patch(FancyBboxPatch((n['x'],n['y']),n['w'],n['h'],boxstyle='round,pad=0,rounding_size=7',facecolor=fill,edgecolor='#b7cbd4',lw=.8,zorder=2))
    ax.plot([n['x']+3,n['x']+3],[n['y']+8,n['y']+n['h']-8],color=colors[n['kind']],lw=3,solid_capstyle='round',zorder=3)
    title=textwrap.wrap(n['title'],max(20,int((n['w']-28)/8)))[:2]
    y=n['y']+22
    for line in title:
        ax.text(n['x']+14,y,line,fontsize=12.3,fontweight='bold',color='#18313c',va='center',zorder=3);y+=17
    for line in textwrap.wrap(n['subtitle'],max(22,int((n['w']-28)/6.3)))[:2]:
        ax.text(n['x']+14,y+2,line,fontsize=9.6,color='#4b6572',va='center',zorder=3);y+=14
    ax.text(n['x']+14,n['y']+n['h']-10,f"{n['kind']} · {n['count']:,} {'file' if n['count']==1 else 'files'}",fontsize=8.5,color='#596f79',va='center',zorder=3)
fig.text(.037,.981,'French electricity demand forecasting',fontsize=28,fontweight='bold',color='#182e38',va='top')
fig.text(.037,.956,'Project architecture and saved research evidence',fontsize=15,color='#526975',va='top')
fig.text(.037,.043,f"{DATA['notebook_count']} notebooks · {DATA['study_count']} research protocols · {len(DATA['nodes'])} grouped components · {len(DATA['edges'])} evidenced relationships · source snapshot {DATA['generated_at'][:10]}",fontsize=11,color='#526975')
fig.text(.037,.027,'Static view shows selected local links. Open architecture.html for all relationship layers, source lines, notebook cells, pan/zoom and file search.',fontsize=10,color='#526975')
fig.text(.037,.011,'RTE D1 availability assumed; original vintages unverified; test previously explored. No production system or measured financial savings.',fontsize=10,color='#704500')
fig.savefig(OUT/'architecture.png',dpi=170,facecolor=fig.get_facecolor())
fig.savefig(OUT/'architecture.svg',facecolor=fig.get_facecolor())
plt.close(fig)
print('Wrote architecture.png and architecture.svg')
