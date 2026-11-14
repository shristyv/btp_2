"""Step 7: audit all locally cached BDD100K frame-level annotations.
No network calls, does not modify pilot data. Run from btp_part2 root.
"""
import csv, json, math, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / 'data/raw/labels'
OUT = ROOT / 'results/phase1_expanded'
MAN = ROOT / 'data/manifests/expanded'
CLASSES = ('bike','bus','car','motor','person','rider','traffic light','traffic sign','truck')
MIN_SIDE = 20


def atomic_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)


def find_origin(path):
    path_text=str(path).replace('\\','/')
    # Works for flattened cache path and nested path layouts.
    m=re.search(r'(?:labels[/_])?100k[/_](train|val)[/_](shard_\d{3})',path_text)
    if not m:
        m=re.search(r'labels_100k_(train|val)_(shard_\d{3})',path_text)
    return m.groups() if m else (None,None)


def env(weather, tod):
    w=str(weather or 'undefined').lower().strip()
    t=str(tod or 'undefined').lower().strip()
    if t=='night': return 'night'
    if w in ('rainy','snowy','foggy'): return 'adverse'
    if w in ('undefined','unknown','') or t in ('undefined','unknown',''): return 'unknown'
    if t=='dawn/dusk': return 'dawn_dusk'
    if t=='daytime' and w in ('clear','partly cloudy'): return 'day_clear'
    if w=='overcast': return 'overcast'
    return 'other'


def parse(path):
    split, shard=find_origin(path)
    if not split: return None
    data=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data,dict): return None
    att=data.get('attributes') or {}
    if not isinstance(att,dict): att={}
    name=str(data.get('name') or '').strip()
    if not name:
        name=path.stem.split('_')[-1]+'.jpg'
    if not name.lower().endswith('.jpg'): name += '.jpg'
    frames=data.get('frames') or []
    if not isinstance(frames,list) or not frames: return None
    objects=(frames[0] or {}).get('objects') or []
    if not isinstance(objects,list): return None
    boxes=[]
    for i,obj in enumerate(objects):
        if not isinstance(obj,dict): continue
        cls=str(obj.get('category','')).strip().lower()
        if cls not in CLASSES: continue
        b=obj.get('box2d') or {}
        try:
            x1,y1,x2,y2=[float(b[k]) for k in ('x1','y1','x2','y2')]
        except (KeyError,ValueError,TypeError): continue
        if not all(map(math.isfinite,(x1,y1,x2,y2))): continue
        if x2-x1<MIN_SIDE or y2-y1<MIN_SIDE: continue
        boxes.append({'box_index':i,'class':cls,'x':x1,'y':y1,'w':x2-x1,'h':y2-y1})
    weather=str(att.get('weather') or 'undefined').strip().lower()
    tod=str(att.get('timeofday') or 'undefined').strip().lower()
    scene=str(att.get('scene') or 'undefined').strip().lower()
    return {'source_split':split,'shard':shard,'source_image':name,
            'remote_path':f'data/images_100k/100k/{split}/{shard}/{name}',
            'weather':weather,'timeofday':tod,'scene':scene,
            'environment':env(weather,tod),'boxes':boxes}


def main():
    if not LABELS.exists(): raise SystemExit(f'No cached annotations found at {LABELS}.')
    OUT.mkdir(parents=True,exist_ok=True); MAN.mkdir(parents=True,exist_ok=True)
    files=sorted(LABELS.rglob('*.json'))
    unique={}; skipped=Counter()
    for p in files:
        try: r=parse(p)
        except (OSError,ValueError,TypeError,UnicodeError) as e:
            skipped['unreadable']+=1; continue
        if not r:
            skipped['unsupported_layout']+=1; continue
        key=(r['source_split'],r['shard'],r['source_image'])
        if key in unique: skipped['duplicate']+=1
        unique[key]=r
    rows=list(unique.values())
    if not rows: raise SystemExit('No annotations matched the BDD100K frame-level format. Check cache layout.')
    inventory=MAN/'annotation_inventory.jsonl'; temp=inventory.with_suffix('.jsonl.tmp')
    with temp.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    temp.replace(inventory)
    valid=[r for r in rows if r['boxes']]
    def count_rows(which):
        result=[]
        for split in ('train','val'):
            filtered=[r for r in valid if r['source_split']==split]
            counter=Counter()
            if which=='class':
                for r in filtered: counter.update(b['class'] for b in r['boxes'])
            else: counter.update(r[which] for r in filtered)
            result += [{'source_split':split,'value':k,'count':v} for k,v in sorted(counter.items())]
        return result
    for field in ('class','environment','weather','timeofday','scene'):
        atomic_csv(OUT/f'{field}_availability.csv',count_rows(field),['source_split','value','count'])
    stats={'cached_annotation_files':len(files),'parsed_unique_frames':len(rows),
           'usable_frames':len(valid),'valid_boxes':sum(len(r['boxes']) for r in valid),
           'usable_by_source_split':dict(Counter(r['source_split'] for r in valid)),
           'skipped':dict(skipped),'box_min_side_px':MIN_SIDE,
           'note':'Availability counts are from the cached sampled annotations, not the entire BDD100K dataset.'}
    (OUT/'audit_summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print('\n=== EXPANDED ANNOTATION AUDIT ===')
    for k,v in stats.items(): print(f'{k}: {v}')
    print('\nClass availability:')
    for row in count_rows('class'): print(f"  {row['source_split']:5} {row['value']:14} {row['count']}")
    print('\nInventory:',inventory)
    if len(valid)<500: print('WARNING: Annotation coverage is still small.')

if __name__=='__main__': main()
