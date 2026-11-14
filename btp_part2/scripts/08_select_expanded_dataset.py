"""Step 8: create an independent final-v2 plan from audited annotations.
No JPEG downloads. Does not replace the Phase 1 pilot selection.
"""
import csv, json, random
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/'data/manifests/expanded/annotation_inventory.jsonl'
OUT=ROOT/'data/manifests/expanded'
RES=ROOT/'results/phase1_expanded'
SEED=42
TARGET={'train':3200,'val':600,'test':600}
MAX_PER_IMAGE=6
CLASSES=('bike','bus','car','motor','person','rider','traffic light','traffic sign','truck')
# Use soft class targets, not invented promises of availability.
# Aim for fairer representation, without excluding cars from the natural test set completely.
CLASS_WEIGHT={'bike':3.2,'bus':2.5,'car':0.45,'motor':4.0,'person':1.2,
              'rider':4.0,'traffic light':1.2,'traffic sign':1.0,'truck':2.0}


def choose(pool,target,rng,reserved):
    frames=[r for r in pool if r['boxes'] and (r['source_split'],r['source_image']) not in reserved]
    rng.shuffle(frames)
    chosen=[]; cls=Counter(); env=Counter(); num=0
    # Targets gently encourage class balance; allow actual availability to govern selection.
    quota={c:max(1,int(target/len(CLASSES))) for c in CLASSES}
    env_quota={e:max(1,int(target/6)) for e in ('day_clear','night','adverse','dawn_dusk','overcast','unknown')}
    while frames and num<target:
        subset=frames if len(frames)<350 else rng.sample(frames,350)
        def value(row):
            counts=Counter(b['class'] for b in row['boxes'])
            rarity=sum(CLASS_WEIGHT.get(c,1)*min(2,n)*max(0.2, 1-cls[c]/max(1,quota[c]))
                       for c,n in counts.items())
            e=row['environment']; balance=2.0*max(0.1,1-env[e]/max(1,env_quota.get(e,target/6)))
            return rarity+balance+0.00001*rng.random()
        picked=max(subset,key=value)
        frames.remove(picked)
        # Important: index is the position in the original objects list, not a newly assigned index.
        selected_boxes=sorted(picked['boxes'], key=lambda b:(
            cls[b['class']]/max(1,quota[b['class']]),
            -CLASS_WEIGHT.get(b['class'],1),rng.random()))
        amount=min(MAX_PER_IMAGE, len(selected_boxes),target-num)
        if amount<=0: continue
        row={**picked,'boxes':selected_boxes[:amount],'split':None,'expected_crops':amount}
        chosen.append(row)
        cls.update(b['class'] for b in row['boxes'])
        env[row['environment']]+=amount
        num+=amount
    return chosen,cls,env,num


def main():
    if not IN.exists(): raise SystemExit('Run 07_audit_expanded_annotations.py first.')
    rows=[json.loads(s) for s in IN.read_text(encoding='utf-8').splitlines() if s.strip()]
    rng=random.Random(SEED)
    source_train=[r for r in rows if r['source_split']=='train']
    source_val=[r for r in rows if r['source_split']=='val']
    if not source_train or not source_val:
        raise SystemExit('Need cached source-train AND source-validation annotations.')
    selected=[]; reserved=set(); summary=[]
    # Select validation first to avoid the training set consuming its rarer candidates.
    for split,pool in (('val',source_train),('test',source_val),('train',source_train)):
        chosen,cls,env,n=choose(pool,TARGET[split],rng,reserved)
        for r in chosen:
            r['split']=split
            reserved.add((r['source_split'],r['source_image']))
        selected.extend(chosen)
        summary.append({'split':split,'target_crops':TARGET[split],
                        'selected_crops':n,'selected_frames':len(chosen),
                        'class_counts':dict(cls),'environment_counts':dict(env)})
        print(f'{split}: {n}/{TARGET[split]} crops from {len(chosen)} original frames')
        print('  classes:',dict(cls)); print('  environments:',dict(env))
        if n<TARGET[split]: print(f'WARNING: {split} target not reached; need more diverse cached annotations.')
    if len(reserved)!=len(selected): raise AssertionError('A source frame appeared in multiple splits.')
    for r in selected:
        if (r['split']=='test') != (r['source_split']=='val'):
            raise AssertionError('Original BDD train/validation boundary violated.')
    OUT.mkdir(parents=True,exist_ok=True); RES.mkdir(parents=True,exist_ok=True)
    path=OUT/'selected_frames_v2.jsonl'; tmp=path.with_suffix('.jsonl.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        for r in selected: f.write(json.dumps(r)+'\n')
    tmp.replace(path)
    (RES/'selection_summary_v2.json').write_text(json.dumps({'seed':SEED,'target':TARGET,
        'max_per_image':MAX_PER_IMAGE,'splits':summary},indent=2),encoding='utf-8')
    print('\nNo JPEGs downloaded. New selection:',path)
    if any(x['selected_crops']<x['target_crops'] for x in summary):
        print('Targets are not fully met. Inspect summary before accepting final dataset size.')

if __name__=='__main__': main()
