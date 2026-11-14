"""Step 11: source-frame grouped, seeded IID/mild/strong FL client partitions.
Labeled budget is global, nested for 10% and 20%, same selected labels across strategies.
"""
import argparse,csv,json,random,math
from collections import Counter,defaultdict
from pathlib import Path

STRATEGIES=('iid','mild_non_iid','strong_non_iid')
ENV=('day_clear','night','adverse','dawn_dusk','overcast','unknown')
NCLIENT=4
SEED=20261114


def load(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def save(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def labeled_pool(rows,k,rng):
    """Globally stratified exact k labels; avoids losing rare classes entirely."""
    groups=defaultdict(list)
    for r in rows:groups[r['class']].append(r['crop_id'])
    for ids in groups.values():rng.shuffle(ids)
    classes=sorted(groups)
    # Start with 1 per class, then use proportional largest remainder to add remaining.
    if k<len(classes):raise ValueError('Label budget less than class count')
    target={c:1 for c in classes}; remaining=k-len(classes)
    for _ in range(remaining):
        c=max(classes,key=lambda x:(len(groups[x])*(k/sum(len(v) for v in groups.values()))-target[x], -classes.index(x)))
        if target[c]>=len(groups[c]):
            avail=[a for a in classes if target[a]<len(groups[a])]
            if not avail:break
            c=max(avail,key=lambda x:len(groups[x])/len(rows)*k-target[x])
        target[c]+=1
    by_class={c:groups[c][:target[c]] for c in classes}
    # Interleave class samples across nested 10/20% so the 10% slice covers all classes.
    picked=[];chosen_by_class=Counter()
    for c in classes:
        picked.append(by_class[c].pop(0));chosen_by_class[c]+=1
    while len(picked)<k:
        choices=[c for c in classes if by_class[c]]
        if not choices:break
        c=min(choices,key=lambda x:(chosen_by_class[x]/max(1,len(groups[x])),classes.index(x)))
        picked.append(by_class[c].pop(0));chosen_by_class[c]+=1
    return picked

def partition(rows,mode,seed):
    rng=random.Random(seed)
    frames=defaultdict(list)
    for r in rows:frames[(r['source_split'],r['source_image'])].append(r)
    groups=list(frames.values());rng.shuffle(groups);groups.sort(key=lambda g:-len(g))
    N=len(rows);size=[0]*NCLIENT; hist=[Counter() for _ in range(NCLIENT)]; classhist=[Counter() for _ in range(NCLIENT)]
    global_env=Counter(r['environment'] for r in rows); global_cls=Counter(r['class'] for r in rows)
    assigned={}; max_size=math.ceil(N/NCLIENT*1.20)
    # Clients have recognizable environment preferences, but *unknown* has none.
    affinity={
        'mild_non_iid':[
            {'day_clear':1.8,'dawn_dusk':1.4}, {'night':1.8},
            {'adverse':1.8,'overcast':1.3}, {'overcast':1.6,'day_clear':1.2}],
        'strong_non_iid':[
            {'day_clear':5,'dawn_dusk':3}, {'night':6},
            {'adverse':5,'overcast':3}, {'overcast':4,'dawn_dusk':2}]
    }
    for g in groups:
        ge=Counter(r['environment'] for r in g);gc=Counter(r['class'] for r in g); m=len(g)
        def score(j):
            projected=size[j]+m
            overload=max(0,projected-max_size)
            if mode=='iid':
                envdist=sum(abs((hist[j][e]+ge[e])-global_env[e]*projected/N) for e in global_env)/max(1,projected)
                clsdist=sum(abs((classhist[j][c]+gc[c])-global_cls[c]*projected/N) for c in global_cls)/max(1,projected)
                return 5.0*size[j]/(N/NCLIENT)+1.5*envdist+0.7*clsdist+50*overload/N
            pref=affinity[mode][j]
            preference=sum(ge[e]*math.log(pref.get(e,1.0)) for e in ge if e!='unknown')/m
            balance=5.0*size[j]/(N/NCLIENT)
            if mode=='mild_non_iid':preference*=0.6
            return balance-preference+50*overload/N
        j=min(range(NCLIENT),key=lambda x:(score(x),rng.random()))
        for r in g:assigned[r['crop_id']]=f'client_{j+1}'
        size[j]+=m;hist[j].update(ge);classhist[j].update(gc)
    assert len(assigned)==len(rows)
    assert all(size)
    return assigned

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--seed',type=int,default=SEED)
    a=p.parse_args();root=a.root.resolve()
    verified=root/'results/phase1_expanded/step10_dataset_validation.json'
    if not verified.exists() or not json.loads(verified.read_text()).get('pass'):raise SystemExit('Run Step 10 successfully first')
    rows=[r for r in load(root/'data/manifests/expanded/all_crops_v2.csv') if r['split']=='train']
    if len(rows)!=3200:raise SystemExit(f'Expected 3200 train examples, got {len(rows)}')
    rng=random.Random(a.seed)
    # Exact 640-label 20% pool, with a nested 320-label 10% pool.
    # First 320 may be class-stratified but does not enforce exact class quotas at 10%.
    picked=labeled_pool(rows,640,rng)
    label10=set(picked[:320]);label20=set(picked)
    if not label10<=label20:raise AssertionError('10% labels must be nested within 20% labels')
    out=root/'data/manifests/expanded/federated_v2';out.mkdir(parents=True,exist_ok=True)
    save(out/'label_budget_10.csv',[{'crop_id':x} for x in sorted(label10)],['crop_id'])
    save(out/'label_budget_20.csv',[{'crop_id':x} for x in sorted(label20)],['crop_id'])
    assignments={}; summary={'seed':a.seed,'label_budgets':{'10':len(label10),'20':len(label20)},'strategies':{}}
    columns=['crop_id','crop_path','source_image','source_split','environment','client_id','label_status']
    for mode in STRATEGIES:
        assignments[mode]=partition(rows,mode,a.seed+STRATEGIES.index(mode))
        summary['strategies'][mode]={}
        for percent,labels in (('10',label10),('20',label20)):
            tallies={}
            for client in range(1,NCLIENT+1):
                name=f'client_{client}'
                selected=[r for r in rows if assignments[mode][r['crop_id']]==name]
                yes=[{**r,'client_id':name,'label_status':'labeled'} for r in selected if r['crop_id'] in labels]
                no=[{**r,'client_id':name,'label_status':'unlabeled'} for r in selected if r['crop_id'] not in labels]
                base=out/mode/f'labels_{percent}'/name
                save(base/'labeled.csv',yes,list(rows[0])+['client_id','label_status'])
                save(base/'unlabeled.csv',no,columns)  # no class or annotation geometry
                tallies[name]={'total':len(selected),'labeled':len(yes),'unlabeled':len(no)}
            summary['strategies'][mode][percent]=tallies
        save(out/f'{mode}_assignments.csv',[{'crop_id':r['crop_id'],'client_id':assignments[mode][r['crop_id']]} for r in rows],['crop_id','client_id'])
    (root/'results/phase1_expanded/step11_partition_summary.json').write_text(json.dumps(summary,indent=2))
    for mode in STRATEGIES:print(mode,summary['strategies'][mode]['10'])
    print('Saved partitions:',out)
    print('WARNING: Training annotations remain in the master manifest for evaluation and auditing. The model training code must NEVER use labels of unlabeled rows.')

if __name__=='__main__':main()
