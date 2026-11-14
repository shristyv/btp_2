"""Step 12: independently check client partitions and produce heterogeneity evidence."""
import argparse,csv,json,math
from pathlib import Path
from collections import Counter,defaultdict

CLASSES=['bike','bus','car','motor','person','rider','traffic light','traffic sign','truck']
ENVS=['day_clear','night','adverse','dawn_dusk','overcast','unknown']
MODES=['iid','mild_non_iid','strong_non_iid']

def read(p):
    with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write(p,records,fields):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
def jsd(a,b,domain):
    A=sum(a.values());B=sum(b.values());
    if not A or not B:return float('nan')
    ans=0.0
    for k in domain:
        p=a[k]/A;q=b[k]/B;m=(p+q)/2
        if p:ans+=0.5*p*math.log2(p/m)
        if q:ans+=0.5*q*math.log2(q/m)
    return ans

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=p.parse_args()
    root=args.root.resolve();base=root/'data/manifests/expanded/federated_v2';out=root/'results/phase1_expanded';out.mkdir(exist_ok=True,parents=True)
    master=read(root/'data/manifests/expanded/all_crops_v2.csv');train=[r for r in master if r['split']=='train'];byid={r['crop_id']:r for r in train}
    errors=[];warnings=[];metrics=[];distributions=[];label_sets={}
    global_c=Counter(r['class'] for r in train);global_e=Counter(r['environment'] for r in train)
    for pct,count in [('10',320),('20',640)]:
        ids=[r['crop_id'] for r in read(base/f'label_budget_{pct}.csv')]
        label_sets[pct]=set(ids)
        if len(ids)!=count or len(set(ids))!=count or not set(ids)<=set(byid):errors.append(f'{pct}% label set invalid')
    if not label_sets.get('10',set())<=label_sets.get('20',set()):errors.append('Label budgets not nested')
    for mode in MODES:
        ass=read(base/f'{mode}_assignments.csv')
        lookup={r['crop_id']:r['client_id'] for r in ass}
        if len(lookup)!=len(train) or len(ass)!=len(train) or set(lookup)!=set(byid):errors.append(f'{mode}: missing/duplicate assigned IDs')
        owners=defaultdict(set)
        for cid,name in lookup.items():
            if cid in byid:owners[(byid[cid]['source_split'],byid[cid]['source_image'])].add(name)
        if any(len(x)>1 for x in owners.values()):errors.append(f'{mode}: original frame shared across clients')
        if set(lookup.values())!={f'client_{i}' for i in range(1,5)}:errors.append(f'{mode}: missing/unexpected client')
        for j in range(1,5):
            name=f'client_{j}';subset=[r for r in train if lookup.get(r['crop_id'])==name]
            cc=Counter(r['class'] for r in subset);ee=Counter(r['environment'] for r in subset)
            metrics.append({'strategy':mode,'client':name,'n':len(subset),
                            'class_jsd':round(jsd(cc,global_c,CLASSES),6),
                            'environment_jsd':round(jsd(ee,global_e,ENVS),6),
                            'known_environment_jsd':round(jsd(Counter({k:v for k,v in ee.items() if k!='unknown'}),
                                 Counter({k:v for k,v in global_e.items() if k!='unknown'}),ENVS[:-1]),6)})
            for attr,domain,cnt in [('class',CLASSES,cc),('environment',ENVS,ee)]:
                for k in domain:distributions.append({'strategy':mode,'client':name,'field':attr,'category':k,'count':cnt[k]})
            for pct,expected in [('10',label_sets['10']),('20',label_sets['20'])]:
                loc=base/mode/f'labels_{pct}'/name
                labeled=read(loc/'labeled.csv');unlabeled=read(loc/'unlabeled.csv')
                lid=[r['crop_id'] for r in labeled];uid=[r['crop_id'] for r in unlabeled]
                all_ids=lid+uid
                if set(all_ids)!={r['crop_id'] for r in subset} or len(all_ids)!=len(subset):
                    errors.append(f'{mode}/{pct}/{name}: client membership mismatch')
                if set(lid)!={r['crop_id'] for r in subset if r['crop_id'] in expected}:
                    errors.append(f'{mode}/{pct}/{name}: labeled budget mismatch')
                for field in ['class','box_index','x','y','w','h']:
                    if unlabeled and field in unlabeled[0]:errors.append(f'{mode}/{pct}/{name}: leaked {field} in unlabeled CSV')
                for r in labeled:
                    if r['crop_id'] in byid and r['class']!=byid[r['crop_id']]['class']:
                        errors.append(f'{mode}/{pct}/{name}: modified label')
                if not labeled:warnings.append(f'{mode}/{pct}/{name}: no labeled examples')
    write(out/'step12_heterogeneity.csv',metrics,['strategy','client','n','class_jsd','environment_jsd','known_environment_jsd'])
    write(out/'step12_distributions.csv',distributions,['strategy','client','field','category','count'])
    stats={mode:{'mean_class_jsd':round(sum(m['class_jsd'] for m in metrics if m['strategy']==mode)/4,6),
                 'mean_environment_jsd':round(sum(m['environment_jsd'] for m in metrics if m['strategy']==mode)/4,6),
                 'mean_known_environment_jsd':round(sum(m['known_environment_jsd'] for m in metrics if m['strategy']==mode)/4,6)} for mode in MODES}
    # Modest to strong is expected, but report measured behavior without silently coercing.
    if not stats['iid']['mean_environment_jsd']<stats['strong_non_iid']['mean_environment_jsd']:
        warnings.append('Strong partition did not exhibit greater environment JSD than IID')
    if not stats['iid']['mean_environment_jsd']<stats['mild_non_iid']['mean_environment_jsd']<stats['strong_non_iid']['mean_environment_jsd']:
        warnings.append('IID < mild < strong environment JSD ordering not achieved; inspect data')
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        fig,ax=plt.subplots(figsize=(9,4.5))
        idx=np.arange(len(MODES));w=.27
        for i,(key,name) in enumerate([('mean_class_jsd','Class'),('mean_environment_jsd','Environment'),('mean_known_environment_jsd','Known environments')]):
            ax.bar(idx+(i-1)*w,[stats[s][key] for s in MODES],width=w,label=name)
        ax.set_xticks(idx,MODES);ax.set_ylabel('Mean Jensen-Shannon divergence (bits)');ax.set_title('Observed heterogeneity across four simulated vehicles');ax.legend();fig.tight_layout()
        fig.savefig(out/'step12_jsd_comparison.png',dpi=180);plt.close(fig)
    except ImportError:warnings.append('Matplotlib/numpy unavailable; chart skipped')
    report={'pass':not errors,'label_budgets':{'10':len(label_sets.get('10',[])),'20':len(label_sets.get('20',[]))},
            'strategy_metrics':stats,'errors':errors[:100],'warnings':warnings,
            'interpretation':'JSD measures partition differences, not predictive model accuracy. Stronger JSD is not proof of model improvement.'}
    (out/'step12_phase1_closeout.json').write_text(json.dumps(report,indent=2))
    print('STEP 12:', 'PASS' if not errors else 'FAIL');print(json.dumps(stats,indent=2))
    for x in errors:print('ERROR',x)
    for x in warnings:print('NOTE',x)
    if errors:raise SystemExit(1)
    print('Phase 1 expanded pipeline validated. This does NOT certify ML performance.')

if __name__=='__main__':main()
