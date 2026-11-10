"""Audit leakage, label hiding, class coverage, environment skew and save figures."""
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import ROOT, config, read_csv, CLASSES


def js_divergence(p,q):
    p=np.asarray(p,dtype=float);q=np.asarray(q,dtype=float)
    if p.sum()==0 or q.sum()==0:return float('nan')
    p/=p.sum();q/=q.sum();m=(p+q)/2
    def kl(a,b):
        idx=a>0
        return float(np.sum(a[idx]*np.log2(a[idx]/b[idx])))
    return 0.5*(kl(p,m)+kl(q,m))


def main():
    cfg=config();out=ROOT/'results'/'phase1';out.mkdir(parents=True,exist_ok=True)
    splits={s:read_csv(ROOT/'data'/'manifests'/f'{s}.csv') for s in ['train','val','test']}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert {r['source_image'] for r in splits[a]}.isdisjoint({r['source_image'] for r in splits[b]})
    for s,group in splits.items():
        assert len({r['crop_id'] for r in group})==len(group),f'duplicate crop IDs {s}'
        assert all((ROOT/r['crop_path']).is_file() for r in group),f'missing crops {s}'
    clients=[f'client_{i+1}' for i in range(cfg['num_clients'])]
    expected={r['crop_id'] for r in splits['train']}
    report={'source_overlap':'none by source filename','splits':{s:len(r) for s,r in splits.items()},'modes':{}}
    global_env=Counter(r['environment'] for r in splits['train'])
    envs=sorted(global_env)
    div_rows=[]
    for mode in ['iid','environment']:
        all_ids=[]; all_source=[]; heat=[]; class_heat=[];rows_per_client=[]
        for client in clients:
            loc=ROOT/'data'/'clients'/client/mode
            labeled=read_csv(loc/'labeled.csv');unlabeled=read_csv(loc/'unlabeled.csv')
            assert 'class' not in unlabeled[0] if unlabeled else True,'Unlabeled CSV exposes label'
            ids=[r['crop_id'] for r in labeled+unlabeled]
            assert len(ids)==len(set(ids)), f'duplicate within {client}/{mode}'
            all_ids.extend(ids)
            sources={r['source_image'] for r in labeled+unlabeled}
            all_source.extend(sources)
            source_lookup={r['crop_id']:r for r in splits['train']}
            actual=[source_lookup[cid] for cid in ids]
            env_count=Counter(r['environment'] for r in actual)
            class_count=Counter(r['class'] for r in actual)
            heat.append([env_count[e] for e in envs]);class_heat.append([class_count[c] for c in CLASSES])
            js=js_divergence([env_count[e] for e in envs],[global_env[e] for e in envs])
            div_rows.append({'mode':mode,'client':client,'js_environment_bits':js})
            rows_per_client.append({'client':client,'total':len(ids),'labeled':len(labeled),'unlabeled':len(unlabeled),
                                    'js_environment_bits':round(js,4), 'labeled_classes':len(set(r['class'] for r in labeled))})
        assert len(all_ids)==len(set(all_ids)),f'samples duplicated across clients for {mode}'
        assert set(all_ids)==expected,f'client assignments do not cover training set: {mode}'
        assert len(all_source)==len(set(all_source)),f'source image assigned to multiple clients: {mode}'
        report['modes'][mode]=rows_per_client
        for title,cols,array in [('environment',envs,heat),('class',CLASSES,class_heat)]:
            a=np.asarray(array,dtype=float)
            fig,ax=plt.subplots(figsize=(max(7,len(cols)),4))
            im=ax.imshow(a/(a.sum(axis=1,keepdims=True)+1e-12),cmap='Blues',vmin=0,vmax=1)
            ax.set_xticks(range(len(cols)),cols,rotation=35,ha='right')
            ax.set_yticks(range(len(clients)),clients)
            ax.set_title(f'{mode.upper()} client {title} distributions (fraction)')
            fig.colorbar(im,ax=ax,label='Within-client fraction')
            fig.tight_layout();fig.savefig(out/f'{mode}_{title}_heatmap.png',dpi=180);plt.close(fig)
    pd.DataFrame(div_rows).to_csv(out/'non_iid_divergence.csv',index=False)
    (out/'split_validation.json').write_text(json.dumps(report,indent=2))
    print('PASS: Source-image disjoint splits, dataset files and client partitions validated.')
    print(pd.DataFrame(div_rows).to_string(index=False))
    print('Plots + validation report:',out)
    print('LIMITATION: Source-video-level grouping and camera-near-duplicates require separate verification.')

if __name__=='__main__':main()
