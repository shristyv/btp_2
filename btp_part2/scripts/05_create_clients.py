"""Create IID and environment-skew client partitions; hide unlabeled class targets."""
import random
from collections import defaultdict, Counter
from pathlib import Path
from common import ROOT, config, read_csv, write_csv


def group_sources(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['source_image']].append(row)
    return groups


def assign_iid(groups, clients, rng):
    # Greedy source-group balancing; source frames never split across clients.
    frames=list(groups);rng.shuffle(frames)
    sizes={c:0 for c in clients};assigned={c:[] for c in clients}
    for src in frames:
        c=min(clients,key=lambda z:(sizes[z],rng.random()))
        assigned[c].extend(groups[src]);sizes[c]+=len(groups[src])
    return assigned


def assign_environment(groups, clients, weights, rng):
    # Greedy partition: prefer client's expected environment, while discouraging unequal client sizes.
    sources=list(groups);rng.shuffle(sources)
    sizes={c:0 for c in clients};assigned={c:[] for c in clients}
    counts={c:Counter() for c in clients}
    target=sum(len(v) for v in groups.values())/len(clients)
    sources.sort(key=lambda s:len(groups[s]),reverse=True)
    for src in sources:
        batch=groups[src]
        env=Counter(r['environment'] for r in batch).most_common(1)[0][0]
        n=len(batch)
        def score(c):
            current=counts[c][env]/max(1,sizes[c])
            # Score environment preference with an explicit soft cap on client size.
            return 2.0*weights[c].get(env,0) - 0.8*current - 1.8*max(0,(sizes[c]+n-target)/max(target,1)) + rng.random()*1e-5
        client=max(clients,key=score)
        assigned[client].extend(batch);sizes[client]+=n;counts[client][env]+=n
    return assigned


def select_global_labeled(rows, frac, rng):
    # Keep the SAME labeled sample IDs for IID and environment-skew partitions.
    by_class=defaultdict(list)
    for r in rows:by_class[r['class']].append(r)
    labeled=set()
    for cls, group in by_class.items():
        ids=[r['crop_id'] for r in group];rng.shuffle(ids)
        n=max(1,round(frac*len(ids)))
        labeled.update(ids[:n])
    return labeled


def mask_labels(rows, labeled):
    lab=[r for r in rows if r['crop_id'] in labeled]
    unlab=[{'crop_id':r['crop_id'],'crop_path':r['crop_path'],
            'source_image':r['source_image']} for r in rows if r['crop_id'] not in labeled]
    return lab,unlab


def main():
    cfg=config();rng=random.Random(cfg['seed']);train=read_csv(ROOT/'data'/'manifests'/'train.csv')
    if not train:raise ValueError('Training manifest is empty; run scripts 01–04')
    clients=[f'client_{i+1}' for i in range(cfg['num_clients'])]
    assert len(clients)==4,'This Phase 1 configuration defines exactly four client profiles.'
    groups=group_sources(train)
    global_labeled=select_global_labeled(train,float(cfg['labeled_fraction']),random.Random(cfg['seed']))
    for mode in ['iid','environment']:
        assigned=(assign_iid(groups,clients,rng) if mode=='iid' else
                  assign_environment(groups,clients,cfg['client_environment_weights'],rng))
        hidden=[]; summary=[]
        for client in clients:
            rows=assigned[client]
            lab,unlab=mask_labels(rows,global_labeled)
            loc=ROOT/'data'/'clients'/client/mode;loc.mkdir(parents=True,exist_ok=True)
            write_csv(loc/'labeled.csv',lab)
            write_csv(loc/'unlabeled.csv',unlab,['crop_id','crop_path','source_image'])
            # Hidden targets live outside the per-client model input directories.
            lab_ids={x['crop_id'] for x in lab}
            hidden.extend({'mode':mode,'client':client,'crop_id':r['crop_id'],'class':r['class']} for r in rows if r['crop_id'] not in lab_ids)
            print(f'{mode}/{client}: {len(lab)} labeled, {len(unlab)} unlabeled; environments {dict(Counter(r["environment"] for r in rows))}')
            summary.append({'mode':mode,'client':client,'total':len(rows),'labeled':len(lab),'unlabeled':len(unlab)})
        write_csv(ROOT/'data'/'manifests'/f'{mode}_hidden_targets_RESEARCH_ONLY.csv',hidden,['mode','client','crop_id','class'])
        write_csv(ROOT/'results'/'phase1'/f'{mode}_client_summary.csv',summary)
    print('Reminder: training code must NEVER read *_hidden_targets_RESEARCH_ONLY.csv')

if __name__=='__main__':main()
