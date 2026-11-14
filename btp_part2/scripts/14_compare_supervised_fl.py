"""Step 14: reproducible supervised FedAvg, FedProx and FedAdam comparison.
Uses previously extracted ResNet18 frozen embeddings. Never reads labels of unlabeled rows.
Selects checkpoints on validation only; test requires explicit --final-test.
"""
import argparse, csv, copy, hashlib, json, random, time
from pathlib import Path

CLASSES = ['bike','bus','car','motor','person','rider','traffic light','traffic sign','truck']
METHODS = ('fedavg','fedprox','fedadam')
STRATEGIES = ('iid','mild_non_iid','strong_non_iid')

def csvrows(p):
    with p.open(newline='', encoding='utf-8') as f: return list(csv.DictReader(f))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--strategy', choices=STRATEGIES, default='strong_non_iid')
    ap.add_argument('--labels', choices=['10','20'], default='10')
    ap.add_argument('--rounds', type=int, default=20)
    ap.add_argument('--local-epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.03, help='Client SGD learning rate')
    ap.add_argument('--mu', type=float, default=0.01, help='FedProx proximal strength')
    ap.add_argument('--server-lr', type=float, default=0.01, help='FedAdam server learning rate')
    ap.add_argument('--beta1', type=float, default=0.9)
    ap.add_argument('--beta2', type=float, default=0.99)
    ap.add_argument('--tau', type=float, default=1e-3, help='FedAdam denominator stabilizer')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--final-test', action='store_true', help='Only after locking experiment selection')
    a=ap.parse_args(); root=a.root.resolve()
    if a.rounds<1 or a.local_epochs<1 or a.batch_size<1 or a.lr<=0 or a.server_lr<=0 or a.mu<0 or a.tau<=0:
        ap.error('Invalid optimizer or training settings')
    gate=root/'results/phase1_expanded/step12_phase1_closeout.json'
    if not gate.exists() or not json.loads(gate.read_text()).get('pass'):
        raise SystemExit('Phase 1 Step 12 validation must pass before training')
    import numpy as np
    import torch
    from torch import nn
    from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score
    torch.set_num_threads(min(4,torch.get_num_threads()))
    rows=csvrows(root/'data/manifests/expanded/all_crops_v2.csv')
    ids={r['crop_id'] for r in csvrows(root/f'data/manifests/expanded/federated_v2/label_budget_{a.labels}.csv')}
    clientmap={r['crop_id']:r['client_id'] for r in csvrows(root/f'data/manifests/expanded/federated_v2/{a.strategy}_assignments.csv')}
    train=[r for r in rows if r['split']=='train' and r['crop_id'] in ids]
    val=[r for r in rows if r['split']=='val']
    test=[r for r in rows if r['split']=='test']
    if len(train)!=int(a.labels)*32 or len(val)!=600 or len(test)!=600:
        raise SystemExit('Unexpected split sizes or labeled pool')
    if any(r['crop_id'] not in clientmap for r in train):raise SystemExit('Missing client assignment')
    signature=hashlib.sha256(('resnet18_imagenet1k_v1|'+''.join(sorted(r['crop_id']+r['crop_path'] for r in rows))).encode()).hexdigest()[:16]
    cache=root/f'data/processed/frozen_features/resnet18_{signature}.pt'
    if not cache.exists():raise SystemExit('Frozen feature cache absent. Run existing Step 13 first to create it.')
    features=torch.load(cache,map_location='cpu',weights_only=True)['features']
    print(f'Loaded {len(features)} cached ResNet18 embeddings from {cache.name}')
    ymap={c:i for i,c in enumerate(CLASSES)}
    def xy(items):
        return (torch.stack([features[r['crop_id']].float() for r in items]),
                torch.tensor([ymap[r['class']] for r in items],dtype=torch.long))
    X,Y=xy(train);VX,VY=xy(val)
    if a.final_test:TX,TY=xy(test)
    byclient={c:torch.tensor([j for j,r in enumerate(train) if clientmap[r['crop_id']]==c],dtype=torch.long)
              for c in (f'client_{i}' for i in range(1,5))}
    if any(len(v)==0 for v in byclient.values()):raise SystemExit('Client has zero labeled samples')
    print('Client labeled counts:',{k:len(v) for k,v in byclient.items()})
    def metrics(model,x,y):
        model.eval()
        with torch.no_grad():pred=model(x).argmax(1).numpy()
        true=y.numpy()
        return dict(accuracy=float(accuracy_score(true,pred)),macro_f1=float(f1_score(true,pred,labels=list(range(9)),average='macro',zero_division=0)),balanced_accuracy=float(balanced_accuracy_score(true,pred)))
    def head():return nn.Linear(512,9)
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    initial=copy.deepcopy(head().state_dict())
    # Independent seed per algorithm/round/client for matched local minibatch order.
    def local_train(global_state,indices,method,rnd,client_idx):
        model=head();model.load_state_dict(copy.deepcopy(global_state));model.train()
        frozen={k:v.detach().clone() for k,v in global_state.items()}
        opt=torch.optim.SGD(model.parameters(),lr=a.lr)
        g=torch.Generator().manual_seed(a.seed*1000003+rnd*101+client_idx)
        xx=X[indices];yy=Y[indices]
        for _ in range(a.local_epochs):
            for b in torch.randperm(len(yy),generator=g).split(a.batch_size):
                opt.zero_grad()
                loss=nn.functional.cross_entropy(model(xx[b]),yy[b])
                if method=='fedprox':
                    proximal=sum((p-frozen[name]).square().sum() for name,p in model.named_parameters())
                    loss=loss+(a.mu/2)*proximal
                loss.backward();opt.step()
        return {k:v.detach().clone() for k,v in model.state_dict().items()}
    out=root/'results/phase2_supervised_v2'/a.strategy/f'labels_{a.labels}'/f'seed_{a.seed}'
    out.mkdir(parents=True,exist_ok=True)
    log=[];summary={};start=time.perf_counter()
    for method in METHODS:
        global_state=copy.deepcopy(initial);m={k:torch.zeros_like(v) for k,v in initial.items()};v={k:torch.zeros_like(v) for k,v in initial.items()}
        best=-1;best_round=None;best_state=None
        for rnd in range(1,a.rounds+1):
            local=[];sizes=[]
            for ci,(name,indices) in enumerate(byclient.items()):
                local.append(local_train(global_state,indices,method,rnd,ci))
                sizes.append(len(indices))
            denom=sum(sizes)
            avg={k:sum((state[k]*(n/denom) for state,n in zip(local,sizes)),torch.zeros_like(global_state[k])) for k in global_state}
            if method in ('fedavg','fedprox'):
                global_state=avg
            else:
                # FedAdam pseudo-gradient = weighted average local delta (global - local).
                for k in global_state:
                    delta=global_state[k]-avg[k]
                    m[k]=a.beta1*m[k]+(1-a.beta1)*delta
                    v[k]=a.beta2*v[k]+(1-a.beta2)*delta.square()
                    mhat=m[k]/(1-a.beta1**rnd);vhat=v[k]/(1-a.beta2**rnd)
                    global_state[k]=global_state[k]-a.server_lr*mhat/(vhat.sqrt()+a.tau)
            model=head();model.load_state_dict(global_state)
            metr=metrics(model,VX,VY)
            log.append({'method':method,'round':rnd,**metr})
            if metr['macro_f1']>best+1e-12:
                best=metr['macro_f1'];best_round=rnd;best_state=copy.deepcopy(global_state)
            print(f'{method:7s} round {rnd:02d}/{a.rounds} val macro-F1 {metr["macro_f1"]:.4f} val accuracy {metr["accuracy"]:.4f}',flush=True)
        checkpoint={'state_dict':best_state,'classes':CLASSES,'method':method,'strategy':a.strategy,'labels':a.labels,'seed':a.seed,'selected_round':best_round,'val_macro_f1':best}
        torch.save(checkpoint,out/f'{method}_best.pt')
        record={'best_val_macro_f1':best,'selected_round':best_round}
        if a.final_test:
            model=head();model.load_state_dict(best_state)
            record['final_test_metrics']=metrics(model,TX,TY)
        summary[method]=record
    with (out/'round_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['method','round','accuracy','macro_f1','balanced_accuracy'])
        writer.writeheader();writer.writerows(log)
    report={'status':'confirmatory test run' if a.final_test else 'validation-only development run',
            'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
            'training_labeled':len(train),'validation':len(val),'test_count':len(test),
            'time_seconds':round(time.perf_counter()-start,2),'models':summary,
            'scope':'ImageNet-pretrained frozen ResNet18 embedding + linear 9-class head, labeled-only, four simulated clients',
            'important':'Do not compare test metrics or tune from test. FedAdam has a separate server learning rate. Client data counts drive FedAvg weights.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print('Saved to',out)
if __name__=='__main__':main()
