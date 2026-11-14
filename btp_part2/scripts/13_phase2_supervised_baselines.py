"""Phase 2 first controlled baseline: frozen ImageNet-pretrained ResNet18 feature extractor;
centralized labeled-only linear head vs labeled-only 4-client FedAvg head.
Validation chooses checkpoint; test evaluated once at end. No pseudo-labels yet.
Requires torch torchvision pillow numpy pandas scikit-learn. Network may download ~45MB pretrained weights.
"""
import argparse,csv,json,random,copy,hashlib,sys
from pathlib import Path
from collections import Counter

CLASSES=['bike','bus','car','motor','person','rider','traffic light','traffic sign','truck']
MODES=['iid','mild_non_iid','strong_non_iid']

def read(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--strategy',choices=MODES,default='iid')
    p.add_argument('--labels',choices=['10','20'],default='10')
    p.add_argument('--rounds',type=int,default=2,help='SMOKE TEST default; research should use more rounds and multiple seeds')
    p.add_argument('--local-epochs',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--learning-rate',type=float,default=.03)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--rebuild-features',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    if a.rounds<1 or a.local_epochs<1:raise SystemExit('rounds/local-epochs must be positive')
    gate=root/'results/phase1_expanded/step12_phase1_closeout.json'
    if not gate.exists() or not json.loads(gate.read_text()).get('pass'):raise SystemExit('Phase 1 must PASS Step 12 before Phase 2')
    try:
        import numpy as np
        import torch
        from torch import nn
        from torch.utils.data import DataLoader,Dataset
        from torchvision import models
        from PIL import Image
        from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score,classification_report,confusion_matrix
    except ImportError as ex:raise SystemExit(f'Missing Phase2 dependency: {ex}\nRun: uv pip install torch torchvision numpy pillow scikit-learn')
    torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    rows=read(root/'data/manifests/expanded/all_crops_v2.csv')
    label_ids={r['crop_id'] for r in read(root/f'data/manifests/expanded/federated_v2/label_budget_{a.labels}.csv')}
    clients={r['crop_id']:r['client_id'] for r in read(root/f'data/manifests/expanded/federated_v2/{a.strategy}_assignments.csv')}
    train=[r for r in rows if r['split']=='train' and r['crop_id'] in label_ids]
    val=[r for r in rows if r['split']=='val'];test=[r for r in rows if r['split']=='test']
    if len(train)!=int(a.labels)*32:raise SystemExit('Wrong labeled train count')
    if len(val)!=600 or len(test)!=600:raise SystemExit('Wrong held-out split count')
    assert all(r['crop_id'] in clients for r in train)
    weights=models.ResNet18_Weights.IMAGENET1K_V1
    feature_model=models.resnet18(weights=weights)
    feature_model.fc=nn.Identity()
    feature_model.to(device).eval()
    preprocess=weights.transforms()
    class ImgData(Dataset):
        def __init__(self,items):self.items=items
        def __len__(self):return len(self.items)
        def __getitem__(self,i):
            r=self.items[i]
            with Image.open(root/r['crop_path']) as im: x=preprocess(im.convert('RGB'))
            return x,r['crop_id']
    # Cache frozen extracted features across seeds and strategies. Cache key ties to pretrained weights and manifest rows.
    cache_dir=root/'data/processed/frozen_features';cache_dir.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(('resnet18_imagenet1k_v1|'+''.join(sorted(r['crop_id']+r['crop_path'] for r in rows))).encode()).hexdigest()[:16]
    cache_file=cache_dir/f'resnet18_{signature}.pt'
    features={}
    if cache_file.exists() and not a.rebuild_features:
        payload=torch.load(cache_file,map_location='cpu',weights_only=True)
        features=payload['features']
        print('Loaded cached frozen features:',len(features))
    else:
        loader=DataLoader(ImgData(rows),batch_size=a.batch_size,shuffle=False,num_workers=0)
        with torch.no_grad():
            for i,(x,ids) in enumerate(loader,1):
                f=feature_model(x.to(device)).cpu().float()
                for j,cid in enumerate(ids):features[cid]=f[j]
                if i%20==0:print('feature batches:',i)
        torch.save({'features':features},cache_file)
        print('Saved frozen features',cache_file)
    feature_model.cpu();del feature_model
    ymap={c:i for i,c in enumerate(CLASSES)}
    def xy(items):
        x=torch.stack([features[r['crop_id']] for r in items]);y=torch.tensor([ymap[r['class']] for r in items],dtype=torch.long)
        return x,y
    X,Y=xy(train);VX,VY=xy(val);TX,TY=xy(test)
    def evaluate(head,x,y):
        with torch.no_grad():pred=head(x).argmax(1).numpy()
        return {'accuracy':float(accuracy_score(y.numpy(),pred)),
                'macro_f1':float(f1_score(y.numpy(),pred,labels=list(range(9)),average='macro',zero_division=0)),
                'balanced_accuracy':float(balanced_accuracy_score(y.numpy(),pred))}
    def new_head():return nn.Linear(512,len(CLASSES))
    torch.manual_seed(a.seed);init=new_head().state_dict()
    def train_head(head,x,y,epochs):
        head.train();opt=torch.optim.SGD(head.parameters(),lr=a.learning_rate)
        for _ in range(epochs):
            order=torch.randperm(len(y))
            for ix in order.split(a.batch_size):
                opt.zero_grad();loss=nn.functional.cross_entropy(head(x[ix]),y[ix]);loss.backward();opt.step()
        head.eval();return head
    # Both algorithms start at the same initialization. Centralized uses the same labeled samples.
    results={};log=[];out=root/'results/phase2';out.mkdir(exist_ok=True,parents=True)
    for algorithm in ('centralized','fedavg'):
        head=new_head();head.load_state_dict(copy.deepcopy(init));best=None;best_state=None;best_round=None
        for rnd in range(1,a.rounds+1):
            if algorithm=='centralized':
                head=train_head(head,X,Y,a.local_epochs)
            else:
                local_states=[];local_n=[]
                for client in [f'client_{i}' for i in range(1,5)]:
                    indices=[j for j,r in enumerate(train) if clients[r['crop_id']]==client]
                    if not indices:continue
                    local=new_head();local.load_state_dict(copy.deepcopy(head.state_dict()))
                    local=train_head(local,X[indices],Y[indices],a.local_epochs)
                    local_states.append({k:v.detach().clone() for k,v in local.state_dict().items()});local_n.append(len(indices))
                if not local_states:raise RuntimeError('No labeled client data')
                merged={k:sum(state[k]*(n/sum(local_n)) for state,n in zip(local_states,local_n)) for k in local_states[0]}
                head.load_state_dict(merged)
            v=evaluate(head,VX,VY)
            log.append({'algorithm':algorithm,'round':rnd,**v})
            if best is None or v['macro_f1']>best+1e-12:
                best=v['macro_f1'];best_state=copy.deepcopy(head.state_dict());best_round=rnd
            print(algorithm,'round',rnd,'val macro_F1',round(v['macro_f1'],4))
        head.load_state_dict(best_state)
        test_metrics=evaluate(head,TX,TY)  # test only after val-based checkpoint selection
        results[algorithm]={'chosen_round':best_round,'best_val_macro_f1':best,
                            'test_metrics':test_metrics}
        torch.save({'state_dict':best_state,'classes':CLASSES,'config':vars(a)},out/f'{algorithm}_{a.strategy}_labels{a.labels}_seed{a.seed}.pt')
    with (out/f'round_log_{a.strategy}_labels{a.labels}_seed{a.seed}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['algorithm','round','accuracy','macro_f1','balanced_accuracy']);w.writeheader();w.writerows(log)
    report={'setup':'frozen pretrained ResNet18, trainable 9-class linear head, supervised labeled-only baseline',
            'status':'SMOKE TEST ONLY' if a.rounds<=2 else 'experimental run (requires repeated seeds and configs)',
            'strategy':a.strategy,'labeled_percent':a.labels,'seed':a.seed,'rounds':a.rounds,
            'local_epochs':a.local_epochs,'learning_rate':a.learning_rate,
            'training_samples':len(train),'validation_samples':len(val),'test_samples':len(test),
            'results':results,'caveats':['Class-balanced sampling means test prevalence is not natural BDD100K prevalence.',
            'ImageNet pretraining uses external labeled data; disclose this transfer-learning assumption.',
            'Centralized and FedAvg have different SGD batch schedules; match data and local epochs, not exact optimizer trajectories.',
            'No semi-supervised training or unlabeled data access occurs in this baseline.']}
    fname=out/f'baseline_{a.strategy}_labels{a.labels}_seed{a.seed}.json'
    fname.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
