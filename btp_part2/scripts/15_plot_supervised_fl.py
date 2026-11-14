"""Step 15: plot measured validation convergence; no test access."""
import argparse,csv,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--strategy',choices=['iid','mild_non_iid','strong_non_iid'],default='strong_non_iid')
    p.add_argument('--labels',choices=['10','20'],default='10')
    p.add_argument('--seeds',type=int,nargs='+',default=[42])
    a=p.parse_args();import matplotlib.pyplot as plt
    root=a.root.resolve();base=root/'results/phase2_supervised_v2'/a.strategy/f'labels_{a.labels}'
    allrows=[]
    for seed in a.seeds:
        fn=base/f'seed_{seed}'/'round_metrics.csv'
        if not fn.exists():raise SystemExit(f'Missing {fn}; run Step 14 first')
        with fn.open(newline='',encoding='utf-8') as f:
            for row in csv.DictReader(f):allrows.append({**row,'seed':seed,'round':int(row['round']),'macro_f1':float(row['macro_f1'])})
    methods=['fedavg','fedprox','fedadam']
    fig,ax=plt.subplots(figsize=(9,5))
    for method in methods:
        rs=sorted(set(r['round'] for r in allrows if r['method']==method))
        means=[];mins=[];maxs=[]
        for rd in rs:
            vals=[r['macro_f1'] for r in allrows if r['method']==method and r['round']==rd]
            means.append(sum(vals)/len(vals));mins.append(min(vals));maxs.append(max(vals))
        ax.plot(rs,means,marker='o' if len(rs)<=12 else None,label=method)
        if len(a.seeds)>1:ax.fill_between(rs,mins,maxs,alpha=.12)
    ax.set(xlabel='Communication round',ylabel='Validation macro-F1',title=f'Supervised FL — {a.strategy}, {a.labels}% labeled')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    out=base/'comparison';out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'validation_macro_f1.png',dpi=180);plt.close(fig)
    data=[]
    for seed in a.seeds:
        summary=json.loads((base/f'seed_{seed}'/'summary.json').read_text())
        for method,row in summary['models'].items():
            data.append({'seed':seed,'method':method,'best_val_macro_f1':row['best_val_macro_f1'],'selected_round':row['selected_round']})
    with (out/'best_validation_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['seed','method','best_val_macro_f1','selected_round']);w.writeheader();w.writerows(data)
    print('Saved plot and table in:',out)
    print('These are validation results, not final test results.')
if __name__=='__main__':main()
