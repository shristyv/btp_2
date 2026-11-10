"""Validate original-source boundaries and write train/val/test manifests."""
from collections import Counter
from common import ROOT, read_csv, write_csv


def main():
    src=ROOT/'data'/'manifests'/'all_crops.csv'
    if not src.exists(): raise FileNotFoundError('Run 03 first')
    rows=read_csv(src)
    partitions={s:[r for r in rows if r['split']==s] for s in ('train','val','test')}
    seen={s:{r['source_image'] for r in group} for s,group in partitions.items()}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert seen[a].isdisjoint(seen[b]),f'LEAKAGE: {a} vs {b}'
    for s,group in partitions.items():
        write_csv(ROOT/'data'/'manifests'/f'{s}.csv',group)
        print(f'{s}: {len(group)} crops from {len(seen[s])} source frames; class counts {dict(Counter(r["class"] for r in group))}')
    print('PASS: no identical source filenames across splits.')
    print('NOTE: separate source-video overlap cannot be excluded without verified sequence identifiers.')

if __name__=='__main__':main()
