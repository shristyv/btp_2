"""Step 10: strict expanded crop validation. Only reads local data; writes reports."""
import argparse, csv, json, hashlib
from pathlib import Path
from collections import Counter, defaultdict

CLASSES={'bike','bus','car','motor','person','rider','traffic light','traffic sign','truck'}
SPLITS={'train':3200,'val':600,'test':600}
REQUIRED={'crop_id','crop_path','split','source_image','source_split','class','environment','box_index'}

def read_csv(path):
    with path.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if not REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError('Missing columns: '+str(sorted(REQUIRED-set(reader.fieldnames or []))))
        return list(reader)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--verify-images',action='store_true',help='Decode every crop with Pillow (slower)')
    args=p.parse_args(); root=args.root.resolve()
    path=root/'data/manifests/expanded/all_crops_v2.csv'
    if not path.is_file(): raise SystemExit('Missing '+str(path))
    rows=read_csv(path); errors=[]; warnings=[]
    by_split=Counter(); classes=defaultdict(Counter); envs=defaultdict(Counter)
    sources=defaultdict(set); ids=set(); paths=set(); source_boxes=set(); hashed=hashlib.sha256()
    try:
        from PIL import Image
    except ImportError:
        Image=None
    if args.verify_images and Image is None: raise SystemExit('Install pillow for --verify-images')
    for i,r in enumerate(rows,2):
        cid=r['crop_id']; split=r['split']; cls=r['class']; source=(r['source_split'],r['source_image'])
        if not cid:errors.append(f'row {i}: empty crop_id')
        if cid in ids:errors.append(f'row {i}: duplicate crop_id {cid}')
        ids.add(cid)
        if r['crop_path'] in paths:errors.append(f'row {i}: duplicate crop_path {r["crop_path"]}')
        paths.add(r['crop_path'])
        if (source,r['box_index']) in source_boxes:errors.append(f'row {i}: duplicate source bounding box {source} {r["box_index"]}')
        source_boxes.add((source,r['box_index']))
        if split not in SPLITS:errors.append(f'row {i}: unexpected split {split}')
        if cls not in CLASSES:errors.append(f'row {i}: unexpected class {cls}')
        if (split=='test' and r['source_split']!='val') or (split in ('train','val') and r['source_split']!='train'):
            errors.append(f'row {i}: source split boundary incorrect')
        cp=Path(r['crop_path'])
        if cp.is_absolute() or '..' in cp.parts:errors.append(f'row {i}: unsafe crop path');continue
        file=root/cp
        if not file.is_file():errors.append(f'row {i}: missing crop {cp}')
        elif args.verify_images:
            try:
                with Image.open(file) as im: im.verify()
            except Exception as exc:errors.append(f'row {i}: bad image {cp}: {exc}')
        by_split[split]+=1;classes[split][cls]+=1;envs[split][r['environment']]+=1;sources[split].add(source)
        hashed.update((cid+'|'+r['crop_path']+'|'+split+'|'+cls+'\n').encode())
    for split,target in SPLITS.items():
        if by_split[split]!=target:errors.append(f'{split}: expected {target} crops, observed {by_split[split]}')
    for a,b in [('train','val'),('train','test'),('val','test')]:
        overlap=sources[a]&sources[b]
        if overlap:errors.append(f'{a}/{b}: {len(overlap)} overlapping source frames')
    for split in SPLITS:
        for cls in sorted(CLASSES):
            n=classes[split][cls]
            if not n:errors.append(f'{split}: class {cls} is absent')
            if split=='test' and n<30:warnings.append(f'test {cls} has {n} samples (<30); per-class estimates unstable')
            if split=='train' and n<50:warnings.append(f'train {cls} has {n} samples (<50); few-shot class')
    unk=sum(envs[s]['unknown'] for s in SPLITS)
    if unk:warnings.append(f'{unk} crops have unknown environment; unknown must not be used as a weather type')
    # A dataset assembled with deliberate class rebalancing is not an unbiased prevalence estimate.
    warnings.append('Class-aware selection changes class prevalence: report evaluation on this sampled distribution, not natural driving prevalence.')
    report={'pass':not errors,'crops':len(rows),'counts':dict(by_split),
            'source_frames':{s:len(sources[s]) for s in SPLITS},
            'class_counts':{s:dict(classes[s]) for s in SPLITS},
            'environment_counts':{s:dict(envs[s]) for s in SPLITS},
            'manifest_digest':hashed.hexdigest(),'errors':errors[:100],'warnings':warnings}
    out=root/'results/phase1_expanded';out.mkdir(parents=True,exist_ok=True)
    (out/'step10_dataset_validation.json').write_text(json.dumps(report,indent=2))
    print('STEP 10:', 'PASS' if report['pass'] else 'FAIL','|',dict(by_split),'|',report['source_frames'])
    print('Errors:',len(errors),'Warnings:',len(warnings))
    for x in errors[:15]:print('ERROR:',x)
    for x in warnings:print('NOTE:',x)
    print('Report:',out/'step10_dataset_validation.json')
    if errors:raise SystemExit(1)

if __name__=='__main__': main()
