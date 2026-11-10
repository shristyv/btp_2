"""Download only selected JPEGs, create 160px crops, keep persistent sub-1GB budget."""
import io
import json
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from common import ROOT, config, BudgetSession, DET_REPO, load_jsonl, write_csv, DownloadLimit

FIELDS=['crop_id','crop_path','split','source_image','source_split','class','weather','timeofday',
        'environment','box_index','x','y','w','h']


def main():
    cfg=config(); dl=BudgetSession(cfg)
    selection=ROOT/'data'/'manifests'/'selected_frames.jsonl'
    if not selection.exists(): raise FileNotFoundError('Run 02 first')
    manifest=ROOT/'data'/'manifests'/'all_crops.csv'
    old=[]
    if manifest.exists():
        from common import read_csv
        old=read_csv(manifest)
    done={r['crop_id'] for r in old}
    new=[]; failed=[]
    for i,item in enumerate(load_jsonl(selection),1):
        raw=ROOT/'data'/'raw'/'images'/item['source_split']/item['source_image']
        expected=[f'{item["source_split"]}_{Path(item["source_image"]).stem}_{b["box_index"]}' for b in item['boxes']]
        if all(x in done for x in expected): continue
        try:
            dl.get(DET_REPO,item['remote_path'],raw)
            with Image.open(raw) as im:
                image=im.convert('RGB')
                for b,crop_id in zip(item['boxes'],expected):
                    if crop_id in done:continue
                    x,y,w,h=(float(b[k]) for k in ['x','y','w','h'])
                    xyxy=(max(0,int(x)),max(0,int(y)),min(image.width,int(x+w)),min(image.height,int(y+h)))
                    if xyxy[2]-xyxy[0]<cfg['min_box_side'] or xyxy[3]-xyxy[1]<cfg['min_box_side']:continue
                    crop=image.crop(xyxy)
                    crop.thumbnail((cfg['crop_max_side'],cfg['crop_max_side']),Image.Resampling.LANCZOS)
                    dest=ROOT/'data'/'processed'/'crops'/item['split']/f'{crop_id}.jpg'
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    crop.save(dest,quality=cfg['jpeg_quality'])
                    new.append({'crop_id':crop_id,'crop_path':str(dest.relative_to(ROOT)).replace('\\','/'),
                                'split':item['split'],'source_image':item['source_image'],
                                'source_split':item['source_split'],'class':b['class'],
                                'weather':item['weather'],'timeofday':item['timeofday'],
                                'environment':item['environment'],'box_index':b['box_index'],
                                'x':x,'y':y,'w':w,'h':h})
                    done.add(crop_id)
        except DownloadLimit as ex:
            print('Download cap reached:',ex)
            break
        except Exception as ex:
            failed.append({'source_image':item['source_image'],'error':str(ex)})
            print('WARN:',item['source_image'],str(ex)[:110])
        finally:
            if raw.exists() and not cfg.get('keep_raw_images',False):raw.unlink()
        if i%30==0:
            write_csv(manifest,old+new,FIELDS)
            print(f'Processed {i} frames; {len(old)+len(new)} crops; transferred {dl.spent/1e6:.1f} MB')
    write_csv(manifest,old+new,FIELDS)
    if failed: write_csv(ROOT/'results'/'phase1'/'failed_downloads.csv',failed)
    print(f'Wrote {len(old)+len(new)} crop rows to {manifest}')
    print(f'Pipeline network response bytes (persistent): {dl.spent/1e6:.1f} MB')

if __name__=='__main__':main()
