"""Step 9: download only final-v2 source images, extract object crops.
Independent from the existing 420-crop pilot; persistent image byte accounting.
"""
import csv,json,math,os,time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image,UnidentifiedImageError

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/'data/manifests/expanded/selected_frames_v2.jsonl'
OUT=ROOT/'data/manifests/expanded/all_crops_v2.csv'
RESULTS=ROOT/'results/phase1_expanded'
CROP_ROOT=ROOT/'data/processed/crops_v2'
RAW=ROOT/'data/raw/images_v2'
LEDGER=ROOT/'data/manifests/image_download_ledger.json'  # shared with pilot
METADATA_LEDGER=ROOT/'data/manifests/metadata_expansion_ledger.json'
BASE='https://huggingface.co/datasets/dronefreak/BDD100K/resolve/main/'
CAP_MB=850
MAX_CROP_SIDE=160
MIN_SIDE=20
JPEG_QUALITY=85
KEEP_SOURCE_IMAGES=False
FIELDS=('crop_id','crop_path','split','source_image','source_split','class',
        'weather','timeofday','scene','environment','box_index','x','y','w','h')


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader()
        w.writerows({k:r.get(k,'') for k in FIELDS} for r in rows)
    os.replace(tmp,path)


def load_old():
    if not OUT.exists(): return []
    with OUT.open(newline='',encoding='utf-8') as f:
        return [r for r in csv.DictReader(f) if (ROOT/r['crop_path']).is_file()]


def reserved_metadata_bytes():
    cached=sum(p.stat().st_size for p in (ROOT/'data/raw/labels').rglob('*.json'))
    # Cached bytes approximate historical annotation traffic; expansion ledger can be larger.
    expanded=0
    if METADATA_LEDGER.exists():
        expanded=int(json.loads(METADATA_LEDGER.read_text()).get('new_bytes',0))
    # Include the pilot's known ~10.14MB metadata traffic as a conservative minimum.
    return max(cached, expanded+10_140_000)


def download(session,remote,dest,ledger,budget):
    url=BASE+quote(remote,safe='/')
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix('.jpg.part')
    tmp.unlink(missing_ok=True)
    try:
        with session.get(url,stream=True,timeout=(20,90)) as response:
            response.raise_for_status()
            n=response.headers.get('Content-Length')
            if n and ledger['download_bytes']+int(n)>budget:
                raise RuntimeError('Image transfer budget would be exceeded.')
            with tmp.open('wb') as f:
                for chunk in response.iter_content(64*1024):
                    if not chunk: continue
                    ledger['download_bytes']+=len(chunk)  # count received bytes even if incomplete
                    LEDGER.write_text(json.dumps(ledger,indent=2),encoding='utf-8')
                    if ledger['download_bytes']>budget:
                        raise RuntimeError('Image transfer budget exhausted during download.')
                    f.write(chunk)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    if not IN.exists(): raise SystemExit('Run 08_select_expanded_dataset.py first.')
    frames=[json.loads(s) for s in IN.read_text(encoding='utf-8').splitlines() if s.strip()]
    if not frames: raise SystemExit('Expanded selection is empty.')
    RESULTS.mkdir(parents=True,exist_ok=True);LEDGER.parent.mkdir(parents=True,exist_ok=True)
    ledger=json.loads(LEDGER.read_text()) if LEDGER.exists() else {'download_bytes':0}
    ledger['download_bytes']=int(ledger.get('download_bytes',0))
    metadata=reserved_metadata_bytes()
    image_cap=int(CAP_MB*1_000_000)-metadata
    if image_cap<=0: raise SystemExit('Cumulative budget exhausted by metadata.')
    existing=load_old(); done={r['crop_id'] for r in existing}; rows=list(existing)
    attempts=Counter(); failed=[]; session=requests.Session()
    session.headers['User-Agent']='Academic-BTP-final-v2/1.0'
    print(f'Selected frames: {len(frames)}; metadata reserve: {metadata/1e6:.1f} MB;')
    print(f'Previously charged image transfers: {ledger["download_bytes"]/1e6:.1f} MB; total cap: {CAP_MB} MB')
    for i,r in enumerate(frames,1):
        req=[(b,f"{r['source_split']}_{r['shard']}_{Path(r['source_image']).stem}_{int(b['box_index'])}") for b in r['boxes']]
        if all(cid in done for _,cid in req):
            attempts['already_done']+=1; continue
        if ledger['download_bytes']>=image_cap:
            print('STOP: download budget reached');break
        raw=RAW/r['source_split']/r['shard']/r['source_image']
        try:
            if not raw.is_file(): download(session,r['remote_path'],raw,ledger,image_cap)
            with Image.open(raw) as im:
                rgb=im.convert('RGB')
                for box,cid in req:
                    if cid in done:continue
                    x,y,w,h=map(float,(box['x'],box['y'],box['w'],box['h']))
                    if not all(math.isfinite(v) for v in (x,y,w,h)):continue
                    l,t=max(0,math.floor(x)),max(0,math.floor(y))
                    rr,bb=min(rgb.width,math.ceil(x+w)),min(rgb.height,math.ceil(y+h))
                    if rr-l<MIN_SIDE or bb-t<MIN_SIDE:
                        attempts['too_small_after_clip']+=1;continue
                    crop=rgb.crop((l,t,rr,bb));crop.thumbnail((MAX_CROP_SIDE,MAX_CROP_SIDE),Image.Resampling.LANCZOS)
                    dest=CROP_ROOT/r['split']/f'{cid}.jpg'
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    crop.save(dest,'JPEG',quality=JPEG_QUALITY)
                    rows.append({'crop_id':cid,'crop_path':dest.relative_to(ROOT).as_posix(),
                        'split':r['split'],'source_image':r['source_image'],'source_split':r['source_split'],
                        'class':box['class'],'weather':r['weather'],'timeofday':r['timeofday'],
                        'scene':r['scene'],'environment':r['environment'],'box_index':box['box_index'],
                        'x':x,'y':y,'w':w,'h':h})
                    done.add(cid);attempts['new_crops']+=1
            attempts['frames_processed']+=1
        except RuntimeError as e:
            print('STOP:',e);break
        except (requests.RequestException,OSError,ValueError,TypeError,KeyError,UnidentifiedImageError) as e:
            failed.append({'remote_path':r['remote_path'],'error':str(e)})
            attempts['failed_frames']+=1
            print('WARNING:',r['source_image'],str(e)[:110])
        finally:
            if raw.is_file() and not KEEP_SOURCE_IMAGES:raw.unlink()
            if i%20==0:
                write_csv(OUT,rows)
                print(f'{i}/{len(frames)} source frames; {len(rows)} valid crops; '+
                      f'{(metadata+ledger["download_bytes"])/1e6:.1f}/{CAP_MB} MB accounted')
            time.sleep(.02)
    write_csv(OUT,rows)
    LEDGER.write_text(json.dumps(ledger,indent=2),encoding='utf-8')
    summary={'crops_total':len(rows),'crops_by_split':dict(Counter(r['split'] for r in rows)),
             'class_counts':dict(Counter(r['class'] for r in rows)),
             'environment_counts':dict(Counter(r['environment'] for r in rows)),
             'metadata_bytes_reserved':metadata,'image_download_bytes_pilot_plus_v2':ledger['download_bytes'],
             'accounted_mb':round((metadata+ledger['download_bytes'])/1e6,2),
             'note':'This is a conservative estimate from available ledgers, not a verified lifetime network total.',
             'attempts':dict(attempts),'failed_frames':len(failed)}
    (RESULTS/'extraction_summary_v2.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if failed:
        path=RESULTS/'failed_images_v2.csv'
        with path.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['remote_path','error']);writer.writeheader();writer.writerows(failed)
    print('\n=== EXPANDED CROP EXTRACTION ===')
    print(json.dumps(summary,indent=2))
    print('Saved:',OUT)

if __name__=='__main__':main()
