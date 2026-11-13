"""Expand BDD100K annotation sample across entire shards, without JPEG downloads.

Run from btp_part2 using the existing virtual environment.
Dependencies: requests, PyYAML (already installed). Retains existing annotation JSONs.
"""
import json
import random
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from common import ROOT, config

REPO = 'dronefreak/BDD100K'
API = f'https://huggingface.co/api/datasets/{REPO}/tree/main/'
RAW = f'https://huggingface.co/datasets/{REPO}/resolve/main/'
CACHE = ROOT / 'data' / 'raw' / 'labels'
OUT = ROOT / 'results' / 'phase1'
MAN = ROOT / 'data' / 'manifests'

# Modify these only if bandwidth or timing is tight.
TRAIN_PER_SHARD = 250  # 9 shards: up to 2250 training annotations
VAL_PER_SHARD = 200    # 2 shards: up to 400 official validation annotations
MAX_NEW_METADATA_MB = 100
CUMULATIVE_LIMIT_MB = 850
SEED = 42


def session():
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=['GET'])
    s.mount('https://', HTTPAdapter(max_retries=retry))
    s.headers['User-Agent'] = 'Academic-BTP-annotation-audit/1.0'
    return s


def listed_json_files(s, shard):
    # The Hugging Face tree endpoint uses cursor pagination via its Link header.
    url = API + quote(shard, safe='/')
    seen_pages = set()
    found = []
    while url:
        if url in seen_pages:
            raise RuntimeError('Repeated pagination URL; stopping to avoid a loop')
        seen_pages.add(url)
        with s.get(url, params={'limit': 1000} if len(seen_pages) == 1 else None,
                   timeout=(15, 90)) as r:
            r.raise_for_status()
            data = r.json()
            found.extend(x['path'] for x in data
                         if x.get('type') == 'file' and x.get('path', '').endswith('.json'))
            url = r.links.get('next', {}).get('url')
    return sorted(set(found))


def local_path(remote):
    # Compatible with local_label_path() in our existing common.py.
    return CACHE / remote.replace('/', '_')


def existing_total_bytes():
    # Reserve for all previously cached annotations, even before our new ledger exists.
    return sum(p.stat().st_size for p in CACHE.rglob('*.json') if p.is_file())


def main():
    cfg = config()
    rng = random.Random(SEED)
    CACHE.mkdir(parents=True, exist_ok=True)
    MAN.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ledger_path = MAN / 'metadata_expansion_ledger.json'
    ledger = json.loads(ledger_path.read_text(encoding='utf-8')) if ledger_path.exists() else {'new_bytes': 0}
    # Existing image download ledger: avoid forgetting previous image traffic.
    image_ledger_path = MAN / 'image_download_ledger.json'
    image_bytes = 0
    if image_ledger_path.exists():
        image_bytes = int(json.loads(image_ledger_path.read_text(encoding='utf-8')).get('download_bytes', 0))

    newly_transferred_this_run = 0
    cumulative_limit = int(min(float(cfg.get('download_limit_mb', 850)), CUMULATIVE_LIMIT_MB) * 1_000_000)
    metadata_limit = int(MAX_NEW_METADATA_MB * 1_000_000)
    s = session()
    report = []
    for split in ('train', 'val'):
        n_shards = 9 if split == 'train' else 2
        per_shard = TRAIN_PER_SHARD if split == 'train' else VAL_PER_SHARD
        for shard_num in range(n_shards):
            shard = f'data/labels/100k/{split}/shard_{shard_num:03d}'
            paths = listed_json_files(s, shard)
            chosen = rng.sample(paths, min(per_shard, len(paths)))
            ok, cached, failed = 0, 0, 0
            for remote in chosen:
                dest = local_path(remote)
                if dest.is_file():
                    cached += 1
                    continue
                # Check conservative budget BEFORE downloading the next item.
                old_metadata = existing_total_bytes()
                projected = old_metadata + image_bytes + newly_transferred_this_run
                if projected >= cumulative_limit - 1_000_000:
                    raise RuntimeError('Cumulative 850 MB allowance reached; stopped safely')
                if newly_transferred_this_run >= metadata_limit - 200_000:
                    raise RuntimeError('100 MB metadata-expansion allowance reached')
                url = RAW + quote(remote, safe='/')
                try:
                    with s.get(url, stream=True, timeout=(15, 90)) as r:
                        r.raise_for_status()
                        length = r.headers.get('Content-Length')
                        if length and int(length) + projected > cumulative_limit:
                            raise RuntimeError('Next annotation would exceed cumulative limit')
                        payload = bytearray()
                        for chunk in r.iter_content(16 * 1024):
                            if chunk:
                                payload.extend(chunk)
                                newly_transferred_this_run += len(chunk)
                                ledger['new_bytes'] = int(ledger.get('new_bytes', 0)) + len(chunk)
                                ledger_path.write_text(json.dumps(ledger, indent=2), encoding='utf-8')
                                if (newly_transferred_this_run >= metadata_limit or
                                    projected + len(payload) >= cumulative_limit):
                                    raise RuntimeError('Download limit reached; stopped')
                    item = json.loads(payload)
                    if not isinstance(item, dict) or 'frames' not in item or 'attributes' not in item:
                        raise ValueError('Unexpected annotation structure')
                    tmp = dest.with_suffix('.json.part')
                    tmp.write_bytes(payload)
                    tmp.replace(dest)
                    ok += 1
                except (requests.RequestException, ValueError, OSError) as ex:
                    failed += 1
                    print(f'  failed {Path(remote).name}: {ex}')
                if (ok + cached) % 50 == 0 and (ok + cached) > 0:
                    print(f'  {shard}: {ok} new / {cached} cached / {failed} failed')
            report.append({'split': split, 'shard': shard, 'listed': len(paths),
                           'requested': len(chosen), 'new': ok, 'cached': cached, 'failed': failed})
            print(f'{shard}: listed {len(paths)}, sampled {len(chosen)}, new {ok}, cached {cached}, failed {failed}')
    import csv
    with (OUT / 'annotation_expansion_summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    print(f'New metadata received this run: {newly_transferred_this_run/1e6:.2f} MB')
    print('Annotation pool expanded. Next: audit distributions before selecting JPEGs.')


if __name__ == '__main__':
    main()
