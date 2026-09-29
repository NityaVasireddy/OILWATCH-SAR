"""Inspect the configured dataset without assuming its layout."""
import os, json, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
try: import rasterio
except Exception: rasterio=None
try: import rarfile
except Exception: rarfile=None

EXTS={".png",".jpg",".jpeg",".tif",".tiff",".npy",".npz",".bmp",".gif"}
MASK_WORDS=("mask","masks","label","labels","gt","groundtruth","ground_truth","annotation","annotations")

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def inspect_file(p):
    d={"path":str(p),"extension":p.suffix.lower(),"bytes":p.stat().st_size}
    try:
        if p.suffix.lower() in {'.tif','.tiff'} and rasterio:
            with rasterio.open(p) as ds:
                a=ds.read()
                d.update(width=ds.width,height=ds.height,channels=ds.count,dtype=str(ds.dtypes),min=float(np.nanmin(a)),max=float(np.nanmax(a)),crs=str(ds.crs) if ds.crs else None,transform=list(ds.transform),resolution=list(ds.res),bounds=list(ds.bounds),nodata=ds.nodata)
        elif p.suffix.lower() in {'.npy','.npz'}:
            a=np.load(p, allow_pickle=False)
            if hasattr(a,'files'):
                d['arrays']={k:{'shape':list(a[k].shape),'dtype':str(a[k].dtype),'min':float(np.nanmin(a[k])),'max':float(np.nanmax(a[k]))} for k in a.files}
            else: d.update(shape=list(a.shape),dtype=str(a.dtype),min=float(np.nanmin(a)),max=float(np.nanmax(a)))
        elif p.suffix.lower() in EXTS:
            im=Image.open(p)
            a=np.asarray(im)
            d.update(width=im.width,height=im.height,channels=(1 if a.ndim==2 else a.shape[2]),dtype=str(a.dtype),min=float(np.nanmin(a)),max=float(np.nanmax(a)))
            if any(w in p.stem.lower() for w in MASK_WORDS): d['unique_values']=np.unique(a).tolist()[:100]
    except Exception as e: d['error']=f'{type(e).__name__}: {e}'
    return d

def main():
    root=Path(os.getenv('DATASET_ROOT','')).expanduser()
    meta=Path('dataset/metadata'); meta.mkdir(parents=True,exist_ok=True)
    out={'dataset_root':str(root),'exists':root.exists(),'archive_notes':[],'directories':[],'files':[],'summary':{}}
    if not root.exists():
        out['archive_notes'].append('DATASET_ROOT is not configured or does not exist. No inspection performed.')
    else:
        if root.is_file() and root.suffix.lower()=='.rar':
            if rarfile is None: out['archive_notes'].append('RAR detected but rarfile is not installed.')
            else:
                try:
                    with rarfile.RarFile(root) as rf:
                        names=rf.namelist(); out['archive_notes'].append(f'RAR archive contains {len(names)} entries; files were not extracted automatically.')
                        out['archive_entries']=names[:1000]
                except Exception as e: out['archive_notes'].append(f'RAR inspection failed: {e}')
        else:
            out['directories']=[str(p.relative_to(root)) for p in root.rglob('*') if p.is_dir()][:5000]
            fs=[p for p in root.rglob('*') if p.is_file()]
            out['files']=[inspect_file(p) for p in fs[:5000]]
            out['summary']={'file_count':len(fs),'extension_counts':dict(Counter(p.suffix.lower() for p in fs)),'image_like_count':sum(p.suffix.lower() in EXTS for p in fs)}
            hashes={}
            for p in fs:
                try: hashes.setdefault(sha256(p),[]).append(str(p.relative_to(root)))
                except Exception: pass
            out['duplicate_groups']=[v for v in hashes.values() if len(v)>1]
    dup_rows = []
    for group in out.get('duplicate_groups', []):
        dup_rows.append({'sha256': sha256(Path(root / group[0])) if root.exists() and root.is_dir() else '', 'paths': '|'.join(group), 'duplicate_count': len(group)})
    import csv
    with open(meta / 'duplicates.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['sha256','paths','duplicate_count']); w.writeheader(); w.writerows(dup_rows)
    Path('dataset/metadata/dataset_inspection.json').write_text(json.dumps(out,indent=2,default=str),encoding='utf-8')
    md=['# Dataset inspection','',f'- DATASET_ROOT: `{root}`',f'- Exists: `{root.exists()}`','']
    md.append('## Summary'); md.append('```json'); md.append(json.dumps(out.get('summary',{}),indent=2)); md.append('```')
    md.append('## Archive / metadata notes'); md.extend(f'- {x}' for x in out.get('archive_notes',[]))
    if out.get('files'):
        md.append('\n## Sample file inspection'); md.append('| Path | Size | Shape | Channels | Dtype | Min | Max | CRS | Resolution |') ; md.append('|---|---:|---|---:|---|---:|---:|---|---|')
        for x in out['files'][:100]: md.append(f"| `{x.get('path')}` | {x.get('bytes','')} | {x.get('width','')}×{x.get('height','')} | {x.get('channels','')} | {x.get('dtype','')} | {x.get('min','')} | {x.get('max','')} | {x.get('crs','')} | {x.get('resolution','')} |")
    Path('docs/dataset_inspection.md').write_text('\n'.join(md),encoding='utf-8')
    print('Wrote dataset/metadata/dataset_inspection.json and docs/dataset_inspection.md')
if __name__=='__main__': main()
