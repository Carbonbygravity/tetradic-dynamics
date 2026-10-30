"""Hash paired normative inputs without modifying them."""
from pathlib import Path
import argparse,hashlib,json,time
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-dir',type=Path,required=True,help='Directory containing normative/dyad*_P*_roi.h5')
    parser.add_argument('--source-specification',type=Path,help='Optional upstream frozen source specification')
    parser.add_argument('--output-manifest',type=Path,required=True,help='New manifest file; never overwrites an existing file')
    args=parser.parse_args()
    root=args.input_dir.resolve()
    output=args.output_manifest.resolve()
    start=time.perf_counter();rows=[]
    for d in [1,2,3,4,*range(6,17)]:
        for partner in [1,2]:
            relative=Path('normative')/f'dyad{d}_P{partner}_roi.h5'
            p=root/relative;s=p.stat()
            h=digest(p);after=p.stat()
            assert (s.st_size,s.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            rows.append(dict(dyad=d,partner=partner,path=str(p.resolve()),relative_path=relative.as_posix(),size_bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=h))
        print('HASHED',d,flush=True)
    report=dict(scope='Complete SHA256 read of all 30 normative source HDF5 files; no source writes.',inputs=rows,source_specification_sha256=digest(args.source_specification.resolve()) if args.source_specification else None,elapsed_seconds=time.perf_counter()-start)
    with output.open('x') as f:json.dump(report,f,indent=2)
    print('DONE',sum(r['size_bytes'] for r in rows),report['elapsed_seconds'],flush=True)
