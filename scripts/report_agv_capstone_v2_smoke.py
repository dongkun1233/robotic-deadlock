"""Read smoke caches and estimate serial main-fit simulation cost; no simulation."""
import argparse
import statistics
from scripts.train_agv_capstone_v2 import read
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    manifest=read(args.manifest)
    base=args.manifest.parent/'training_smoke_12'
    status=read(base/'status.json') if (base/'status.json').exists() else None
    print('Smoke status:',status)
    total_s=0.
    for name in ('single','double','multi'):
        files=list((base/'cache'/name).glob('*.json.gz'))
        rows=[r for p in files for r in read(p)['rows']]
        if not rows:
            print(name,'no measured episodes yet')
            continue
        average=statistics.mean(r['episode_wall_s'] for r in rows)
        training=sum(s['layout']==name and s['split']=='train' for s in manifest['scenarios'])
        fits=len(manifest['optimizer_seeds'])*len(manifest['methods'])
        candidates=6+(manifest['search_budget']-6)*fits
        projected=candidates*training*average
        total_s+=projected
        size=sum(p.stat().st_size for p in files)
        bytes_per_episode=size/len(rows)
        print(name,'measured episodes',len(rows),'mean wall seconds',round(average,3),
              'planned main episodes up to',candidates*training,
              'rough serial simulation hours',round(projected/3600,2),
              'rough compressed cache GB',round(bytes_per_episode*candidates*training/1e9,2))
    print('Partial-layout estimate if smoke is unfinished; rough total serial simulation hours:',round(total_s/3600,2))
    print('Excludes optimizer/cache I/O overhead; assumes smoke runtimes represent main cases.')
    print('Duplicate candidate reuse can reduce work; these estimates are not a completion-time guarantee.')


if __name__=='__main__':
    main()
