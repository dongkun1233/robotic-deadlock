"""Audit completed training records and consolidate episode-level results."""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('output_dir',type=Path)
    args=p.parse_args()
    folder=args.output_dir
    state=json.loads((folder/'checkpoint.json').read_text())
    config=state['config']
    assert config['split']=='train'
    assert (folder/'run_complete.json').exists()
    # Git stores one compact archive; restore the original per-evaluation CSVs
    # on demand so the same audit can run after checking out the branch.
    missing=[t for t in state['trials'] if not (folder/t['raw_file']).exists()]
    if missing:
        archived={}
        with gzip.open(folder/'training_episode_results.csv.gz','rt',newline='') as file:
            for row in csv.DictReader(file):
                key=(row['model'],int(row['optimizer_seed']),row['algorithm'],int(row['evaluation']))
                archived.setdefault(key,[]).append({k:v for k,v in row.items() if k not in ('model','optimizer_seed','algorithm','evaluation')})
        for t in missing:
            path=(folder/t['raw_file']).resolve()
            assert path.is_relative_to((folder/'raw').resolve())
            path.parent.mkdir(exist_ok=True)
            rows=archived[(t['model'],t['seed'],t['algorithm'],t['evaluation'])]
            with path.open('x',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=list(rows[0]))
                writer.writeheader();writer.writerows(rows)
    keys=set()
    episodes=[]
    for t in state['trials']:
        key=(t['model'],t['seed'],t['algorithm'],t['evaluation'])
        assert key not in keys, key
        keys.add(key)
        with (folder/t['raw_file']).open(newline='') as f:
            rows=list(csv.DictReader(f))
        assert len(rows)==config['training_scenarios']
        ids=[r['scenario_id'] for r in rows]
        assert len(ids)==len(set(ids))
        assert all(i.startswith('train_') for i in ids)
        incomplete=sum(r['timed_out']=='True' for r in rows)
        average=statistics.mean(float(r['total_steps']) for r in rows)
        assert incomplete==t['incomplete']
        assert abs(average-t['mean_total_steps'])<1e-10
        assert abs(incomplete*(config['max_steps']+1)+average-t['objective'])<1e-10
        for row in rows:
            assert (int(row['completed'])<int(row['robots'])) == (row['timed_out']=='True')
            episodes.append({'model':t['model'],'optimizer_seed':t['seed'],
                'algorithm':t['algorithm'],'evaluation':t['evaluation'],**row})
    algorithms=('basic','bo','pso')
    for model in config['models']:
        for seed in config['seeds']:
            initial=[]
            for algorithm in algorithms:
                history=sorted([t for t in state['trials'] if (t['model'],t['seed'],t['algorithm'])==(model,seed,algorithm)],key=lambda t:t['evaluation'])
                assert [t['evaluation'] for t in history]==list(range(1,config['budget']+1))
                initial.append([(t['weights'],t['objective']) for t in history[:config['initial_count']]])
            assert initial[0]==initial[1]==initial[2]
    expected=len(config['models'])*len(config['seeds'])*len(algorithms)*config['budget']
    assert len(state['trials'])==expected
    with (folder/'best_training_weights.csv').open(newline='') as file:
        best=list(csv.DictReader(file))
    for row in best:
        match=[t for t in state['trials'] if t['model']==row['model'] and t['seed']==int(row['seed']) and t['algorithm']==row['algorithm']]
        assert float(row['objective'])==min(t['objective'] for t in match)
    # The archive contains all objective evaluations, including bad/timed-out candidates.
    with (folder/'training_episode_results.csv.gz').open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed:
            with io.TextIOWrapper(compressed,encoding='utf-8',newline='') as text:
                writer=csv.DictWriter(text,fieldnames=list(episodes[0]))
                writer.writeheader()
                writer.writerows(episodes)
    summary=[]
    for model in config['models']:
        for algorithm in algorithms:
            rows=[r for r in best if r['model']==model and r['algorithm']==algorithm]
            values=[float(r['mean_total_steps']) for r in rows]
            summary.append({'model':model,'algorithm':algorithm,'optimizer_seeds':len(rows),
                'mean_best_training_steps':statistics.mean(values),
                'min_best_training_steps':min(values),'max_best_training_steps':max(values),
                'selected_incomplete_scenarios':sum(int(r['incomplete']) for r in rows)})
    with (folder/'training_summary.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(summary[0]))
        writer.writeheader();writer.writerows(summary)
    audit={'passed':True,'objective_evaluations':expected,'training_episode_rows':len(episodes),
        'shared_initial_vectors_and_objectives_verified':True,'training_only_verified':True,
        'episode_objectives_recomputed':True,'checkpoint_resume_test':'see tests/test_multicross_search.py',
        'validation_evaluated':False,'test_evaluated':False}
    (folder/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
    print('FCFS training:',state['baseline'])
    for row in summary:print(row)


if __name__=='__main__':main()
