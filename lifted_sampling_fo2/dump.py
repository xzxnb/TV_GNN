import pickle
import json
from pathlib import Path

def dump(samples, path):
    ret = []
    for sample in samples:
        ret.append(list(
            str(l) for l in sample
        ))
    with open(path, 'w') as f:
        json.dump(ret, f)

def dump_filt_aux(samples, path):
    ret = []
    for sample in samples:
        ret.append(list(
            str(l)for l in sample
            if not (str(l).startswith("aux") or str(l).startswith("neighbor"))
        ))
    with open(path, 'w') as f:
        json.dump(ret, f)



dirs = [Path('./lifted_sampling_fo2/outputs/color1/num100k_mln/domain10')]
root_dir = Path('GNN/wfomi_data/json/color1_100k_mln')
root_dir.mkdir(parents=True, exist_ok=True)
for d in dirs:
    out_dir = root_dir
    out_dir.mkdir(exist_ok=True)
    for f in d.iterdir():
        # if f suffix is .pickle
        if f.suffix != '.pkl':
            continue
        name = f.stem
        _, domain_size, tv, idx = name.split('_')
        tv_val = round(float(tv[2:]), 1)
        tv = f'{tv[:2]}{tv_val}'
        d_dir = out_dir / domain_size
        d_dir.mkdir(exist_ok=True)
        tv_dir = d_dir / tv
        tv_dir.mkdir(exist_ok=True)
        samples = pickle.load(open(f, 'rb'))
        out_path = tv_dir / f'{idx}.json'
        dump_filt_aux(samples, out_path)

