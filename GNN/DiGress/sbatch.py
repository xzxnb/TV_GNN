import os

commands = [
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=6 dataset.name=fo2 filename=friends-person5 seed=0"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=6 dataset.name=fo2 filename=friends-person5 seed=1"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=6 dataset.name=fo2 filename=friends-person5 seed=2"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=4 dataset.name=fo2 filename=friends-person5 seed=0"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=4 dataset.name=fo2 filename=friends-person5 seed=1"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=4 dataset.name=fo2 filename=friends-person5 seed=2"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=2 dataset.name=fo2 filename=friends-person5 seed=0"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=2 dataset.name=fo2 filename=friends-person5 seed=1"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=2 dataset.name=fo2 filename=friends-person5 seed=2"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=1 dataset.name=fo2 filename=friends-person5 seed=0"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=1 dataset.name=fo2 filename=friends-person5 seed=1"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=1 dataset.name=fo2 filename=friends-person5 seed=2"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=6 dataset.name=fo2 filename=friends-person10 seed=0"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=6 dataset.name=fo2 filename=friends-person10 seed=1"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=6 dataset.name=fo2 filename=friends-person10 seed=2"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=4 dataset.name=fo2 filename=friends-person10 seed=0"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=4 dataset.name=fo2 filename=friends-person10 seed=1"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=4 dataset.name=fo2 filename=friends-person10 seed=2"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=2 dataset.name=fo2 filename=friends-person10 seed=0"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=2 dataset.name=fo2 filename=friends-person10 seed=1"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=2 dataset.name=fo2 filename=friends-person10 seed=2"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=1 dataset.name=fo2 filename=friends-person10 seed=0"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=1 dataset.name=fo2 filename=friends-person10 seed=1"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=1 dataset.name=fo2 filename=friends-person10 seed=2"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=6 dataset.name=fo2 filename=friends-person20 seed=0"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=6 dataset.name=fo2 filename=friends-person20 seed=1"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=6 dataset.name=fo2 filename=friends-person20 seed=2"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=4 dataset.name=fo2 filename=friends-person20 seed=0"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=4 dataset.name=fo2 filename=friends-person20 seed=1"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=4 dataset.name=fo2 filename=friends-person20 seed=2"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=2 dataset.name=fo2 filename=friends-person20 seed=0"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=2 dataset.name=fo2 filename=friends-person20 seed=1"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=2 dataset.name=fo2 filename=friends-person20 seed=2"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=1 dataset.name=fo2 filename=friends-person20 seed=0"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=1 dataset.name=fo2 filename=friends-person20 seed=1"',
    # 'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=1 dataset.name=fo2 filename=friends-person20 seed=2"',
    'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=1 dataset.name=molecules max_num_atoms=20 seed=0"',
    'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=2 dataset.name=molecules max_num_atoms=20 seed=0"',
    'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=4 dataset.name=molecules max_num_atoms=20 seed=0"',
    'singularity exec --nv ../shinyhappiness.sif /bin/bash -c "PYTHONPATH=./:../ python dgd/main.py model=continuous model.n_layers=6 dataset.name=molecules max_num_atoms=20 seed=0"',
]

os.makedirs("sbatch_outputs", exist_ok=True)

for command in commands:
    output_name = (
        command.split("dgd/main.py")[-1]
        .replace(" ", "")
        .replace("=", "-")
        .replace("/", "-")
        .replace('"', "")
        .replace(".", "-")
    ) + ".out"
    batch = f"""#!/bin/sh
{command}
"""
    with open(f"sbatch_outputs/{output_name}.batch", "w") as f:
        f.write(batch)
    run = f"sbatch -p gpulong --gres=gpu:1 -o sbatch_outputs/{output_name} sbatch_outputs/{output_name}.batch"
    print(f"Running: {run}")
    os.system(run)
