# Command cheatsheet

- Create dataset

```bash
docker-compose run --rm base python data_creation/create_molecules.py --atom-size 8 && \
docker-compose run --rm base python data_creation/create_molecules.py --atom-size 12 && \
docker-compose run --rm base python data_creation/create_molecules.py --atom-size 15 && \
docker-compose run --rm base python data_creation/create_molecules.py --atom-size 20 && \
docker-compose run --rm base python data_creation/create_molecules.py --atom-size 25 && \
docker-compose run --rm base python data_creation/create_molecules.py --atom-size 30
```

- Plot

```bash
tmux new-session -d -s splot 'docker-compose run --rm base python molecule_history.py --fo2-file friends_smokes'
```

## DiGress

<https://arxiv.org/abs/2209.14734>

```bash
tmux new-session -d -s sdd8 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=discrete general.gpus=2 max_num_atoms=8"'
tmux new-session -d -s scc8 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=continuous general.gpus=2 max_num_atoms=8"'

tmux new-session -d -s sdd12 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=discrete general.gpus=2 max_num_atoms=12"'
tmux new-session -d -s scc12 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=continuous general.gpus=2 max_num_atoms=12"'

tmux new-session -d -s sdd15 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=discrete general.gpus=3 max_num_atoms=15"'
tmux new-session -d -s scc15 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=continuous general.gpus=3 max_num_atoms=15"'

tmux new-session -d -s sdd20 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=discrete general.gpus=3 max_num_atoms=20"'
tmux new-session -d -s scc20 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=continuous general.gpus=3 max_num_atoms=20"'

tmux new-session -d -s sdd25 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=discrete general.gpus=0 max_num_atoms=25"'
tmux new-session -d -s scc25 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=continuous general.gpus=0 max_num_atoms=25"'

tmux new-session -d -s sdd30 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=discrete general.gpus=1 max_num_atoms=30"'
tmux new-session -d -s scc30 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=continuous general.gpus=2 max_num_atoms=30"'
```

tmux new-session -d -s fs6 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=continuous general.gpus=2 dataset.name=fo2 filename=friends_smokes_p5"'
tmux new-session -d -s fs6 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=continuous general.gpus=2 dataset.name=fo2 filename=friends_smokes_p6"'
tmux new-session -d -s fs7 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=continuous general.gpus=2 dataset.name=fo2 filename=friends_smokes_p7"'
tmux new-session -d -s fs8 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=continuous general.gpus=2 dataset.name=fo2 filename=friends_smokes_p8"'
tmux new-session -d -s fs9 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=continuous general.gpus=2 dataset.name=fo2 filename=friends_smokes_p9"'
tmux new-session -d -s fs10 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=continuous general.gpus=2 dataset.name=fo2 filename=friends_smokes_p10"'

For generations at the end, modify `DiGress/configs/general/general_default.yaml` and run again.

## Moler

cp moler_patched_train.py ~/.pyenv/versions/3.8.3/lib/python3.8/site-packages/molecule_generation/cli/train.py

<https://pypi.org/project/molecule-generation/>

```bash
cp data/molecules/size_8/train.smi data/molecules/size_8/train.smiles && \
cp data/molecules/size_8/train.smi data/molecules/size_8/valid.smiles && \
cp data/molecules/size_8/train.smi data/molecules/size_8/test.smiles
```

```bash
dc run --rm base molecule_generation preprocess data/molecules/size_8/ data/molecules/size_8_moler/ data/molecules/size_8_moler_tracke/ && \
dc run --rm base molecule_generation preprocess data/molecules/size_30/ data/molecules/size_30_moler/ data/molecules/size_30_moler_tracke/
```

```bash
dc run --rm base molecule_generation train MoLeR data/molecules/size_8_moler_tracke/ --save-dir output_v2_moler_8 && \
dc run --rm base molecule_generation train MoLeR data/molecules/size_12_moler_tracke/ --save-dir output_moler_12 && \
dc run --rm base molecule_generation train MoLeR data/molecules/size_15_moler_tracke/ --save-dir output_moler_15 && \
dc run --rm base molecule_generation train MoLeR data/molecules/size_20_moler_tracke/ --save-dir output_moler_20 && \
dc run --rm base molecule_generation train MoLeR data/molecules/size_25_moler_tracke/ --save-dir output_moler_25 && \
dc run --rm base molecule_generation train MoLeR data/molecules/size_30_moler_tracke/ --save-dir output_moler_30 && \
```

```bash
dc run --rm base molecule_generation sample output_moler_8 100000 > output_moler_8/generated_smiles.the-end.txt && \
dc run --rm base molecule_generation sample output_moler_12 100000 > output_moler_12/generated_smiles.the-end.txt && \
dc run --rm base molecule_generation sample output_moler_15 100000 > output_moler_15/generated_smiles.the-end.txt && \
dc run --rm base molecule_generation sample output_moler_20 100000 > output_moler_20/generated_smiles.the-end.txt
dc run --rm base molecule_generation sample output_moler_25 100000 > output_moler_25/generated_smiles.the-end.txt
dc run --rm base molecule_generation sample output_moler_30 100000 > output_moler_30/generated_smiles.the-end.txt
```

## DEG

```bash
conda activate DEG
python main.py --training_data="../data/molecules/size_8/train.smiles" --output_dir output_deg_8
```

```bash
python retro_star_listener.py --filenames "output_deg_8/generated_samples.txt" --output_filenames "output_deg_8/output_syn.txt"
```

## RNN

```bash
dc run --rm base python Molecule-RNN/vocab/chembl_selfies_vocab.py /app/data/molecules/size_8/train.smi /app/data/molecules/size_8/selfies.rnn.vocab && \
dc run --rm base python Molecule-RNN/vocab/chembl_regex_vocab.py /app/data/molecules/size_8/train.smi /app/data/molecules/size_8/regex.rnn.vocab && \
dc run --rm base python Molecule-RNN/vocab/chembl_char_vocab.py /app/data/molecules/size_8/train.smi /app/data/molecules/size_8/char.rnn.vocab
```

```bash
dc run --rm base python Molecule-RNN/train.py /app/data/molecules/size_8/train.smi selfies /app/data/molecules/size_8/selfies.rnn.vocab /app/output_rnn_8_selfies/ && \
dc run --rm base python Molecule-RNN/train.py /app/data/molecules/size_8/train.smi regex /app/data/molecules/size_8/regex.rnn.vocab /app/output_rnn_8_regex/ && \
dc run --rm base python Molecule-RNN/train.py /app/data/molecules/size_8/train.smi char /app/data/molecules/size_8/char.rnn.vocab /app/output_rnn_8_char/ && \

dc run --rm base python Molecule-RNN/train.py /app/data/molecules/size_12/train.smi selfies /app/data/molecules/size_12/selfies.rnn.vocab /app/output_rnn_12_selfies/ && \
dc run --rm base python Molecule-RNN/train.py /app/data/molecules/size_12/train.smi regex /app/data/molecules/size_12/regex.rnn.vocab /app/output_rnn_12_regex/ && \
dc run --rm base python Molecule-RNN/train.py /app/data/molecules/size_12/train.smi char /app/data/molecules/size_12/char.rnn.vocab /app/output_rnn_12_char/ && \

dc run --rm base python Molecule-RNN/train.py /app/data/molecules/size_15/train.smi selfies /app/data/molecules/size_15/selfies.rnn.vocab /app/output_rnn_15_selfies/ && \
dc run --rm base python Molecule-RNN/train.py /app/data/molecules/size_15/train.smi regex /app/data/molecules/size_15/regex.rnn.vocab /app/output_rnn_15_regex/ && \
dc run --rm base python Molecule-RNN/train.py /app/data/molecules/size_15/train.smi char /app/data/molecules/size_15/char.rnn.vocab /app/output_rnn_15_char/ && \

dc run --rm base python Molecule-RNN/train.py /app/data/molecules/size_20/train.smi selfies /app/data/molecules/size_20/selfies.rnn.vocab /app/output_rnn_20_selfies/ && \
dc run --rm base python Molecule-RNN/train.py /app/data/molecules/size_20/train.smi regex /app/data/molecules/size_20/regex.rnn.vocab /app/output_rnn_20_regex/ && \
dc run --rm base python Molecule-RNN/train.py /app/data/molecules/size_20/train.smi char /app/data/molecules/size_20/char.rnn.vocab /app/output_rnn_20_char/ && \
```

```bash
dc run --rm base python Molecule-RNN/sample.py /app/output_rnn_20_selfies/ 100000 && \
dc run --rm base python Molecule-RNN/sample.py /app/output_rnn_20_regex/ 100000 && \
dc run --rm base python Molecule-RNN/sample.py /app/output_rnn_20_char/ 100000
```

## PaccMann Chemistry VAE

<https://github.com/PaccMann/paccmann_chemistry/tree/master>

```bash
dc run --rm base bash
conda init && source ~/.bashrc && conda activate paccmann_chemistry && pip install tensorboard
python paccmann_chemistry/examples/train_vae.py \
    /app/data/molecules/size_30/train.smi \
    /app/data/molecules/size_30/valid.smi \
    "none" \
    output_paccmann_vae_30 \
    /app/paccmann_chemistry/examples/example_params.json \
    paccmann_vae_30
```

## CCGVAE -- not working

```bash
dc run --rm base bash
conda init && source ~/.bashrc && conda activate ccgvae

export PYTHONPATH="/app/ccgvae"
cd ccgvae

python make_dataset.py 8
python CCGVAE.py --size 8
```
