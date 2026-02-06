#!/usr/bin/env/python
"""
Usage:
    make_dataset.py [options]

Options:
    -h --help           Show this screen.
    --dataset NAME      QM9 or ZINC
"""

import shutil
import sys
import json
import os
import sys

from pprint import pprint
import numpy as np
from docopt import docopt
from rdkit import Chem
from rdkit.Chem import QED

import utils
from utils import build_dataset_info, read_our_data

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# get current directory in order to work with full path and not dynamic
current_dir = os.path.dirname(os.path.realpath(__file__))


def train_valid_split(dataset):
    n_mol_out = 0

    # save the train, valid dataset.
    raw_data = {"train": [], "valid": [], "test": []}
    file_count = 0

    for split, split_smiles in dataset.items():
        for i, smiles in enumerate(split_smiles):
            val = QED.qed(Chem.MolFromSmiles(smiles))
            hist = make_hist(smiles)
            if hist is not None:
                raw_data[split].append(
                    {"smiles": smiles, "QED": val, "hist": hist.tolist()}
                )
                file_count += 1
                if file_count % 1000 == 0:
                    print("Finished reading: %d" % file_count, end="\r")
            else:
                n_mol_out += 1

    print("Number of molecules left out: ", n_mol_out)
    return raw_data


def make_hist(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atoms = mol.GetAtoms()
    hist = np.zeros(dataset_info["hist_dim"])
    for atom in atoms:
        if dataset == "qm9":
            atom_str = atom.GetSymbol()
        else:
            # zinc dataset # transform using "<atom_symbol><valence>(<charge>)"  notation
            symbol = atom.GetSymbol()
            valence = atom.GetTotalValence()
            charge = atom.GetFormalCharge()
            atom_str = "%s%i(%i)" % (symbol, valence, charge)

            if atom_str not in dataset_info["atom_types"]:
                print("Unrecognized atom type %s" % atom_str)
                return None

        ind = dataset_info["atom_types"].index(atom_str)
        val = dataset_info["maximum_valence"][ind]
        hist[
            val - 1
        ] += 1  # in the array the valence number start from 1, instead the array start from 0
    return hist


def preprocess(size, raw_data, dataset):
    print("Parsing smiles as graphs...")
    processed_data = {"train": [], "valid": [], "test": []}

    file_count = 0
    for section in ["train", "valid", "test"]:
        all_smiles = []  # record all smiles in training dataset
        for i, (smiles, QED, hist) in enumerate(
            [(mol["smiles"], mol["QED"], mol["hist"]) for mol in raw_data[section]]
        ):
            nodes, edges = utils.to_graph(smiles, dataset)
            if len(edges) <= 0:
                print("Error. Molecule with len(edges) <= 0")
                continue
            processed_data[section].append(
                {
                    "targets": [[QED]],
                    "graph": edges,
                    "node_features": nodes,
                    "smiles": smiles,
                    "hist": hist,
                }
            )
            all_smiles.append(smiles)
            if file_count % 1000 == 0:
                print("Finished processing: %d" % file_count, end="\r")
            file_count += 1
        print("%s: 100 %%                   " % (section))
        with open(
            "/home/jungpete/projects/shiny-happiness/data/molecules/size_%s/ccgvae_molecules_%s.json"
            % (size, section),
            "w",
        ) as f:
            json.dump(processed_data[section], f)

    print("Train molecules = " + str(len(processed_data["train"])))
    print("Valid molecules = " + str(len(processed_data["valid"])))
    print("Test molecules = " + str(len(processed_data["test"])))


if __name__ == "__main__":
    size = sys.argv[1]
    print(size)
    dataset = "/home/jungpete/projects/shiny-happiness/data/molecules/size_%s" % size

    print("Reading dataset: " + str(dataset))
    data = []

    data = read_our_data(dataset)

    dataset_info = build_dataset_info(data, size)
    pprint(dataset_info)
    raw_data = train_valid_split(data)
    preprocess(size, raw_data, dataset)
