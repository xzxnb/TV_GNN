# Copyright: Wentao Shi, 2021
import typer
from dataloader import SELFIEVocab, RegExVocab, CharVocab
from model import RNN
import argparse
import torch
import yaml
import selfies as sf
from tqdm import tqdm
from rdkit import Chem

# suppress rdkit error
from rdkit import rdBase
from dataloader import dataloader_gen, none_on_exp

rdBase.DisableLog("rdApp.error")


def main(
    result_dir: str,
    n: int,
    batch_size: int = 2048,
    filename: str = "generated_samples.the-end.txt",
    filemode: str = "w",
):
    result_dir = result_dir.rstrip("/") + "/"
    num_batches = n // batch_size

    # load the configuartion file in output
    config_dir = result_dir + "config.yaml"
    with open(config_dir, "r") as f:
        config = yaml.full_load(f)

    # detect cpu or gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # load vocab
    which_vocab, vocab_path = config["which_vocab"], config["vocab_path"]

    if which_vocab == "selfies":
        vocab = SELFIEVocab(vocab_path)
    elif which_vocab == "regex":
        vocab = RegExVocab(vocab_path)
    elif which_vocab == "char":
        vocab = CharVocab(vocab_path)
    else:
        raise ValueError("Wrong vacab name for configuration which_vocab!")

    # load model
    rnn_config = config["rnn_config"]
    model = RNN(rnn_config).to(device)
    model.load_state_dict(
        torch.load(result_dir + "trained_model.pt", map_location=torch.device(device))
    )
    model.eval()

    # sample, filter out invalid molecules, and save the valid molecules
    out_file = open(result_dir + filename, filemode)
    num_valid, num_invalid = 0, 0
    for _ in tqdm(range(num_batches)):
        # sample molecules as integers
        sampled_ints = model.sample(batch_size=batch_size, vocab=vocab, device=device)

        # convert integers back to SMILES
        molecules = []
        sampled_ints = sampled_ints.tolist()
        for ints in sampled_ints:
            molecule = []
            for x in ints:
                if vocab.int2tocken[x] == "<eos>":
                    break
                else:
                    molecule.append(vocab.int2tocken[x])
            molecules.append("".join(molecule))

        # convert SELFIES back to SMILES
        if vocab.name == "selfies":
            molecules = [
                sf.decoder(x)
                for x in molecules
                if none_on_exp(lambda: sf.decoder(x)) is not None
            ]

        # save the valid sampled SMILES to output file,
        for smiles in molecules:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    num_invalid += 1
                else:
                    num_valid += 1
                    out_file.write(Chem.CanonSmiles(smiles) + "\n")
            except:
                num_valid += 1
                pass

    # and compute the valid rate
    print(
        "sampled {} valid SMILES out of {}, success rate: {}".format(
            num_valid, num_valid + num_invalid, num_valid / (num_valid + num_invalid)
        )
    )


if __name__ == "__main__":
    typer.run(main)
