import typer
import json
import random
import shutil
from pathlib import Path
from rdkit import Chem, RDLogger
from nmln import ontology, handler
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


def main(atom_size: int, input_file: str):
    """
    dc run --rm base python data_creation/fo2_sampled_to_our_input_format.py 8 data-fo2/hydrocarbons-alkanes_atomsize8.txt
    dc run --rm base python data_creation/fo2_sampled_to_our_input_format.py 11 data-fo2/hydrocarbons-alkanes_atomsize11.txt
    dc run --rm base python data_creation/fo2_sampled_to_our_input_format.py 14 data-fo2/hydrocarbons-alkanes_atomsize14.txt
    dc run --rm base python data_creation/fo2_sampled_to_our_input_format.py 17 data-fo2/hydrocarbons-alkanes_atomsize17.txt
    """
    assert (
        f"atomsize{atom_size}" in input_file
    ), f"atom_size {atom_size} not in {input_file}"
    name = input_file.split("/")[-1].split(".")[0]
    ontology_path = (
        "data_creation/ontology_" + input_file.split("/")[-1].split("_atomsize")[0]
    )
    with open(ontology_path, "r") as f:
        print(f"Loaded ontology from {ontology_path}")
        print(f.read())
        # if input("Continue? [y/n]: ").strip() != "y":
        #     return
    output_dir = Path("data/molecules") / name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    invalid_samples_file = input_file.replace(".txt", ".invalid.txt")
    val_smiles_file = output_dir / "valid.smi"
    train_smiles_file = output_dir / "train.smi"
    test_smiles_file = output_dir / "test.smi"

    constants = [str(i) for i in range(atom_size)]
    domain = ontology.Domain(name="atoms", constants=constants)
    predicates = [
        ontology.Predicate(p, domains=[domain for _ in range(a)])
        for p, a in ontology.load_predicates(ontology_path).items()
    ]
    ontology_ = ontology.Ontology(domains=[domain], predicates=predicates)
    handler_initialized = handler.MoleculesHandler(atom_size, ontology_)

    seen = set()

    valid_total = 0
    valid_train, valid_val = 0, 0

    with open(train_smiles_file, "w") as ftrain, open(
        val_smiles_file, "w"
    ) as fval, open(test_smiles_file, "w") as ftest, open(
        invalid_samples_file, "w"
    ) as finvalid:
        with open(input_file, "r") as f:
            for idx, line in tqdm(enumerate(f)):
                data = [
                    x.replace("atomsize", "") + "."
                    for x in json.loads(line.replace("'", '"'))
                    if not x.lower().startswith(
                        "aux"
                    )  # These are just for sampler, ignore here.
                ]

                lin = ontology_.file_content_to_linearState(data)
                mols = handler_initialized.fromLin2Mol([lin])

                if not mols:
                    finvalid.write(f"{line.strip()}\n")
                    continue

                try:
                    our_smile = Chem.CanonSmiles(Chem.MolToSmiles(mols[0]))
                except Exception as e:
                    finvalid.write(f"{line.strip()}\n")
                    print(f"Error in creating smiles from mol: {e}")
                    continue

                valid_total += 1
                if our_smile in seen:
                    continue
                seen.add(our_smile)

                split = "train" if random.random() < 0.8 else "val"
                (output_dir / split).mkdir(parents=True, exist_ok=True)

                with open(output_dir / split / f"{idx}.mol", "w") as f:
                    for x in data:
                        f.write(f"{x}\n")

                if split == "train":
                    ftrain.write(f"{our_smile}\n")
                    valid_train += 1
                if split in ["val", "test"]:
                    fval.write(f"{our_smile}\n")
                    ftest.write(f"{our_smile}\n")
                    valid_val += 1

    print(
        f"Valid train: {valid_train}, valid val: {valid_val}, valid total (without dedup): {valid_total}"
    )


if __name__ == "__main__":
    typer.run(main)
