import typer
import typing as t
import pandas as pd

from collections import Counter
from tqdm import tqdm
from pathlib import Path
from rdkit import Chem
from nmln import ontology, handler


def main(
    save_folder: Path,
    atom_size: int,
    bond_types: t.List[str] = ["single", "double", "skipBond", "skipSkipBond"],
    atom_types: t.List[str] = ["c", "n", "o", "s", "p", "cl", "f"],
    smiles_dir: t.Optional[Path] = None,
):
    # Download from http://chembl.blogspot.com/2020/03/chembl-26-released.html.
    df = pd.read_csv("data/chembl_26_chemreps.txt", sep="\t")
    df = df["canonical_smiles"]
    data = df.values

    smiles_list = []
    smile_to_idx = {}
    smile_to_write_id = {}

    constants = [str(i) for i in range(atom_size)]
    domain = ontology.Domain(name="atoms", constants=constants)
    predicates = [
        ontology.Predicate(p, domains=[domain for _ in range(a)])
        for p, a in ontology.load_predicates("data_creation/ontology").items()
    ]
    ontology_ = ontology.Ontology(domains=[domain], predicates=predicates)
    handler_initialized = handler.MoleculesHandler(atom_size, ontology_)
    filename_to_smile = {}

    for i, smile in enumerate(tqdm(data)):
        try:
            if "+" in smile or "-" in smile:
                continue

            smile = Chem.CanonSmiles(smile)
            mol = Chem.MolFromSmiles(smile)
            num = mol.GetNumAtoms()

            if num == atom_size:
                if "AROMATIC" not in bond_types:
                    Chem.Kekulize(mol, clearAromaticFlags=True)

                smile = Chem.CanonSmiles(Chem.MolToSmiles(mol))

                to_write = []

                for a in mol.GetAtoms():
                    sym = str(a.GetSymbol()).lower()

                    if sym not in atom_types:
                        to_write = []
                        break

                    else:
                        aid = a.GetIdx()
                        to_write.append("%s(%d)." % (sym.lower(), aid))

                if len(to_write) == 0:
                    continue

                for b in mol.GetBonds():
                    sid = b.GetBeginAtom().GetIdx()
                    eid = b.GetEndAtom().GetIdx()
                    type = str(b.GetBondType()).lower()

                    if type not in bond_types:
                        to_write = []
                        break

                    else:
                        to_write.append(f"{type.lower()}({sid},{eid}).")
                        to_write.append(f"{type.lower()}({eid},{sid}).")

                to_write = sorted(to_write)
                to_write_id = "".join(to_write)
                smile_to_write_id[to_write_id] = smile_to_write_id.get(
                    to_write_id, len(smile_to_write_id)
                )

                filename = f"{smile_to_write_id[to_write_id]}.mol"
                filepath = save_folder / filename

                our_smile = Chem.CanonSmiles(
                    Chem.MolToSmiles(
                        handler_initialized.fromLin2Mol(
                            [ontology_.file_content_to_linearState(to_write)]
                        )[0]
                    )
                )

                if filepath.exists() or our_smile in smile_to_idx:
                    continue

                smile_to_idx[our_smile] = smile_to_idx.get(our_smile, len(smile_to_idx))

                if len(to_write) > 0:
                    filename_to_smile[filename] = our_smile
                    with open(filepath, "w") as f:
                        for s in to_write:
                            f.write(s + "\n")

                    smiles_list.append(smile + "\n")

        except Exception as e:
            print(e)

    if smiles_dir is not None:
        with open(smiles_dir / f"smile{atom_size}", "w") as smiles_file:
            for line in smiles_list:
                smiles_file.write(line)

    print(Counter(smiles_list).most_common(5))
    return filename_to_smile


if __name__ == "__main__":
    typer.run(main)
