import os
import shutil
import random
import typer
import typing as t
import pytorch_lightning as pl

from tqdm import tqdm
from pathlib import Path
from multiprocessing import Process
from data_creation import chembl_to_fol_and_smiles, infer_skipbonds


def main(
    atom_size: int = 8,
    out_folder: t.Optional[Path] = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    add_skipbonds: bool = False,
):
    pl.seed_everything(0)

    out_folder = out_folder or Path(
        f"/home/jungpete/projects/shiny-happiness/data/molecules/size_{atom_size}"
    )
    tmp_folder = Path(f"{str(out_folder)}_tmp")
    train_folder = out_folder / "train"
    val_folder = out_folder / "val"
    test_folder = out_folder / "test"

    val_smiles_file = out_folder / "valid.smi"
    train_smiles_file = out_folder / "train.smi"
    test_smiles_file = out_folder / "test.smi"

    while True:
        try:
            tmp_folder.mkdir(parents=True, exist_ok=False)
            train_folder.mkdir(parents=True, exist_ok=False)
            val_folder.mkdir(parents=True, exist_ok=False)
            test_folder.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            if input("Remove existing? [y/n]: ").strip() == "y":
                try:
                    shutil.rmtree(out_folder)
                except FileNotFoundError:
                    pass
                try:
                    shutil.rmtree(tmp_folder)
                except FileNotFoundError:
                    pass
                continue
            else:
                raise
        else:
            break

    filename_to_smile = chembl_to_fol_and_smiles.main(
        tmp_folder,
        atom_size=atom_size,
        smiles_dir=out_folder,
    )

    with open(val_smiles_file, "w") as fval, open(
        train_smiles_file, "w"
    ) as ftrain, open(test_smiles_file, "w") as ftest:
        for entry in tqdm(tmp_folder.glob("*")):
            file_name = os.path.basename(entry)
            smile = filename_to_smile[file_name]
            ratio = random.random()
            out, smiles_out = (
                (val_folder, fval)
                if 0 <= ratio < val_ratio
                else (
                    (test_folder, ftest)
                    if val_ratio <= ratio < val_ratio + test_ratio
                    else (train_folder, ftrain)
                )
            )
            smiles_out.write(f"{smile}\n")
            # Launch in process because some occasional Prolog's fatal error can not be catched by except Exception block.
            work = Process(
                target=infer_skipbonds.main,
                args=(file_name, tmp_folder, out, add_skipbonds),
            )
            work.start()
            work.join()

    shutil.rmtree(tmp_folder)


if __name__ == "__main__":
    typer.run(main)
