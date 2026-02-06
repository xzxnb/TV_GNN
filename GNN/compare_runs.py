import typer
import mlflow
import logging
import typing as t

from rdkit import Chem
from nmln import dataset, ontology, handler
from mlflow_to_matplotlib import main as mlflow_to_matplotlib_main
from joblib import Memory

location = "./cachedir"
memory = Memory(location, verbose=0)


def main(
    max_num_atoms: int,
    runs: t.List[str],
) -> None:
    client = mlflow.tracking.MlflowClient()

    (
        train_canons,
        val_canons,
        cleared_val_canons,
        removed_from_val_perc,
    ) = get_clear_val_canon_smiles(max_num_atoms, "data/molecules/ontology")

    if runs[0] == "exit":
        return

    run_names = []
    run_to_ratio_generated_in_train = {}
    run_to_ratio_generated_in_val = {}
    run_to_cleared_perc_generated = {}
    run_to_not_cleared_perc_generated = {}

    for run_id in runs:
        run = client.get_run(run_id)
        run_name = run.data.tags["mlflow.runName"]
        metric = client.get_metric_history(run_id, "val/ratio_generated_in_val")

        if max_num_atoms != int(run.data.params["max_num_atoms"]):
            raise RuntimeError(
                f"max_num_atoms does not match, {max_num_atoms} != {run.data.params['max_num_atoms']}."
            )

        dst_path = client.download_artifacts(
            run_id=run_id, path="all_generated_canon_smiles.txt", dst_path="/tmp/"
        )

        with open(dst_path) as f:
            generated_canons = set(line.strip() for line in f if line.strip())

        run_names.append(run_name)
        run_to_ratio_generated_in_val[run_name] = metric
        run_to_cleared_perc_generated[run_name] = len(
            generated_canons & cleared_val_canons
        ) / len(cleared_val_canons)
        run_to_not_cleared_perc_generated[run_name] = len(
            generated_canons & val_canons
        ) / len(val_canons)
        run_to_ratio_generated_in_train[run_name] = len(
            generated_canons & train_canons
        ) / len(train_canons)

    min_achieved_step = min(
        [
            max(m.step for m in metric)
            for metric in run_to_ratio_generated_in_val.values()
        ]
    )
    half_achieved_step = min_achieved_step // 2
    quart_achieved_step = min_achieved_step // 4

    run_to_ratio_generated_in_val_same_epoch = {}
    run_to_ratio_generated_in_val_half_same_epoch = {}
    run_to_ratio_generated_in_val_quart_same_epoch = {}

    for key in run_to_ratio_generated_in_val.keys():
        run_to_ratio_generated_in_val_same_epoch[key] = list(
            filter(
                lambda m: m.step <= min_achieved_step,
                run_to_ratio_generated_in_val[key],
            )
        )
        run_to_ratio_generated_in_val_half_same_epoch[key] = list(
            filter(
                lambda m: m.step <= half_achieved_step,
                run_to_ratio_generated_in_val[key],
            )
        )
        run_to_ratio_generated_in_val_quart_same_epoch[key] = list(
            filter(
                lambda m: m.step <= quart_achieved_step,
                run_to_ratio_generated_in_val[key],
            )
        )

    run_names = sorted(
        run_names, key=lambda k: run_to_ratio_generated_in_val_same_epoch[k][-1].value
    )

    print(f"Step = {min_achieved_step}, half = {half_achieved_step}")
    for key in run_names:
        took = (
            (
                run_to_ratio_generated_in_val_same_epoch[key][-1].timestamp
                - run_to_ratio_generated_in_val_same_epoch[key][0].timestamp
            )
            / 1000
            / 60
            / 60
            / 24
        )
        print(
            f"{key}: "
            f"test: {run_to_ratio_generated_in_val_same_epoch[key][-1].value:.4f} "
            f"train: {run_to_ratio_generated_in_train[key]:.4f} "
            f"took: {took:.2f} days "
            # f"/ {run_to_ratio_generated_in_val_half_same_epoch[key][-1].value:.2f} "
            # f"/ {run_to_ratio_generated_in_val_quart_same_epoch[key][-1].value:.2f} "
            # f"/ {run_to_cleared_perc_generated[key]:.2f} "
            # f"/ {run_to_not_cleared_perc_generated[key]:.2f}"
            # f"/ {run_to_ratio_generated_in_val[key][-1].value:.2f} "
        )

    mlflow_to_matplotlib_main(runs, filename=f"{max_num_atoms}")


# @memory.cache
def get_clear_val_canon_smiles(
    max_num_atoms: int,
    ontology_path: str,
    base_mol_dir: str = None,
) -> t.Tuple[t.Set[str], t.Set[str], t.Set[str], float]:
    constants = [str(i) for i in range(max_num_atoms)]
    domain = ontology.Domain(name="atoms", constants=constants)
    predicates = [
        ontology.Predicate(p, domains=[domain for _ in range(a)])
        for p, a in ontology.load_predicates(ontology_path).items()
    ]
    ontology_ = ontology.Ontology(domains=[domain], predicates=predicates)
    handler_initialized = handler.MoleculesHandler(max_num_atoms, ontology_)

    base_mol_dir = (
        base_mol_dir
        or f"/home/jungpete/projects/nmln-torch/data/molecules/training{max_num_atoms}"
    )
    train_ds = dataset.Dataset(
        constants=constants,
        domain=domain,
        predicates=predicates,
        ontology=ontology_,
        mol_dir=f"{base_mol_dir}/train",
    )
    val_ds = dataset.Dataset(
        constants=constants,
        domain=domain,
        predicates=predicates,
        ontology=ontology_,
        mol_dir=f"{base_mol_dir}/val",
    )

    train_mols = handler_initialized.fromLin2Mol(train_ds.y)
    val_mols = handler_initialized.fromLin2Mol(val_ds.y)

    train_canons = set(Chem.CanonSmiles(Chem.MolToSmiles(mol)) for mol in train_mols)
    val_canons = set(Chem.CanonSmiles(Chem.MolToSmiles(mol)) for mol in val_mols)

    cleared_val_canons = val_canons - train_canons
    removed_perc = (len(val_canons) - len(cleared_val_canons)) / len(val_canons)

    logging.warning(
        f"{len(train_ds)} in train set,"
        f" {len(val_canons)} in val set"
        f", but only {len(cleared_val_canons)} is left after clearing"
        f", that's {removed_perc * 100:.2f}% removed."
    )

    return train_canons, val_canons, cleared_val_canons, removed_perc


if __name__ == "__main__":
    typer.run(main)
