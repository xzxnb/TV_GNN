import re
import os
import pickle
import typing as t
import typer
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it
from typing import Union, Dict
from rdkit import Chem, RDLogger
from tqdm import tqdm
from math import ceil, sqrt
from pathlib import Path
from collections import Counter, defaultdict
from compare_runs import get_clear_val_canon_smiles
from pprint import pprint

RDLogger.DisableLog("rdApp.*")
plt.rcParams["text.usetex"] = True
DEBUG = False
ANONYMIZE = True


def create_zip(files, zip_name):
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            zipf.write(file)


MOLECULES = {
    8: {
        "DiGress/outputs/seed0-v2-continuous-8-_nparams1058188_generated_smiles.in-training.txt": "DiGress continous tiny",
        # "DiGress/outputs/seed0-v2-continuous-8-_nparams1946124_generated_smiles.in-training.txt": "DiGress continous small",
        # "DiGress/outputs/seed0-v2-continuous-8-_nparams3721996_generated_smiles.in-training.txt": "DiGress continous medium",
        # "DiGress/outputs/seed0-v2-continuous-8-_generated_smiles.in-training.txt": "DiGress continous big (original)",
        # "DiGress/outputs/seed0-v2-discrete-8-_generated_smiles.in-training.txt": "DiGress discrete big (original)",
        # "DiGress/outputs/seed0-v2-discrete-8-_nparams2837644_generated_smiles.in-training.txt": "DiGress discrete medium (original)",
        # # "DiGress/outputs/v2-discrete-8-_generated_smiles.in-training.txt", # DiGress discrete big (original) (older training)
        # "DiGress/outputs/seed1-v2-continuous-8-_generated_smiles.in-training.txt": "Digress continuous big (original), seed 1",
        # "DiGress/outputs/seed2-v2-continuous-8-_generated_smiles.in-training.txt": "Digress continuous big (original), seed 2",
        # "DiGress/outputs/seed0-v2-discrete-8-_nparams1061772_generated_smiles.in-training.txt": "DiGress discrete small",
        "output_moler_8/generated_smiles.txt": "Moler",
        "output_paccmann_vae_8/paccmann_vae_8/generated_molecules.txt": "PaccMann",
        "output_rnn_8_char/generated_samples.txt": "RNN Char",
        "output_rnn_8_regex/generated_samples.txt": "RNN Regex",
        "output_rnn_8_selfies/generated_samples.txt": "RNN Selfies",
    },
    12: {
        "DiGress/outputs/seed0-v2-continuous-12-_nparams1058188_generated_smiles.in-training.txt": "DiGress continous tiny",
        # "DiGress/outputs/seed0-v2-continuous-12-_nparams1946124_generated_smiles.in-training.txt": "DiGress continous small",
        # "DiGress/outputs/seed0-v2-continuous-12-_nparams3721996_generated_smiles.in-training.txt": "DiGress continous medium",
        # "DiGress/outputs/seed0-v2-continuous-12-_generated_smiles.in-training.txt": "DiGress continous big (original)",
        # "DiGress/outputs/seed1-v2-continuous-12-_generated_smiles.in-training.txt": "Digress continuous big (original), seed 1",
        # "DiGress/outputs/seed2-v2-continuous-12-_generated_smiles.in-training.txt": "Digress continuous big (original), seed 2",
        # "DiGress/outputs/seed0-v2-discrete-12-_generated_smiles.in-training.txt": "DiGress discrete big (original)",
        # "DiGress/outputs/seed0-v2-discrete-12-_nparams1061772_generated_smiles.in-training.txt": "DiGress discrete small",
        "output_moler_12/generated_smiles.txt": "Moler",
        "output_paccmann_vae_12/paccmann_vae_12/generated_molecules.txt": "PaccMann",
        "output_rnn_12_regex/generated_samples.txt": "RNN Regex",
        "output_rnn_12_selfies/generated_samples.txt": "RNN Selfies",
        "output_rnn_12_char/generated_samples.txt": "RNN Char",
    },
    15: {
        "DiGress/outputs/seed0-v2-continuous-15-_nparams1058188_generated_smiles.in-training.txt": "DiGress continous tiny",
        # "DiGress/outputs/seed1-v2-continuous-15-_generated_smiles.in-training.txt": "DiGress continous big (original) (original), seed 1",
        # "DiGress/outputs/seed2-v2-continuous-15-_generated_smiles.in-training.txt": "DiGress continous big (original) (original), seed 2",
        # "DiGress/outputs/seed0-v2-continuous-15-_nparams3721996_generated_smiles.in-training.txt": "DiGress continous medium",
        # "DiGress/outputs/seed0-v2-continuous-15-_nparams1946124_generated_smiles.in-training.txt": "DiGress continous small",
        # "DiGress/outputs/seed0-v2-discrete-15-_nparams1061772_generated_smiles.in-training.txt": "DiGress discrete small",
        # "DiGress/outputs/seed0-v2-discrete-15-_generated_smiles.in-training.txt": "DiGress discrete big (original)",
        # "DiGress/outputs/seed0-v2-continuous-15-_generated_smiles.in-training.txt": "DiGress continous big (original)",
        "output_moler_15/generated_smiles.txt": "Moler",
        "output_paccmann_vae_15/paccmann_vae_15/generated_molecules.txt": "PaccMann",
        "output_rnn_15_selfies/generated_samples.txt": "RNN Selfies",
        "output_rnn_15_char/generated_samples.txt": "RNN Char",
        "output_rnn_15_regex/generated_samples.txt": "RNN Regex",
    },
    20: {
        # "DiGress/outputs/seed0-v2-discrete-20-_generated_smiles.in-training.txt": "DiGress discrete big (original)",
        # "DiGress/outputs/seed0-v2-discrete-20-_nparams1061772_generated_smiles.in-training.txt": "DiGress discrete small",
        # "DiGress/outputs/seed2-v2-continuous-20-_generated_smiles.in-training.txt": "DiGress continuous big (original), seed 2",
        # "DiGress/outputs/seed0-v2-continuous-20-_nparams1946124_generated_smiles.in-training.txt": "DiGress continous small",
        # "DiGress/outputs/seed0-v2-continuous-20-_generated_smiles.in-training.txt": "DiGress continous big (original)",
        # "DiGress/outputs/seed1-v2-continuous-20-_generated_smiles.in-training.txt": "Digress continuous big (original), seed 1",
        # "DiGress/outputs/seed0-v2-continuous-20-_nparams3721996_generated_smiles.in-training.txt": "DiGress continous medium",
        "DiGress/outputs/seed0-v2-continuous-20-_nparams1058188_generated_smiles.in-training.txt": "DiGress continous tiny",
        "output_moler_20/generated_smiles.txt": "Moler",
        "output_paccmann_vae_20/paccmann_vae_20/generated_molecules.txt": "PaccMann",
        "output_rnn_20_selfies/generated_samples.txt": "RNN Selfies",
        "output_rnn_20_regex/generated_samples.txt": "RNN Regex",
        "output_rnn_20_char/generated_samples.txt": "RNN Char",
    },
}

NAME_TO_ANON_NAME = {
    "DiGress continous tiny": "Method 1",
    "Moler": "Method 2",
    "PaccMann": "Method 3",
    "RNN Char": "Method 4",
    "RNN Regex": "Method 5",
    "RNN Selfies": "Method 6",
}

# for max_num_atoms, filtered_files_to_names in MOLECULES.items():
#     create_zip(
#         list(filtered_files_to_names.keys())
#         + [
#             f"data/molecules/size_{max_num_atoms}/train.smi",
#             f"data/molecules/size_{max_num_atoms}/valid.smi",
#         ],
#         f"atoms{max_num_atoms}.zip",
#     )

#     print(f"Max num atoms: {max_num_atoms}")
#     for file, name in filtered_files_to_names.items():
#         print(f"   {name}")

# exit()


def main(
    log: bool = False,
    fo2_file: Union[str, None] = None,
    molecule_dir: Union[str, None] = None,
):
    print("Here")
    Path("report").mkdir(exist_ok=True, parents=True)
    all_generated_files = (
        # list(Path(".").rglob(f"*generated_smiles.txt"))
        # + list(Path(".").rglob(f"*generated_molecules.txt"))
        # + list(Path(".").rglob(f"*generated_samples.txt"))
        # list(Path(".").rglob(f"*generated_smiles.in-training.txt"))
        # + list(Path(".").rglob(f"*generated_samples.in-training.txt"))
        # + list(Path(".").rglob(f"*generated_molecules.in-training.txt"))
        # + list(Path(".").rglob(f"*generated_fo2.in-training.txt"))
        # + list(Path(".").rglob(f"*generated_smiles.the-end.txt"))
        # + list(Path(".").rglob(f"*generated_samples.the-end.txt"))
        # + list(Path(".").rglob(f"*generated_molecules.the-end.txt"))
        # + list(Path(".").rglob(f"*generated_fo2.the-end.txt"))
    )
    print(all_generated_files)
    print("###")

    # for max_num_atoms in tqdm(
    #     [
    #         8,
    #         12,
    #         15,
    #         20,
    #     ],
    #     desc="Plotting molecule history",
    # ):
    #     filtered_files = [
    #         x
    #         for x in all_generated_files
    #         if not str(x).startswith(".backup")
    #         and ("fo2" in str(x) if fo2_file else "fo2" not in str(x))
    #         # and ("in-training" in str(x) or "the-end" in str(x))
    #         and (
    #             f"{max_num_atoms}_generated_smiles" in str(x)
    #             or f"{max_num_atoms}-_generated_smiles" in str(x)
    #             or f"_{max_num_atoms}/generated_samples" in str(x)
    #             or f"_{max_num_atoms}/generated_smiles" in str(x)
    #             or f"_{max_num_atoms}/generated_molecules" in str(x)
    #             or f"_{max_num_atoms}_" in str(x)
    #             or f"_p{max_num_atoms}_" in str(x)
    #             or (
    #                 f"{max_num_atoms}-generated_smiles"
    #                 in re.sub("_nparams\d+_", "", str(x))
    #             )
    #         )
    #         # and ("hydrocarbons-alkanes" in str(x) or "hydrocarbons-aklanes" in str(x))
    #     ]
    for max_num_atoms, filtered_files_to_names in MOLECULES.items():
        if not filtered_files_to_names:
            print("No files for ", max_num_atoms)
            continue
        print(
            "Filtered files for", max_num_atoms, "n files", len(filtered_files_to_names)
        )
        print(filtered_files_to_names)
        plot_history(
            max_num_atoms,
            list(filtered_files_to_names.keys()),
            log,
            fo2_file,
            molecule_dir,
            filtered_files_to_names,
        )


def load_smiles_as_canon(
    max_num_atoms: int, molecule_dir: Union[str, None], split: str
) -> list:
    smiles = []
    molecule_dir = (
        molecule_dir
        or f"/home/jungpete/projects/shiny-happiness/data/molecules/size_{max_num_atoms}"
    )
    for line in open(molecule_dir.rstrip("/") + f"/{split}.smi"):
        # mol = Chem.MolFromSmiles(line.strip())
        canon = Chem.CanonSmiles(line)
        smiles.append(canon)
    return smiles


def convert_smiles_to_canon(smiles: list) -> list:
    converted = []
    errored = 0
    for smile in smiles:
        try:
            canon = Chem.CanonSmiles(smile)
        except:
            errored += 1
            print("Can not convert", smile)
            continue
        converted.append(canon)
    if errored:
        print("Errored", errored)
    return converted


def load_fo2(fo2_file, max_num_atoms, split: str):
    val_file = f"/home/jungpete/shiny-happiness/data-fo2/{fo2_file}_p{max_num_atoms}.{split}.txt"
    val_fols: t.List[t.Tuple[str, ...]] = []
    with open(val_file) as f:
        for line in f:
            parsed_line = eval(line)
            val_fols.append(tuple(parsed_line))
    fold_deduplicated_val = sorted(set(val_fols))
    fold_deduplicated_val_as_str = [";".join(fo) for fo in fold_deduplicated_val]
    return fold_deduplicated_val_as_str


def plot_history(
    max_num_atoms: int,
    files: list,
    log: bool,
    fo2_file: Union[str, None],
    molecule_dir: Union[str, None],
    file_to_name: Dict[str, str],
):
    print(f"Plotting history for {max_num_atoms} atoms")
    val_canons = (
        load_smiles_as_canon(max_num_atoms, molecule_dir, "valid")
        if not fo2_file
        else load_fo2(fo2_file, max_num_atoms)
    )
    val_canons_set = set(val_canons)
    train_canons = (
        load_smiles_as_canon(max_num_atoms, molecule_dir, "train")
        if not fo2_file
        else load_fo2(fo2_file, max_num_atoms, "train")
    )
    train_canons_set = set(train_canons)

    group_to_methods = {
        "everything": list(file_to_name.values()),
        "only_digress": [v for k, v in file_to_name.items() if "digress" in k.lower()],
    }

    group_to_fig_axe = {
        key: plt.subplots(1, 2, figsize=(6 * 2, 6)) for key in group_to_methods.keys()
    }

    for _, axes in group_to_fig_axe.values():
        for ax in axes:
            ax.set_ylim([0, 1])

    molecules_counters = []
    methods = set()

    color_generator = get_cmap(len(files))
    method_to_order = defaultdict(lambda: 1)

    # End of configs

    method_to_data = {}
    method_to_highest_generated = defaultdict(lambda: 0)
    allowed_methods = [m for methods_ in group_to_methods.values() for m in methods_]
    method_to_data_file = f"report/method_to_data_{max_num_atoms}.pickle"

    if not os.path.exists(method_to_data_file):
        for idx, file in enumerate(files):
            method = filepath_to_title(file, file_to_name)
            print("Processing", method, file)
            methods.add(method)

            # if method not in allowed_methods:
            #     print(method, " not in ", allowed_methods)
            #     continue

            style, color = "-", color_generator(idx)

            history = get_molecule_history(file)
            if not history:
                # raise RuntimeError("No history for ", method, file)
                print("No history for ", method, file)
                continue
            history_dedup = dedup_by(history, lambda x: x)

            molecule_counter = Counter(history)
            molecules_counters.append((method, molecule_counter))

            generated_molecules_sorted = sorted(
                molecule_counter.keys(),
                key=lambda x: molecule_counter[x],
                reverse=True,
            )

            (
                sorted_val_sizes,
                sorted_val_recalls,
                sorted_val_precisions,
            ) = get_cumulative_perc_deduplicated(generated_molecules_sorted, val_canons)

            # If there are multiple same runs, take one that were running the longest (generated most molecules)
            if method_to_highest_generated[method] >= len(sorted_val_sizes):
                print("Warning - skipping", method, file)
                continue

            ranked_conf_matrix = get_conf_matrix(molecule_counter, val_canons)
            ranked_conf_matrix.to_csv(
                f"report/{'molecule' if not fo2_file else fo2_file}_conf_matrix_{max_num_atoms}_{method}.csv"
            )

            method_to_highest_generated[method] = len(sorted_val_sizes)
            method_to_data[method] = (
                method,
                style,
                color,
                history_dedup,
                sorted_val_sizes,
                sorted_val_recalls,
                sorted_val_precisions,
                " (in-training)" if "in-training" in str(file) else " (the-end)",
            )

        with open(method_to_data_file, "wb") as f:
            pickle.dump((methods, method_to_data), f)

    else:
        with open(method_to_data_file, "rb") as f:
            methods, method_to_data = pickle.load(f)

    try:
        max_generated_number = max([len(data[3]) for data in method_to_data.values()])
    except ValueError:
        print("No data to plot for ", max_num_atoms)
        return
    unique_in_val_number = len(val_canons_set)

    table = {
        "exp name": [],
        "#samples": [],
        "recall": [],
        "precision": [],
        "novelty": [],
        "uniqueness": [],
        "validity": [],
    }

    for group, group_methods in group_to_methods.items():
        fig, axes = group_to_fig_axe[group]
        for method in group_methods:
            if method not in method_to_data:
                continue
            (
                method_2,
                style,
                color,
                history_dedup,
                sorted_val_sizes,
                sorted_val_recalls,
                sorted_val_precisions,
                type_,
            ) = method_to_data[method]
            assert method == method_2, (method, method_2)
            label = method + type_ if not ANONYMIZE else NAME_TO_ANON_NAME[method]
            history_dedup_set = set(history_dedup)

            if label not in table["exp name"]:
                table["exp name"].append(label)
                table["#samples"].append(sorted_val_sizes[-1])
                table["recall"].append(sorted_val_recalls[-1])
                table["precision"].append(sorted_val_precisions[-1])
                table["novelty"].append(
                    len(history_dedup_set - train_canons_set) / len(history_dedup_set)
                )
                table["uniqueness"].append(1.0)  # We deduplicated.
                table["validity"].append(1.0)  # We have only valid samples.

            axes[0].set_xlabel(
                "Number of generated unique molecules sorted by frequency"
            )
            axes[0].set_ylabel("How many from validation data was generated (recall)")
            axes[0].plot(
                sorted_val_sizes[:max_generated_number],
                sorted_val_recalls[:max_generated_number],
                style,
                label=label,
                alpha=1.0,
                color=color,
            )
            axes[1].set_xlabel(
                "Number of generated unique molecules sorted by frequency"
            )
            axes[1].set_ylabel(
                "How many from generated molecules are in validation data (precision)"
            )
            axes[1].plot(
                sorted_val_sizes[:unique_in_val_number],
                sorted_val_precisions[:unique_in_val_number],
                style,
                label=label,
                alpha=1.0,
                color=color,
            )
            # axes[2].set_xlabel("recall")
            # paxes[2].set_ylabel("Precision")
            # axes[2].set_xlim([0, 1])
            # axes[2].plot(
            #     sorted_val_recalls[:unique_in_val_number],
            #     sorted_val_precisions[:unique_in_val_number],
            #     style,
            #     label=label,
            #     alpha=1.0,
            #     color=color,
            # )

    for group, (fig, axes) in group_to_fig_axe.items():
        for ax in axes:
            ax.legend(loc="upper left", prop={"size": 12})
            if log:
                ax.set_yscale("log")
            fig.savefig(
                f"report/{'molecule' if not fo2_file else fo2_file}_history_{max_num_atoms}_{group}.jpg"
            )

    table_df = pd.DataFrame(table).sort_values("exp name", ascending=True)
    table_df.to_csv(
        f"report/{'molecule' if not fo2_file else fo2_file}_table_{max_num_atoms}.csv",
        index=False,
    )
    with open(
        f"report/{'molecule' if not fo2_file else fo2_file}_table_{max_num_atoms}.latex",
        "w",
    ) as f:
        f.write(
            table_df.to_latex(
                index=False,
                caption=f"Precision and recall on molecules of size {max_num_atoms}",
                label=f"tab:molecules_{max_num_atoms}",
                float_format=lambda x: (
                    f"{x:.1f}" if x == 1.0 or x == 0.0 else f"{x:.5f}"
                ),
            )
        )


def get_conf_matrix(molecule_counter, val_canons) -> pd.DataFrame:
    val_canons_set = set(val_canons)
    counts = sorted(set(molecule_counter.values()), reverse=True)

    data = []

    for rank in counts:
        generated_molecules_above_rank = set(
            [m for m, c in molecule_counter.items() if c >= rank]
        )
        generated_molecules_below_rank = set(
            [m for m, c in molecule_counter.items() if c < rank]
        )

        tp = len(generated_molecules_above_rank & val_canons_set)
        fp = len(generated_molecules_above_rank - val_canons_set)
        tn = len(generated_molecules_below_rank - val_canons_set)
        fn = len(generated_molecules_below_rank & val_canons_set)

        data.append(
            {
                "len_val_canons_set": len(val_canons_set),
                "len_generated_molecules_above_rank": len(
                    generated_molecules_above_rank
                ),
                "len_generated_molecules_below_rank": len(
                    generated_molecules_below_rank
                ),
                "rank": rank,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )

    df = pd.DataFrame.from_records(data)
    return df


def get_cumulative_perc_deduplicated(gen_smiles, dataset_smiles):
    dataset_smiles_set = set(dataset_smiles)

    recalls = []
    precisions = []
    sizes = []
    subset = set()

    for molecule in tqdm(gen_smiles):
        subset.add(molecule)
        recalls.append(len(subset & dataset_smiles_set) / len(dataset_smiles_set))
        precisions.append(len(subset & dataset_smiles_set) / len(subset))
        sizes.append(len(subset))

    return sizes, recalls, precisions


def get_molecule_history(filepath: Union[str, Path]) -> list:
    history = []
    with open(filepath) as file:
        for idx, line in tqdm(enumerate(file), desc=f"Reading {filepath}"):
            line = line.strip()
            if not line:
                continue

            items = line.split()

            if "moler" in str(filepath) and idx < 20 and len(items) > 1:
                # Output made of >, not actual molecules.
                continue
            elif len(items) == 3:
                smile = items[2]
            elif len(items) == 2 and items[1] == "working":
                smile = items[0]
            elif len(items) == 1:
                smile = items[0]
            else:
                raise Exception(f"Unknown format, {items}")

            history.append(smile)

            if DEBUG and len(history) > 1000:
                break

    return convert_smiles_to_canon(history) if "fo2" not in str(filepath) else history


def dedup_by(smiles: list, callable=lambda x: x) -> list:
    seen = set()
    deduped = []
    for smile in smiles:
        if callable(smile) not in seen:
            deduped.append(smile)
            seen.add(callable(smile))
    return deduped


def filepath_to_title(filepath: Path, file_to_name: Dict[str, str]) -> str:
    filepath_str = str(filepath)

    if filepath_str in file_to_name:
        return file_to_name[filepath_str]

    if "DiGress" and "continuous" in filepath_str:
        return "DiGress continuous"

    elif "DiGress" and "discrete" in filepath_str:
        return "DiGress discrete"

    elif "paccmann" in filepath_str and "vae" in filepath_str:
        return "Paccmann VAE"

    elif "moler" in filepath_str.lower():
        return "MoLeR"

    elif "data_efficient_grammar" in filepath_str.lower():
        return "Data Efficient Grammar"

    elif "rnn" and "selfies" in filepath_str.lower():
        return "RNN Selfies"

    elif "rnn" and "regex" in filepath_str.lower():
        return "RNN Regex"

    elif "rnn" and "char" in filepath_str.lower():
        return "RNN Char"

    else:
        raise RuntimeError(f"Unknown method for {filepath}")


def flat(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


if __name__ == "__main__":
    typer.run(main)
