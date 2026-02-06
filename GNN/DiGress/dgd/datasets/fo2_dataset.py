import typing as t
import random
import os
import numpy as np
import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from DiGress.dgd.datasets.abstract_dataset import AbstractDatasetInfos
from DiGress.dgd.datasets.abstract_dataset import AbstractDataModule

bonds = {"friends": 0}


class FO2DataModule(AbstractDataModule):
    def __init__(
        self,
        cfg,
        train_size: int,
        val_size: int = 10000,
        auto_train_ratio: float = 0.8,
        val_paths: t.List[str] = None,
        train_path1: str = None,
        train_path2: str = None,
    ):
        super().__init__(cfg=cfg)
        assert (
            train_path1 or val_paths
        ), "Either file_name or both train_path must be provided"
        self.val_paths = val_paths
        self.train_path1 = train_path1
        self.train_path2 = train_path2
        self.auto_train_ratio = auto_train_ratio
        self.train_size = train_size
        self.val_size = val_size

    def prepare_data(self, _=None) -> None:
        train_fols1: t.List[t.Tuple[str, ...]] = []
        with open(self.train_path1) as f:
            for line in f:
                parsed_line = eval(line)
                train_fols1.append(tuple(parsed_line))
                if int(self.cfg.train.n_train_data) > 0 and len(train_fols1) >= int(
                    self.cfg.train.n_train_data
                ):
                    break
        # fols_deduplicated_train = sorted(set(train_fols1))
        print(
            f"Train1 number of samples: {len(train_fols1)}"
        )
        random.shuffle(train_fols1)
        train1 = train_fols1[:self.train_size]

        train_fols2: t.List[t.Tuple[str, ...]] = []
        with open(self.train_path2) as f:
            for line in f:
                parsed_line = eval(line)
                train_fols2.append(tuple(parsed_line))
                if int(self.cfg.train.n_train_data) > 0 and len(train_fols2) >= int(
                        self.cfg.train.n_train_data
                ):
                    break
        # fols_deduplicated_train = sorted(set(train_fols2))
        print(
            f"Train2 number of samples: {len(train_fols2)}"
        )
        random.shuffle(train_fols2)
        train2 = train_fols2[:self.train_size]

        val1: t.List[t.Tuple[str, ...]] = []
        val2: t.List[t.Tuple[str, ...]] = []
        if self.val_paths is None:
            val1 = train_fols1[self.train_size:self.train_size + self.val_size]
            val2 = train_fols2[self.train_size:self.train_size + self.val_size]
        else:
            with open(self.val_paths[0]) as f:
                for line in f:
                    parsed_line = eval(line)
                    val1.append(tuple(parsed_line))
            with open(self.val_paths[1]) as f:
                for line in f:
                    parsed_line = eval(line)
                    val2.append(tuple(parsed_line))


        print(
            f"Training real samples: {len(train1)}, generated samples: {len(train2)}"
        )
        print(
            f"Validation real samples: {len(val1)}, generated samples: {len(val2)}"
        )

        all_dataset = FO2Dataset(
            train1 + val1 + train2 + val2, "all"
        )
        train_dataset = FO2Dataset(
            train1,
            "train",
            all_dataset.types,
            all_dataset.constant_to_idx,
            all_dataset.edge_to_idx,
        )
        val_dataset = FO2Dataset(
            val1,
            "val",
            train_dataset.types,
            train_dataset.constant_to_idx,
            train_dataset.edge_to_idx,
        )
        gen_train_dataset = FO2Dataset(
            train2,
            "gen",
            train_dataset.types,
            train_dataset.constant_to_idx,
            train_dataset.edge_to_idx,
        )
        gen_val_dataset = FO2Dataset(
            val2,
            "gen",
            train_dataset.types,
            train_dataset.constant_to_idx,
            train_dataset.edge_to_idx,
        )

        train_and_gen_dataset = torch.utils.data.ConcatDataset(
            [train_dataset, gen_train_dataset]
        )
        test_and_gen_dataset = torch.utils.data.ConcatDataset(
            [val_dataset, gen_val_dataset]
        )

        self.types = all_dataset.types
        self.idx_to_type = all_dataset.idx_to_type
        self.constant_to_idx = all_dataset.constant_to_idx
        self.idx_to_constant = all_dataset.idx_to_constant
        self.edge_to_idx = all_dataset.edge_to_idx
        self.idx_to_edge = all_dataset.idx_to_edge

        super().prepare_data(
            datasets={
                "_train": train_dataset,
                "_val": val_dataset,
                "_test": val_dataset,
                "train_and_gen": train_and_gen_dataset,
                "test_and_gen": test_and_gen_dataset,
            }
        )

    def to_sentence(self, nodes, adj_matrix) -> str:
        sentence = []

        constant_to_type = {}
        for idx, node_type in enumerate(nodes):
            constant = self.idx_to_constant[idx]
            type_ = self.idx_to_type[node_type.item()]
            if type_ != "NONE":
                sentence.append(f"{type_}({constant})")
            constant_to_type[constant] = type_

        for from_idx, to_idx in zip(*torch.nonzero(adj_matrix, as_tuple=True)):
            fulledge = self.idx_to_edge[adj_matrix[from_idx, to_idx].item()]
            subedges = fulledge.split(";")
            for edge in subedges:
                from_constant = self.idx_to_constant[from_idx.item()]
                to_constant = self.idx_to_constant[to_idx.item()]
                sentence.append(f"{edge}({from_constant},{to_constant})")

        sentence = sorted(sentence)
        return "[" + ",".join(f"'{x}'" for x in sentence) + "]"


class FO2Dataset(Dataset):
    def __init__(
        self,
        objects: t.List[t.List[str]],
        split: str,
        types={},
        constant_to_idx={},
        edge_to_idx={},
    ) -> None:
        super().__init__()
        self.objects = objects
        self.split = split
        self.types = types
        self.idx_to_type = {v: k for k, v in self.types.items()}
        self.constant_to_idx = constant_to_idx
        self.idx_to_constant = {v: k for k, v in self.constant_to_idx.items()}
        self.edge_to_idx = edge_to_idx
        self.idx_to_edge = {v: k for k, v in self.edge_to_idx.items()}
        self.data = self.process_objects(self.objects)

    def len(self) -> int:
        return len(self.data)

    def get(self, idx: int) -> Data:
        return self.data[idx]

    def process_objects(self, sentences: t.List[t.List[str]]) -> t.List[Data]:
        data_list = []

        if self.split == "all":
            self.types["NONE"] = self.types.get("NONE", len(self.types))
            self.edge_to_idx["NONE"] = self.edge_to_idx.get(
                "NONE", len(self.edge_to_idx)
            )
            for i, sentence in enumerate(sentences):
                for predicate in sentence:
                    inside_of_parenthetes = predicate.split("(")[1].rstrip(")")
                    constants = inside_of_parenthetes.split(",")

                    for constant in constants:
                        self.constant_to_idx[constant] = self.constant_to_idx.get(
                            constant, len(self.constant_to_idx)
                        )

                    if predicate.count(",") == 0:
                        predicate_symbol = predicate.split("(")[0]
                        self.types[predicate_symbol] = self.types.get(
                            predicate_symbol, len(self.types)
                        )

                    elif predicate.count(",") == 1:
                        edge = predicate.split("(")[0]
                        self.edge_to_idx[edge] = self.edge_to_idx.get(
                            edge, len(self.edge_to_idx)
                        )

                    else:
                        raise RuntimeError(predicate)

        self.idx_to_constant = {v: k for k, v in self.constant_to_idx.items()}
        self.idx_to_type = {v: k for k, v in self.types.items()}
        self.idx_to_edge = {v: k for k, v in self.edge_to_idx.items()}

        for i, sentence in enumerate(sentences):
            adj_matrix = torch.zeros(
                (len(self.idx_to_constant), len(self.idx_to_constant))
            )
            edge_indexes_to_edge_types = {}

            constant_to_type = {}

            edge_type_to_adj_matrix = {
                e: torch.zeros((len(self.idx_to_constant), len(self.idx_to_constant)))
                for e in self.edge_to_idx.keys()
                if e != "NONE"
            }

            for predicate in sentence:
                if predicate.count(",") == 0:
                    predicate_symbol = predicate.split("(")[0]
                    constant = predicate.split("(")[1].rstrip(")")
                    constant_to_type[constant] = self.types[predicate_symbol]

            for predicate in sentence:
                if predicate.count(",") == 0:
                    continue

                assert predicate.count(",") == 1, predicate

                constants = predicate.split("(")[1].rstrip(")").split(",")
                from_, to_ = constants
                from_idx, to_idx = (
                    self.constant_to_idx[from_],
                    self.constant_to_idx[to_],
                )

                edge = predicate.split("(")[0]

                adj_matrix[from_idx, to_idx] = 1
                adj_matrix[to_idx, from_idx] = 1

                edge_type_to_adj_matrix[edge][from_idx, to_idx] = 1
                edge_type_to_adj_matrix[edge][to_idx, from_idx] = 1

                if (from_idx, to_idx) not in edge_indexes_to_edge_types:
                    edge_indexes_to_edge_types[(from_idx, to_idx)] = []
                if (to_idx, from_idx) not in edge_indexes_to_edge_types:
                    edge_indexes_to_edge_types[(to_idx, from_idx)] = []

                if edge not in edge_indexes_to_edge_types[(from_idx, to_idx)]:
                    edge_indexes_to_edge_types[(from_idx, to_idx)].append(edge)
                if edge not in edge_indexes_to_edge_types[(to_idx, from_idx)]:
                    edge_indexes_to_edge_types[(to_idx, from_idx)].append(edge)

            if all(len(x) == 1 for x in edge_indexes_to_edge_types.values()):
                pass
            elif all(len(x) in (1, 2) for x in edge_indexes_to_edge_types.values()):
                first_edge_with_two_types = [
                    x for x in edge_indexes_to_edge_types.values() if len(x) == 2
                ][0]
                new_edge = ";".join(sorted([x for x in first_edge_with_two_types]))
                if new_edge not in self.edge_to_idx:
                    new_edge_idx = len(self.edge_to_idx)
                    self.edge_to_idx[new_edge] = new_edge_idx
                    self.idx_to_edge[new_edge_idx] = new_edge
            else:
                raise RuntimeError(edge_indexes_to_edge_types)

            x = F.one_hot(
                torch.tensor(
                    [
                        constant_to_type.get(
                            self.idx_to_constant[idx], self.types["NONE"]
                        )
                        for idx in range(len(self.constant_to_idx))
                    ]
                ),
                num_classes=len(self.types),
            ).float()
            y = torch.zeros([1, 0]).float()
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj_matrix)
            edge_attr = torch.zeros(
                edge_index.shape[-1], len(self.edge_to_idx), dtype=torch.float
            )
            for this_edge_index, (from_, to_) in enumerate(zip(*edge_index)):
                edge_types = edge_indexes_to_edge_types[(from_.item(), to_.item())]
                stringified_edge_types = ";".join(sorted(edge_types))
                edge_attr[this_edge_index][self.edge_to_idx[stringified_edge_types]] = 1

            n = adj_matrix.shape[-1]
            num_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes
            )
            # Unsqueeze to add batch dimension so that it's concated properly by dataloader.
            data.node_feats = x.unsqueeze(0)
            data.adj_matrix = adj_matrix.unsqueeze(0)
            data.edge_type_to_adj_matrix = {
                k: v.unsqueeze(0) for k, v in edge_type_to_adj_matrix.items()
            }
            data.gan_y = torch.tensor(1.0 if self.split == "gen" else 0.0)
            data_list.append(data)

        return data_list


class FO2DatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = "nx_graphs"
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
