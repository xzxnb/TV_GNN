import typing as t
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from compare_runs import get_clear_val_canon_smiles
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from dgd.datasets.abstract_dataset import AbstractDatasetInfos
from dgd.datasets.abstract_dataset import MolecularDataModule

bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}


class MoleculeDataModule(MolecularDataModule):
    def __init__(
        self, max_num_atoms, ontology_file, base_mol_dir, limit_train, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.max_num_atoms = max_num_atoms
        self.ontology_file = ontology_file
        self.base_mol_dir = base_mol_dir
        self.limit_train = limit_train
        self.train_smiles = None
        self.types = None

    def prepare_data(self) -> None:
        train_dataset = MoleculeDataset(
            self.max_num_atoms,
            "train",
            base_mol_dir=self.base_mol_dir,
            ontology_file=self.ontology_file,
            limit=self.limit_train,
        )
        self.train_dataset = train_dataset
        self.train_smiles = train_dataset.train_smiles
        self.types = train_dataset.types
        val_dataset = MoleculeDataset(
            self.max_num_atoms,
            "val",
            types=train_dataset.types,
            base_mol_dir=self.base_mol_dir,
            ontology_file=self.ontology_file,
        )
        self.val_dataset = val_dataset
        self.val_smiles = val_dataset.val_smiles
        self.val_smiles_set = val_dataset.val_smiles_set
        super().prepare_data(
            datasets={
                "train": train_dataset,
                "val": val_dataset,
                "test": val_dataset,
            }
        )


class MoleculeDataset(Dataset):
    def __init__(
        self,
        size: int,
        split: str,
        ontology_file,
        base_mol_dir,
        types={},
        limit=0,
    ) -> None:
        super().__init__()
        self.size = size
        self.types = types
        self.limit = limit
        train_smiles, val_smiles, _, _ = get_clear_val_canon_smiles(
            size,
            ontology_file,
            base_mol_dir=base_mol_dir,
        )
        self.train_smiles, self.val_smiles = list(train_smiles), list(val_smiles)
        self.val_smiles_set = set(self.val_smiles)

        self.data = self.process_smiles(
            self.train_smiles if split == "train" else self.val_smiles,
            limit=self.limit,
        )

    def len(self) -> int:
        return len(self.data)

    def get(self, idx: int) -> Data:
        return self.data[idx]

    def process_smiles(
        self, smiles: t.List[str], limit: t.Optional[int] = None
    ) -> t.List[Data]:
        data_list = []

        for i, smile in enumerate(smiles):
            mol = Chem.MolFromSmiles(smile)
            for atom in mol.GetAtoms():
                self.types[atom.GetSymbol()] = self.types.get(
                    atom.GetSymbol(), len(self.types)
                )

        for i, smile in enumerate(smiles):
            mol = Chem.MolFromSmiles(smile)

            N = mol.GetNumAtoms()

            type_idx = []
            for atom in mol.GetAtoms():
                self.types[atom.GetSymbol()] = self.types.get(
                    atom.GetSymbol(), len(self.types)
                )
                type_idx.append(self.types[atom.GetSymbol()])

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(self.types)).float()
            y = torch.zeros((1, 0), dtype=torch.float)

            # if self.remove_h:
            #     type_idx = torch.Tensor(type_idx).long()
            #     to_keep = type_idx > 0
            #     edge_index, edge_attr = subgraph(to_keep, edge_index, edge_attr, relabel_nodes=True,
            #                                      num_nodes=len(to_keep))
            #     x = x[to_keep]
            #     # Shift onehot encoding to match atom decoder
            #     x = x[:, 1:]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

            # if self.pre_filter is not None and not self.pre_filter(data):
            #     continue
            # if self.pre_transform is not None:
            #     data = self.pre_transform(data)

            data_list.append(data)

            if limit and len(data_list) >= limit:
                break

        return data_list


class MoleculesInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, types, recompute_statistics=False):
        self.remove_h = cfg.dataset.remove_h
        self.need_to_strip = (
            False  # to indicate whether we need to ignore one output from the model
        )

        self.name = "molecules"
        self.val_smiles_set = datamodule.val_smiles_set

        self.atom_encoder = types
        self.atom_decoder = list(types.keys())
        self.num_atom_types = len(types)
        self.n_nodes = datamodule.node_counts()
        self.max_n_nodes = int(np.argmax(self.n_nodes))
        self.valencies = datamodule.valency_count(self.max_n_nodes)
        self.valencies = self.valencies[: (self.valencies == 0).nonzero()[0].item()]
        self.max_weight = 390
        self.atom_weights = {i: i + 1 for i in range(len(types))}

        np.set_printoptions(suppress=True, precision=5)

        self.node_types = datamodule.node_types()
        self.edge_types = datamodule.edge_counts()
        self.valency_distribution = self.valencies

        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
