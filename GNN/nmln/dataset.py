import torch
import numpy as np
import typing as t

from pathlib import Path
from nmln.ontology import Predicate, Domain, Ontology


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        constants: t.List[str],
        predicates: t.List[Predicate],
        domain: Domain,
        ontology: Ontology,
        mol_dir: t.Union[str, Path],
    ):
        super().__init__()

        self.mol_dir = Path(mol_dir)

        self.constants = constants
        self.files = self.load_files(self.mol_dir)
        self.domain = domain
        self.predicates = predicates
        self.ontology = ontology

        self.y = torch.from_numpy(
            np.stack(
                [self.ontology.file_to_linearState(file) for file in self.files], axis=0
            ).astype(np.float32)
        )

    @staticmethod
    def load_files(path: Path) -> list:
        print(f"Loading files from {path}")
        return sorted([p for p in path.glob("*") if not str(p).endswith("ontology")])

    @property
    def max_num_atoms(self):
        return len(self.constants)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.y[idx]

    def get_collate_fn(self) -> t.Callable:
        def collate_fn(batch):
            return torch.vstack(batch)

        return collate_fn

    def get_dataloader(
        self,
        shuffle: bool,
        batch_size: int,
        num_workers: int,
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=self.get_collate_fn(),
            prefetch_factor=22 if num_workers > 0 else 2,
            persistent_workers=num_workers > 0,
            drop_last=True,
        )
