# These imports are tricky because they use c++, do not move them
from rdkit import Chem

try:
    import graph_tool
except ModuleNotFoundError:
    pass

import dagshub
import os
import pathlib
import warnings
import mlflow

import torch
import wandb
import hydra
import omegaconf
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from gan.mlflow_utils import SafeMLFlowLogger

from dgd import utils
from dgd.datasets import (
    guacamol_dataset,
    qm9_dataset,
    moses_dataset,
    molecule_dataset,
    fo2_dataset,
)
from dgd.datasets.spectre_dataset import (
    SBMDataModule,
    Comm20DataModule,
    PlanarDataModule,
    SpectreDatasetInfos,
)
from dgd.metrics.abstract_metrics import (
    TrainAbstractMetricsDiscrete,
    TrainAbstractMetrics,
)
from dgd.analysis.spectre_utils import (
    PlanarSamplingMetrics,
    SBMSamplingMetrics,
    Comm20SamplingMetrics,
)
from diffusion_model import LiftedDenoisingDiffusion
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from dgd.metrics.molecular_metrics import (
    TrainMolecularMetrics,
    SamplingMolecularMetrics,
    TrainMolecularMetricsDiscrete,
)
from dgd.analysis.visualization import MolecularVisualization, NonMolecularVisualization
from dgd.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from dgd.diffusion.extra_features_molecular import ExtraMolecularFeatures

warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, model_kwargs):
    """Resumes a run. It loads previous config without allowing to update keys (used for testing)."""
    saved_cfg = cfg.copy()
    name = cfg.general.name + "_resume"
    resume = cfg.general.test_only
    if cfg.model.type == "discrete":
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(
            resume, smiles_filename_contain="the-end", **model_kwargs
        )
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(
            resume, smiles_filename_contain="the-end", **model_kwargs
        )
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split("outputs")[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    if cfg.model.type == "discrete":
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(
            resume_path, **model_kwargs
        )
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(
            resume_path, **model_kwargs
        )
    new_cfg = model.cfg

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + "_resume"

    new_cfg = cfg.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    kwargs = {
        "name": cfg.general.name,
        "project": f"graph_ddm_{cfg.dataset.name}",
        "config": config_dict,
        "settings": wandb.Settings(_disable_stats=True),
        "reinit": True,
        "mode": cfg.general.wandb,
    }
    wandb.init(**kwargs)
    wandb.save("*.txt")
    return cfg


@hydra.main(version_base="1.1", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]

    seed = int(cfg["seed"])
    max_num_atoms = (
        int(cfg["max_num_atoms"]) if dataset_config["name"] != "fo2" else None
    )
    limit_train = int(cfg["limit_train"])
    assert dataset_config["name"] == "fo2" or max_num_atoms > 0
    file_name = cfg["filename"]

    # Initialize logger.

    if os.environ.get("DAGSHUB_REPO_OWNER") or os.environ.get("DAGSHUB_REPO_NAME"):
        dagshub.init(
            repo_owner=os.environ.get("DAGSHUB_REPO_OWNER"),
            repo_name=os.environ.get("DAGSHUB_REPO_NAME"),
            mlflow=True,
        )

    seed_everything(seed)
    cfg.seed = seed

    cfg.model.n_layers = int(cfg.model.n_layers)
    cfg.model.divide_dims = int(cfg.model.divide_dims)
    cfg.model.force_n_edges = int(cfg.model.force_n_edges)
    cfg.model.diffusion_steps = int(cfg.model.diffusion_steps)

    run_name = f"DiGress - {cfg.model.type}"
    if limit_train:
        run_name = f"{limit_train=}-{run_name}"
    if seed:
        run_name = f"{seed=}-{run_name}"
    if cfg.model.n_layers:
        run_name = f"{cfg.model.n_layers=}-{run_name}"
    if cfg.model.diffusion_steps:
        run_name = f"{cfg.model.diffusion_steps=}-{run_name}"
    if cfg.general.check_val_every_n_epochs:
        run_name = f"{run_name}-{cfg.general.check_val_every_n_epochs=}"
    if cfg.general.sample_every_val:
        run_name = f"{run_name}-{cfg.general.sample_every_val=}"

    logger_mlflow = SafeMLFlowLogger(run_name=run_name)
    logger_mlflow.log_hyperparams({"max_num_atoms": max_num_atoms, **cfg})
    logger_mlflow.log_metrics({"seed": seed})

    if cfg.model.divide_dims > 0:
        assert cfg.model.divide_dims % 2 == 0
        for name, value in cfg.model.hidden_dims.items():
            cfg.model.hidden_dims[name] = value // cfg.model.divide_dims

    ontology_file = cfg["ontology_file"]
    base_mol_dir = cfg["base_mol_dir"].strip() or None
    base_mol_dir_name = (
        base_mol_dir.strip().rstrip("/").split("/")[-1].strip()
        if base_mol_dir
        else None
    )
    assert base_mol_dir_name is None or f"atomsize{max_num_atoms}" in base_mol_dir_name

    # cfg.general.name = f"debug-seed{seed}-v2-afterpaper-{cfg.model.type}-{max_num_atoms or cfg['filename']}-{base_mol_dir_name or ''}"
    cfg.general.name = f"fixedmultiedge-bigepochsample-seed{seed}-v2-afterpaper-{cfg.model.type}-{max_num_atoms or cfg['filename']}-{base_mol_dir_name or ''}"

    if cfg.model.force_n_edges:
        cfg.general.name = f"force_n_edges{cfg.model.force_n_edges}-" + cfg.general.name

    if dataset_config["name"] in ["sbm", "comm-20", "planar", "fo2"]:
        dataset_infos = None

        if dataset_config["name"] == "fo2":
            datamodule = fo2_dataset.FO2DataModule(file_name, cfg)
            datamodule.prepare_data()
            sampling_metrics = SBMSamplingMetrics(datamodule.dataloaders)
            dataset_infos = fo2_dataset.FO2DatasetInfos(datamodule, dataset_config)
        elif dataset_config["name"] == "sbm":
            datamodule = SBMDataModule(cfg)
            sampling_metrics = SBMSamplingMetrics(datamodule.dataloaders)
        elif dataset_config["name"] == "comm-20":
            datamodule = Comm20DataModule(cfg)
            sampling_metrics = Comm20SamplingMetrics(datamodule.dataloaders)
        else:
            datamodule = PlanarDataModule(cfg)
            sampling_metrics = PlanarSamplingMetrics(datamodule.dataloaders)

        dataset_infos = (
            SpectreDatasetInfos(datamodule, dataset_config)
            if dataset_infos is None
            else dataset_infos
        )
        train_metrics = (
            TrainAbstractMetricsDiscrete()
            if cfg.model.type == "discrete"
            else TrainAbstractMetrics()
        )
        visualization_tools = None  # NonMolecularVisualization()

        if cfg.model.type == "discrete" and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(
                cfg.model.extra_features, dataset_info=dataset_infos
            )
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

        model_kwargs = {
            "dataset_infos": dataset_infos,
            "train_metrics": train_metrics,
            "sampling_metrics": sampling_metrics,
            "visualization_tools": visualization_tools,
            "extra_features": extra_features,
            "domain_features": domain_features,
        }
    elif dataset_config["name"] in ["qm9", "guacamol", "moses", "molecules"]:
        if dataset_config["name"] in ("qm9", "molecules"):
            datamodule = molecule_dataset.MoleculeDataModule(
                max_num_atoms,
                ontology_file=ontology_file,
                base_mol_dir=base_mol_dir,
                cfg=cfg,
                limit_train=limit_train,
            )
            datamodule.prepare_data()
            train_smiles = datamodule.train_smiles
            dataset_infos = molecule_dataset.MoleculesInfos(
                datamodule=datamodule, cfg=cfg, types=datamodule.types
            )
        elif dataset_config["name"] == "qm9":
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            datamodule.prepare_data()
            train_smiles = qm9_dataset.get_train_smiles(
                cfg=cfg,
                train_dataloader=datamodule.train_dataloader(),
                dataset_infos=dataset_infos,
                evaluate_dataset=False,
            )
        elif dataset_config["name"] == "guacamol":
            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            datamodule.prepare_data()
            train_smiles = None

        elif dataset_config.name == "moses":
            datamodule = moses_dataset.MOSESDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            datamodule.prepare_data()
            train_smiles = None
        else:
            raise ValueError("Dataset not implemented")

        if cfg.model.type == "discrete" and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(
                cfg.model.extra_features, dataset_info=dataset_infos
            )
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

        logger_mlflow.log_metrics(
            {
                "len/train_ds": len(datamodule.train_dataset),
                "len/val_ds": len(datamodule.val_dataset),
            }
        )

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

        if cfg.model.type == "discrete":
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        else:
            train_metrics = TrainMolecularMetrics(dataset_infos)

        # We do not evaluate novelty during training
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = None  # MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

        model_kwargs = {
            "dataset_infos": dataset_infos,
            "train_metrics": train_metrics,
            "sampling_metrics": sampling_metrics,
            "visualization_tools": visualization_tools,
            "extra_features": extra_features,
            "domain_features": domain_features,
        }
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split("checkpoints")[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split("checkpoints")[0])

    utils.create_folders(cfg)
    cfg = setup_wandb(cfg)

    if cfg.model.type == "discrete":
        model = DiscreteDenoisingDiffusion(
            cfg=cfg,
            force_n_edges=cfg.model.force_n_edges,
            logger_mlflow=logger_mlflow,
            **model_kwargs,
        )
    else:
        model = LiftedDenoisingDiffusion(
            cfg=cfg,
            force_n_edges=cfg.model.force_n_edges,
            logger_mlflow=logger_mlflow,
            **model_kwargs,
        )

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"/home/jungpete/projects/shiny-happiness/DiGress/checkpoints/{cfg.general.name}",
        filename="{epoch}",
        monitor="val/epoch_NLL",
        save_top_k=5,
        mode="min",
        every_n_epochs=1,
    )
    callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == "test":
        print(
            "[WARNING]: Run is called 'test' -- it will run in debug mode on 20 batches. "
        )
    elif name == "debug":
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")
    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[cfg.general.gpus] if torch.cuda.is_available() else None,
        limit_train_batches=20 if name == "test" else None,
        limit_val_batches=20 if name == "test" else None,
        limit_test_batches=20 if name == "test" else None,
        val_check_interval=cfg.general.val_check_interval,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=cfg.general.name == "debug",
        strategy=None,
        enable_progress_bar=False,
        callbacks=callbacks,
        logger=[],
    )

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ["debug", "test"]:
            trainer.test(model, datamodule=datamodule)
    else:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        n_generated = 0
        while n_generated < cfg.general.samples_to_generate_at_test:
            print(f"Generated {n_generated}")
            n_generated += model.validation_epoch_end(None)


if __name__ == "__main__":
    main()
