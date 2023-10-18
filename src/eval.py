import logging
import numpy as np
from typing import Optional
import mlflow
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning import seed_everything
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from src.utils import config_utils

log = logging.getLogger(__name__)

def plot_potential(potential,logger):
    fig, ax = plt.subplots()
    potential = potential.detach().cpu().numpy()
    max = potential.max()
    ax.hist(potential, bins=100, density=True,range=(potential.min(),max))
    ax.set_xlabel("Potential energy")
    ax.set_ylabel("Density")
    fig.savefig("potential.png")
    logger.experiment.log_artifact(logger.run_id, "potential.png")

def eval(config: DictConfig, model: LightningModule, trainer: Trainer, datamodule: LightningDataModule) -> Optional[float]:
    """Contains the evaluation pipeline.

    Uses the configuration to execute the evaluation pipeline on a given model.

    args:
        config (DictConfig): Configuration composed by Hydra.
        model (LightningModule): The model that is evaluated
        trainer (Trainer)
        datamodule (LightningDataModule)
    """

    if 'seed' in config:
        seed_everything(config.seed)

    # Send some parameters from config to all lightning loggers
    log.info('Logging hyperparameters!')
    config_utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=[],
        logger=trainer.logger,
    )
    # add your evaluation logic here
    #trainer.test(model=model, datamodule=datamodule)
    out = model.sample(**config.sample)
    sample = out["x"]
    traj = out["traj"]
    potential = datamodule.distribution.potential(sample)
    np.save("sample.npy",sample.detach().cpu().numpy())
    np.save("traj.npy",traj.detach().cpu().numpy())
    np.save("potential.npy",potential.detach().cpu().numpy())
    plot_potential(potential, trainer.logger)
