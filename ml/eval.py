import logging
import numpy as np
from typing import Optional
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning import seed_everything
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from ml.utils import config_utils
import torch

log = logging.getLogger(__name__)

def plot_potential(potential,logger):
    fig, ax = plt.subplots()
    potential = potential.detach().cpu().numpy()
    #max = min(potential.max(),0)
    max = potential.max()
    ax.hist(potential, bins=100, density=True,range=(potential.min(),max))
    ax.set_xlabel("Potential energy")
    ax.set_ylabel("Density")
    fig.savefig("potential.png")
    plt.close()
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

    #if 'seed' in config:
    #    seed_everything(config.seed)

    model._device = trainer.strategy.root_device

    # Send some parameters from config to all lightning loggers
    # log.info('Logging hyperparameters!')
    # config_utils.log_hyperparameters(
    #     config=config,
    #     model=model,
    #     datamodule=datamodule,
    #     trainer=trainer,
    #     callbacks=[],
    #     logger=trainer.logger,
    # )
    # add your evaluation logic here
    # result = trainer.test(model=model, datamodule=datamodule)
    # prob = result[0]["logp"]
    # np.save("prob_sample.npy",prob.detach().cpu().numpy())
    print(config.sample)
    with torch.no_grad():
        out = model.sample(**config.sample)
    sample = out["x"]
    np.save("sample.npy",sample.detach().cpu().numpy())
    if "traj" in out:
        traj = out["traj"]
        np.save("traj.npy",traj.detach().cpu().numpy())
        
    if "logp" in out:
        prob = out["logp"]
        np.save("prob.npy",prob.detach().cpu().numpy())
    ## Sample visualization.
    # sample = sample.clamp(0.0, 1.0)
    # import matplotlib.pyplot as plt
    # from torchvision.utils import make_grid

    # sample_grid = make_grid(sample, nrow=int(np.sqrt(config.sample.nsamples)))
    # plt.figure(figsize=(6,6))
    # plt.axis('off')
    # plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    # plt.savefig("sample.png")

    potential = datamodule.distribution.potential(sample)
    np.save("potential.npy",potential.detach().cpu().numpy())
    plot_potential(potential, trainer.logger)
    # bad_idx = (potential>0).flatten()
    # init = out["init"]
    # log_prob = (-init**2/2).sum(dim=(1,2))
    # log_prob_bad = log_prob[bad_idx]
    # plt.hist(log_prob_bad.detach().cpu().numpy(),alpha=0.4,bins=20,label="bad")
    # plt.hist(log_prob.detach().cpu().numpy(),alpha=0.4,bins=50)
    # plt.legend()
    # plt.savefig("log_prob.png")
    # good_idx = (log_prob>-5).flatten()
    # out_good = model.sample(init=init[good_idx],**config.sample)
    # sample_good = out_good["x"]
    # potential_good = datamodule.distribution.potential(sample_good)
    # plot_potential(potential_good, trainer.logger)

    



