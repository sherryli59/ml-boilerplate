from pytorch_lightning import LightningModule, LightningDataModule, seed_everything
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf
import torch
import logging

logger = logging.getLogger(__name__)

def load_experiment(log_dir: str, checkpoint: str):
    """
      Loads an existing model and its dataloader.

      args:
        path (string): The path to the log folder
        checkpoint (string): the name of the checkpoint file
    """
    # load conf
    config: DictConfig = OmegaConf.load(log_dir + '/.hydra/config.yaml')
    logger.info("\n" + OmegaConf.to_yaml(config))
    # reinitialize model and datamodule
    #model = get_class(config.model._target_)
    datamodule: LightningDataModule = instantiate(config.datamodule)
    datamodule.setup()
    if "seed" in config:
        seed_everything(config.seed)
    model:LightningModule = instantiate(config.model)
    checkpoint_model = torch.load(checkpoint)
    model.load_state_dict(checkpoint_model["state_dict"], strict=False)
    #model = model.load_from_checkpoint(checkpoint)
    return model, datamodule, config
