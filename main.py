import logging
import hydra
from omegaconf import DictConfig, OmegaConf

import os
os.environ["HYDRA_FULL_ERROR"] = "1"

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    logger.info("\n" + OmegaConf.to_yaml(config))
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.train import train
    from src.utils import config_utils

    config_utils.extras(config)
    # Train model
    return train(config)

    
    



if __name__ == "__main__":
    main()
