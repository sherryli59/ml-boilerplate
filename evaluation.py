import logging
from typing import List
#import dotenv
import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers import Logger
import mlflow
from mlflow.tracking.context import registry as context_registry
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
#dotenv.load_dotenv(override=True)

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs/", config_name="eval_config.yaml")
def main(config: DictConfig):
    logger.info(config)

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from hydra import utils
    from src.eval import eval
    from src.utils import model, config_utils

    config_utils.extras(config)
    mlflow.set_tracking_uri('file://' + utils.get_original_cwd() + '/mlruns')

    log_dir = None
    # get the hydra logdir using the exp_id
    if config.get('log_dir'):
        log_dir = config.log_dir
    else:
        if config.get('run_id'):
            run_id = config.run_id
        else:
            experiment_name = config.logger.mlflow.experiment_name
            #print current working directory
            current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
            experiment_id=current_experiment['experiment_id']
            #df = mlflow.search_runs([experiment_id], order_by=["metrics.val/loss DESC"])
            df = mlflow.search_runs([experiment_id], order_by=["attributes.start_time DESC"],
                                filter_string="tags.stage = 'train' and tags.mlflow.runName = %s"%config.run_name,)
            run_id = df.loc[0,'run_id']    # load the saved model and datamodule
        print("run_id:",run_id)
        client = mlflow.tracking.MlflowClient(
            tracking_uri=config.logger.mlflow.tracking_uri)
        data = client.get_run(run_id).to_dictionary()
        log_dir = data['data']['params']['hydra/log_dir']
        print("log directory:", log_dir)
    log_dir = utils.get_original_cwd() + '/' + log_dir
    model, datamodule, exp_config = model.load_experiment(config,log_dir)

    user_specified_tags = {}  
    user_specified_tags[MLFLOW_RUN_NAME] = config.run_name+"_eval"
    tags = context_registry.resolve_tags(user_specified_tags)
    # instanciate mlflow and the trainer for the evaluation
    mlf_logger = utils.instantiate(
        config.logger.mlflow, tags=tags, experiment_name=exp_config.logger.mlflow.experiment_name)
    trainer = utils.instantiate(
        config.trainer, callbacks=[], logger=[mlf_logger], _convert_='partial'
    )

    return eval(config, model, trainer, datamodule)


if __name__ == "__main__":
    main()
