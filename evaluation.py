import logging
from typing import List
#import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
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
    from ml.eval import eval
    from ml.utils import model, config_utils

    config_utils.extras(config)
    mlflow.set_tracking_uri('file://' + utils.get_original_cwd() + '/mlruns')

    # get the hydra logdir using the exp_id
    if config.get('run_id'):
        run_id = config.run_id
    else:
        experiment_name = config.logger.mlflow.experiment_name
        #print current working directory
        current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
        experiment_id=current_experiment['experiment_id']
        #df = mlflow.search_runs([experiment_id], order_by=["metrics.val/loss DESC"])
        df = mlflow.search_runs([experiment_id], order_by=["attributes.start_time DESC"],
                            filter_string="tags.stage = 'train' and tags.mlflow.runName = '%s'"%config.run_name,)
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        if str(hydra_cfg.mode) == "RunMode.MULTIRUN":
            idx = hydra_cfg.job.num
        else:
            idx = 0
        run_id = df.loc[idx,'run_id']    # load the saved model and datamodule
    print("run_id:",run_id)
    client = mlflow.tracking.MlflowClient(
        tracking_uri=config.logger.mlflow.tracking_uri)
    run = client.get_run(run_id).to_dictionary()
    model_path = run['data']['params']['ml-flow/best_model_path']
    params = OmegaConf.create(run['data']['params'])
    log_dir = run['data']['params']['hydra/log_dir']
    print("log directory:", log_dir)
    #log_dir = utils.get_original_cwd() + '/' + log_dir
    model, datamodule, exp_config = model.load_experiment(log_dir=log_dir,checkpoint=model_path)
    user_specified_tags = {}  
    user_specified_tags[MLFLOW_RUN_NAME] = config.run_name+"_eval"
    tags = context_registry.resolve_tags(user_specified_tags)
    # instanciate mlflow and the trainer for the evaluation
    mlf_logger = utils.instantiate(
        config.logger.mlflow, run_id=run_id, experiment_name=exp_config.logger.mlflow.experiment_name)
    trainer = utils.instantiate(
        config.trainer, callbacks=[], logger=[mlf_logger], _convert_='partial'
    )
    model._device = trainer.strategy.root_device
    return eval(config, model, trainer, datamodule)


if __name__ == "__main__":
    main()
