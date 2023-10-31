"""Collection of mlflow utilities to enhance the mlflow logging"""
import logging
import shutil
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities import rank_zero_only, model_summary


log = logging.getLogger(__name__)



class MLFlowModelCheckpoint(ModelCheckpoint):
    """
    This class wraps ModelCheckpoint to allow the storing of checkpoints and model summaries to Mlflow instead
    """

    def __init__(
        self,
        mlf_logger: MLFlowLogger,
        del_ckpts_outside_mlflow: bool = True,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        mlf_logger: MLFlowLogger
            the logger for the current running experiment
        del_ckpts_outside_mlflow: bool
            if True, we delete the 'normal' Lightning checkpoints after storing them in mlflow,
            this removes the redundancy and saves space
        args
        kwargs
        """
        super().__init__(*args, **kwargs)
        self.del_ckpts_outside_mlflow = del_ckpts_outside_mlflow
        self.mlf_logger = mlf_logger
        #self.run_id = self.mlf_logger.run_id

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        super().on_train_start(trainer, pl_module)
        self.store_model_summary(pl_module)

    def on_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        super().on_train_end(trainer, pl_module)
        # we could insert logic here to save models on epoch end -- currently we save them at train_end

    def on_train_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # save all the checkpoints that can be found in the checkpoint folder to mlflow
        self.store_models(trainer)

    @rank_zero_only
    def store_models(self,trainer=None) -> None:
        """
        Stores the models found in the lightning checkpoint folder into mlflow as artifacts

        Returns
        -------

        """
        # log.info("Exporting checkpoints to mlflow")
        # paths = Path(self.dirpath).glob("*")
        # checkpoint_paths = [
        #     x for x in paths if x.is_file()
        # ]  # these are the checkpoints saved by lightning

        # for path in checkpoint_paths:
        #     self.mlf_logger.experiment.log_artifact(
        #         self.mlf_logger.run_id, path, artifact_path="models"
        #     )
        
        # # delete the lightning checkpoints, since they are saved to mlflow now
        # if self.del_ckpts_outside_mlflow:
        #     log.info(f"Deleting checkpoints at {self.dirpath}")
        #     shutil.rmtree(self.dirpath)

        if trainer is not None:
            self.mlf_logger.experiment.log_param(
            self.mlf_logger.run_id, "ml-flow/best_model_path", trainer.checkpoint_callback.best_model_path)

    @rank_zero_only
    def store_model_summary(self, pl_module: "pl.LightningModule") -> None:
        """
        Stores a summary of the model as 'model.txt' in mlflow artifacts
        Parameters
        ----------
        pl_module: LightningModule
            the lightning module of which a summary is to be stored

        Returns
        -------

        """
        summary = model_summary.summarize(pl_module, max_depth=10).__str__()
        self.mlf_logger.experiment.log_text(
            self.mlf_logger.run_id, summary, artifact_file="model_summary.txt"
        )