import pytorch_lightning as pl
import logging
import typing as t
import time


class SafeMLFlowLogger(pl.loggers.mlflow.MLFlowLogger):
    """
    Subclassing `pl.loggers.mlflow.MLFlowLogger` and overriding metrics methods to handle unsucesful uploads.
    """

    def log_metrics(
        self, metrics: t.Dict[str, float], step: t.Optional[int] = None
    ) -> None:
        # It sometimes happens that connection errors out.
        try:
            super().log_metrics(metrics=metrics, step=step)

        except Exception:
            logging.exception("Exception while logging metrics to MLFlow.")
            time.sleep(0.1)
