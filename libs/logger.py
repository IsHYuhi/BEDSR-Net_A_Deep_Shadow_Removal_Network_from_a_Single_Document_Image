from logging import getLogger

import pandas as pd

logger = getLogger(__name__)


class TrainLogger(object):
    def __init__(self, log_path: str, resume: bool) -> None:
        self.log_path = log_path
        self.columns = [
            "epoch",
            "lr",
            "train_time[sec]",
            "train_loss",
            #"train_acc@1",
            #"train_f1s",
            "val_time[sec]",
            "val_loss",
            #"val_acc@1",
            #"val_f1s",
        ]

        if resume:
            self.df = self._load_log()
        else:
            self.df = pd.DataFrame(columns=self.columns)

    def _load_log(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.log_path)
            logger.info("successfully loaded log csv file.")
            return df
        except FileNotFoundError as err:
            logger.exception(f"{err}")
            raise err

    def _save_log(self) -> None:
        self.df.to_csv(self.log_path, index=False)
        logger.debug("training logs are saved.")

    def update(
        self,
        epoch: int,
        lr: float,
        train_time: int,
        train_loss: float,
        # train_acc: float,
        # train_f1s: float,
        val_time: int,
        val_loss: float,
        # val_acc1: float,
        # val_f1s: float,
    ) -> None:
        tmp = pd.Series(
            [
                epoch,
                lr,
                train_time,
                train_loss,
                # train_acc,
                # train_f1s,
                val_time,
                val_loss,
                # val_acc1,
                # val_f1s,
            ],
            index=self.columns,
        )

        self.df = self.df.append(tmp, ignore_index=True)
        self._save_log()

        logger.info(
            f"epoch: {epoch}\tepoch time[sec]: {train_time + val_time}\tlr: {lr}\t"
            f"train loss: {train_loss:.4f}\tval loss: {val_loss:.4f}\t"
            #f"val_acc1: {val_acc1:.5f}\tval_f1s: {val_f1s:.5f}"
        )