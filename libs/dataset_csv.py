import dataclasses
from logging import getLogger

logger = getLogger(__name__)

__all__ = ["DATASET_CSVS"]


@dataclasses.dataclass(frozen=True)
class DatasetCSV:
    train: str
    val: str
    test: str


DATASET_CSVS = {
    # paths from `src` directory
    "Jung": DatasetCSV(
        train="./csv/Jung/train.csv",
        val="./csv/Jung/val.csv",
        test="./csv/Jung/test.csv",
    ),
}
