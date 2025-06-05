# utils/data_loader.py
import pandas as pd
from datasets import Dataset, DatasetDict

def load_dataset_from_folder(folder_path):
    train = pd.read_csv(f"{folder_path}/train.csv")
    val = pd.read_csv(f"{folder_path}/validation.csv")
    test = pd.read_csv(f"{folder_path}/test.csv")

    return DatasetDict({
        "train": Dataset.from_pandas(train.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val.reset_index(drop=True)),
        "test": Dataset.from_pandas(test.reset_index(drop=True)),
    })
