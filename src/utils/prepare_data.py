# scripts/prepare_data.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_and_save_datasets(input_path, output_dir):
    df = pd.read_csv(input_path)
    df["Review ID clean"] = df["Review ID"].str.extract(r"(REV\d+)")
    unique_ids = df["Review ID clean"].unique()

    train_ids, temp_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    train_df = df[df["Review ID clean"].isin(train_ids)].drop(columns=["Review ID clean"])
    val_df = df[df["Review ID clean"].isin(val_ids)].drop(columns=["Review ID clean"])
    test_df = df[df["Review ID clean"].isin(test_ids)].drop(columns=["Review ID clean"])

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print(f"Saved split datasets to: {output_dir}")

if __name__ == "__main__":
    prepare_and_save_datasets(
        input_path="data/raw/full_dataset.csv",        # đường dẫn đến file gốc
        output_dir="data/processed"     # nơi lưu các file đã chia
    )
