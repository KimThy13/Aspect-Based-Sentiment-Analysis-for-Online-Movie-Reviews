import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

def load_and_split_dataset(csv_path, id_column="Review ID", test_size=0.2, val_size=0.1, random_state=42):
    # 1. Đọc dữ liệu từ file CSV
    df = pd.read_csv(csv_path)

    # 2. Làm sạch Review ID để gom nhóm các component về cùng 1 review gốc
    df["Review ID clean"] = df[id_column].str.extract(r"(REV\d+)")

    # 3. Lấy các Review ID duy nhất
    unique_reviews = df["Review ID clean"].unique()

    # 4. Chia train (80%) và tạm (20%)
    train_ids, temp_ids = train_test_split(unique_reviews, test_size=test_size, random_state=random_state)

    # 5. Chia tiếp tạm thành validation và test
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=random_state)

    # 6. Tạo DataFrame cho từng tập
    train_df = df[df["Review ID clean"].isin(train_ids)].drop(columns=["Review ID clean"])
    val_df = df[df["Review ID clean"].isin(val_ids)].drop(columns=["Review ID clean"])
    test_df = df[df["Review ID clean"].isin(test_ids)].drop(columns=["Review ID clean"])

    # 7. Chuyển sang DatasetDict
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
    })

    return dataset
