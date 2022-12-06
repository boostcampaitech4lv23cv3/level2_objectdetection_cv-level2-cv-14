import json
import pandas as pd
from split_traindata import split_dataset


def sudo(submis_path="submission.csv", random_seed=2022, threshold=0.95):
    """
    Args:
        submis_path (str, optional): sudolabel할 submisiion 파일의 path를 입력해주세요.".
        random_seed (int, optional): random_seed 값 입니다.
        threshold (float, optional): submisiion의 bbox의 threshold 값을 지정해 주세요
    """
    with open("../dataset/test.json") as f:
        data_test = json.load(f)
    with open("../dataset/train.json") as f:
        data_train = json.load(f)
    submission = pd.read_csv(submis_path)
    annotations, count = [], 0
    for idx, pred in enumerate((submission.PredictionString.str.split(" "))):
        if not isinstance(pred, float):
            N = len(pred) // 6
            for i in range(N):
                cla, cof, x, y, w, h = pred[6 * i : 6 * i + 6]
                if float(cof) > threshold:
                    ano = {
                        "image_id": idx + 10000,
                        "category_id": int(cla),
                        "area": float(w) * float(h),
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "iscrowd": 0,
                        "id": count + len(data_train["annotations"]),
                    }
                    annotations.append(ano)
                    count += 1

    data_test["annotations"] = annotations
    data_test["images"] = [
        data_test["images"][i]
        for i in set(i["image_id"] - 10000 for i in data_test["annotations"])
    ]
    print("추가된 annotations",len(data_test["annotations"]))
    print("추가된 images",len(data_test["images"]))
    for i in data_test["images"]:
        i["id"] += 10000
    sudo_train, _ = split_dataset(
        "../dataset/train.json", val_ratio=0.2, random_seed=random_seed
    )
    sudo_train["annotations"] += data_test["annotations"]
    sudo_train["images"] += data_test["images"]

    sudo_train_json = f"../dataset/sudo_train_{random_seed}.json"
    with open(sudo_train_json, "w") as val_writer:
        json.dump(sudo_train, val_writer)


if __name__ == "__main__":
    sudo(submis_path="output.csv")
