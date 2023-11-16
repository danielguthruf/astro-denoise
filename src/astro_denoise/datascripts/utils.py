import os


def get_data_dict(data: list) -> dict:
    data_dict: dict = {"train": [], "validation": [], "test": []}
    for d in data:
        data_path = os.path.join("./data/processed", d)
        print(data_path)

        for split in ["train", "validation", "test"]:
            split_folder_path = os.path.join(data_path, split)

            if os.path.exists(split_folder_path):
                file_paths = [
                    os.path.join(split_folder_path, filename)
                    for filename in os.listdir(split_folder_path)
                ]
                data_dict[split].extend(file_paths)

    return data_dict
