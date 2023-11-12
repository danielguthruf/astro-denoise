import os


def get_data_dict(data_folders):
    if isinstance(data_folders, str):
        data_folders = [data_folders]

    data_dict = {"train": [], "validation": [], "test": []}

    for data_folder in data_folders:
        data_folder_path = os.path.join("../data/processed", data_folder)
        print(data_folder_path)

        for split in ["train", "validation", "test"]:
            split_folder_path = os.path.join(data_folder_path, split)

            if os.path.exists(split_folder_path):
                file_paths = [
                    os.path.join(split_folder_path, filename)
                    for filename in os.listdir(split_folder_path)
                ]
                data_dict[split].extend(file_paths)

    return data_dict
