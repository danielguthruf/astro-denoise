import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transformation_pipeline(data_pipeline_name: str, data_split: str) -> A.Compose:
    if data_pipeline_name == "base":
        return data_pipeline(data_split)
    elif data_pipeline_name == "custom":
        pass
    else:
        raise ValueError("Invalid pipeline name. Choose 'basic', 'custom'.....")


def data_pipeline(data_split: str = "train") -> A.Compose:
    if data_split == "train":
        return A.Compose(
            [A.Resize(256, 256), A.RandomCrop(224, 224), A.HorizontalFlip(), ToTensorV2()]
        )
    elif data_split == "validation":
        pass
    elif data_split == "test":
        pass
    else:
        raise ValueError("Invalid data_split name. Choose 'train','validation', 'test'.")
