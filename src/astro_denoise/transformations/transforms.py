import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transformation_pipeline(pipeline_name, data_split):
    if pipeline_name == "basic":
        return basic_transform_pipeline(data_split)
    elif pipeline_name == "custom":
        pass
    else:
        raise ValueError("Invalid pipeline name. Choose 'basic', 'custom'.....")


def basic_transform_pipeline(data_split="train"):
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
