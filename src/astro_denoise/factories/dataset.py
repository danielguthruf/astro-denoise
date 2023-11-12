from astro_denoise.datascripts.dataset import BaseDataset
from astro_denoise.datascripts.utils import get_data_dict
from astro_denoise.transformations.transforms import get_transformation_pipeline


class DatasetFactory:
    def __init__(self, data_folders="test_data"):
        self.data_dict = get_data_dict(data_folders=data_folders)

        self.datasets = {
            "BaseDataset": BaseDataset,
        }

    def create_dataset(
        self, data_split="train", dataset_name="BaseDataset", transformation_pipeline_name="basic"
    ):
        transformation_pipeline = get_transformation_pipeline(
            transformation_pipeline_name, data_split
        )

        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown Dataset  name: {dataset_name}")
        return self.datasets[dataset_name](self.data_dict[data_split], transformation_pipeline)
