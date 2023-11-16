from typing import Union

from astro_denoise.datascripts.dataset import BaseDataset
from astro_denoise.datascripts.utils import get_data_dict
from astro_denoise.transformations.transforms import get_transformation_pipeline


class DatasetFactory:
    def __init__(self, data: Union[str, list] = "test_data"):
        if not isinstance(data, list):
            self.data = [data]

        self.data_dict = get_data_dict(self.data)
        self.datasets = {
            "Base": BaseDataset,
        }

    def create_dataset(self, data_split="train", dataset="Base", data_pipeline="base"):
        transformation_pipeline = get_transformation_pipeline(data_pipeline, data_split)

        if dataset not in self.datasets:
            raise ValueError(f"Unknown Dataset  name: {dataset}")
        return self.datasets[dataset](self.data_dict[data_split], transformation_pipeline)
