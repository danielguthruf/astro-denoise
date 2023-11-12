from astro_denoise.factories.dataset import DatasetFactory


def initalizer(config, mode="training"):
    dataset_factory = DatasetFactory(config)
