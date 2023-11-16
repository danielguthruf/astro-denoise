from astro_denoise.factories.dataset import DatasetFactory


def initalizer(cfg):
    dataset = DatasetFactory(data=cfg.data.name)
    print("hi")
