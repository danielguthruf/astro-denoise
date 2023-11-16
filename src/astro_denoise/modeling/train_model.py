import hydra
from omegaconf import DictConfig

from astro_denoise.modeling.utils import initalizer


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    initalizer(cfg)
    # initalize dataset
    # initalize loader
    # initalize factories
    #


if __name__ == "__main__":
    main()
