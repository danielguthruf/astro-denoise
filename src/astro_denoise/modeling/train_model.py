import argparse

from astro_denoise.modeling.utils import initalizer


def main():
    initalizer(config, mode="training")
    # initalize dataset
    # initalize loader
    # initalize factories
    #


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument Parser")

    parser.add_argument(
        "--config", type=str, default="default", help="name of you config file yaml"
    )
    args = parser.parse_args()
