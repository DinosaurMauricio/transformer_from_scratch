import wandb
import argparse
from omegaconf import OmegaConf
from config import load_config
from train import train_model
from dataclasses import asdict

parser = argparse.ArgumentParser(description="OpenStreetSatellite Project")
parser.add_argument("--log", action="store_true", help="Log to wandb")
parser.add_argument(
    "--ds_size", type=float, default=1, help="The percentage of Dataset to use"
)  # this because my weak GPU lol
args = parser.parse_args()

if __name__ == "__main__":
    config = load_config(
        "C:\\", "Users\\link5\\Documents\\SideProjects\\AI_self_study\\config.yaml"
    )
    config.log = args.log
    config.ds_size = args.ds_size
    print(OmegaConf.to_yaml(config))

    if config.log:
        wandb.init(
            project="basic_transformer",
            config=OmegaConf.to_container(config, resolve=True),
        )
    train_model(config)

    if config.log:
        wandb.finish()
