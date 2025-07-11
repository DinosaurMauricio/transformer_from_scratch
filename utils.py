from pathlib import Path


def get_weights_file_path(config, epoch):
    return Path(config["model_folder"]) / f"model_{epoch}.pt"


def latest_weights_file_path(config):
    model_dir = Path(config["model_folder"])
    weights_files = list(model_dir.glob("model_*.pt"))
    if not weights_files:
        return None
    return str(sorted(weights_files)[-1])
