from pathlib import Path


def get_config():
    # Dummy config. Replace with your own.
    return {
        "lang_src": "en",
        "lang_tgt": "de",
        "seq_len": 128,
        "batch_size": 32,
        "d_model": 512,
        "lr": 0.0001,
        "num_epochs": 10,
        "tokenizer_file": "./tokenizer_{}.json",
        "model_folder": "./models/",
        "preload": None,
    }


def get_weights_file_path(config, epoch):
    return Path(config["model_folder"]) / f"model_{epoch}.pt"


def latest_weights_file_path(config):
    model_dir = Path(config["model_folder"])
    weights_files = list(model_dir.glob("model_*.pt"))
    if not weights_files:
        return None
    return str(sorted(weights_files)[-1])
