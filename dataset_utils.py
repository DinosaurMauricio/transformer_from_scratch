import random
from torch.utils.data import DataLoader, Subset, random_split
from datasets import load_dataset
from tokenizer_utils import get_or_build_tokenizer
from dataset import BilingualDataset


def get_ds(config):
    ds_raw = load_dataset(
        "opus_books", f"{config['lang_src']}-{config['lang_tgt']}", split="train"
    )
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    subset_size = int(config.ds_size * len(ds_raw))
    indices = random.sample(range(len(ds_raw)), subset_size)
    ds_raw = Subset(ds_raw, indices)

    train_size = int(0.9 * len(ds_raw))
    val_size = len(ds_raw) - train_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_size, val_size])

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_loader, val_loader, tokenizer_src, tokenizer_tgt
