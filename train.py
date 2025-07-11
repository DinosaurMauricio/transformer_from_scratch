import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

from dataset_utils import get_ds
from model_utils import get_model
from validation import run_validation
from utils import get_weights_file_path, latest_weights_file_path


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(
        config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)
    writer = SummaryWriter("experiment_name")

    initial_epoch, global_step = 0, 0
    preload = config["preload"]
    model_filename = (
        latest_weights_file_path(config)
        if preload == "latest"
        else (get_weights_file_path(config, preload) if preload else None)
    )

    if model_filename:
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        for batch in batch_iterator:
            model.train()
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )
            proj_output = model.project(decoder_output)

            loss = loss_fn(proj_output.view(-1, proj_output.size(-1)), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # writer.add_scalar("train loss", loss.item(), global_step)
            # writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if config.log:
                wandb.log({"loss": loss}, commit=False)

        # run_validation(
        #    model,
        #    val_dataloader,
        #    tokenizer_src,
        #    tokenizer_tgt,
        #    config["seq_len"],
        #    device,
        #    lambda msg: batch_iterator.write(msg),
        #    global_step,
        #    config.log,
        #    loss_fn,
        # )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            get_weights_file_path(config, f"{epoch:02d}"),
        )
