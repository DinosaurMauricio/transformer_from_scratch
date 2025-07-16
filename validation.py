import torch
import wandb
from dataset import causal_mask
from pycocoevalcap.bleu.bleu import Bleu
from tqdm import tqdm


def greedy_decode(
    model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )
        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)


def run_validation(
    model,
    validation_ds,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    loss_fn,
    log,
):
    model.eval()
    count = 0
    total_loss = 0.0
    predicted_texts = []
    target_texts = []

    batch_iterator = tqdm(validation_ds, desc=f"Evaluation")
    with torch.no_grad():
        for batch in batch_iterator:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)
            assert encoder_input.size(0) == 1

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )
            proj_output = model.project(decoder_output)

            loss = loss_fn(proj_output.view(-1, proj_output.size(-1)), label.view(-1))

            total_loss += loss.item()

            model_out = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
            )

            pred = tokenizer_tgt.decode(
                model_out.detach().cpu().numpy(), skip_special_tokens=True
            )
            tgt = batch["tgt_text"][0]

            predicted_texts.append(pred)
            target_texts.append(tgt)

        if log:
            wandb.log({"loss_val": total_loss / len(validation_ds)}, commit=False)

    gts = {i: [target_texts[i]] for i in range(len(target_texts))}
    res = {i: [predicted_texts[i]] for i in range(len(predicted_texts))}

    bleu_scorer = Bleu(n=4)
    bleu_score, _ = bleu_scorer.compute_score(gts, res)

    if log:
        wandb.log(
            {
                "bleu-1": bleu_score[0],
                "bleu-2": bleu_score[1],
                "bleu-3": bleu_score[2],
                "bleu-4": bleu_score[3],
            }
        )
