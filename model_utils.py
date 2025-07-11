from model import build_transformer


def get_model(config, vocab_src_len, vocab_tgt_len):
    return build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
    )
