"""Marian enc-dec for Track C1 (custom SPM vocab)."""

from __future__ import annotations

from transformers import MarianConfig, MarianMTModel

from src.training.spm_tokenizer import LegalSpmTokenizer


def build_legal_mt_config(tokenizer: LegalSpmTokenizer, **kw) -> MarianConfig:
    d = {
        'd_model': 512,
        'encoder_layers': 6,
        'decoder_layers': 6,
        'encoder_attention_heads': 8,
        'decoder_attention_heads': 8,
        'encoder_ffn_dim': 2048,
        'decoder_ffn_dim': 2048,
        'max_position_embeddings': 512,
        'dropout': 0.1,
    }
    d.update(kw)
    drop = float(d['dropout'])
    return MarianConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=int(d['d_model']),
        encoder_layers=int(d['encoder_layers']),
        decoder_layers=int(d['decoder_layers']),
        encoder_attention_heads=int(d['encoder_attention_heads']),
        decoder_attention_heads=int(d['decoder_attention_heads']),
        encoder_ffn_dim=int(d['encoder_ffn_dim']),
        decoder_ffn_dim=int(d['decoder_ffn_dim']),
        max_position_embeddings=int(d['max_position_embeddings']),
        dropout=drop,
        attention_dropout=drop,
        activation_dropout=drop,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
        share_encoder_decoder_embeddings=True,
        scale_embedding=True,
        is_encoder_decoder=True,
    )


def build_legal_mt_model(
    tokenizer: LegalSpmTokenizer,
    model_cfg: dict | None = None,
) -> MarianMTModel:
    return MarianMTModel(build_legal_mt_config(tokenizer, **(model_cfg or {})))
