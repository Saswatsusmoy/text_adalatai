"""Small Marian enc-dec for Track C1 (custom SPM vocab)."""

from __future__ import annotations

from transformers import MarianConfig, MarianMTModel

from src.training.spm_tokenizer import LegalSpmTokenizer


def build_legal_mt_config(
    tokenizer: LegalSpmTokenizer,
    d_model: int = 512,
    encoder_layers: int = 6,
    decoder_layers: int = 6,
    encoder_attention_heads: int = 8,
    decoder_attention_heads: int = 8,
    encoder_ffn_dim: int = 2048,
    decoder_ffn_dim: int = 2048,
    max_position_embeddings: int = 512,
    dropout: float = 0.1,
) -> MarianConfig:
    return MarianConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        encoder_attention_heads=encoder_attention_heads,
        decoder_attention_heads=decoder_attention_heads,
        encoder_ffn_dim=encoder_ffn_dim,
        decoder_ffn_dim=decoder_ffn_dim,
        max_position_embeddings=max_position_embeddings,
        dropout=dropout,
        attention_dropout=dropout,
        activation_dropout=dropout,
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
    model_cfg = model_cfg or {}
    config = build_legal_mt_config(
        tokenizer,
        d_model=int(model_cfg.get('d_model', 512)),
        encoder_layers=int(model_cfg.get('encoder_layers', 6)),
        decoder_layers=int(model_cfg.get('decoder_layers', 6)),
        encoder_attention_heads=int(model_cfg.get('encoder_attention_heads', 8)),
        decoder_attention_heads=int(model_cfg.get('decoder_attention_heads', 8)),
        encoder_ffn_dim=int(model_cfg.get('encoder_ffn_dim', 2048)),
        decoder_ffn_dim=int(model_cfg.get('decoder_ffn_dim', 2048)),
        max_position_embeddings=int(model_cfg.get('max_position_embeddings', 512)),
        dropout=float(model_cfg.get('dropout', 0.1)),
    )
    return MarianMTModel(config)
