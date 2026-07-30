"""
Phase 3 -- Model adaptation.

Track D (production): train_nllb_lora + configs/training*.yaml
  subsample (A1/A2/Bp) -> LoRA on NLLB-600M -> dual-policy selection

Track C: spm_tokenizer, legal_mt_*, vocab_extend_nllb(_v2)
  C0 domain SPM; C1 Marian; C1c NLLB vocab-extend

Shared: common, config, cuda_backend, dist_utils, nllb_data
"""
