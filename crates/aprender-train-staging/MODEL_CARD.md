# Model Card: entrenar Training Library

Following Mitchell et al. (2019) "Model Cards for Model Reporting"

## Model Details

- **Library**: entrenar v0.7.5
- **Type**: Training library for transformer language models
- **Architecture**: LLaMA-style decoder-only transformer (configurable)
- **Supported sizes**: 50M to 1B+ parameters
- **Precision**: FP32 master weights, BF16 compute (optional)
- **Training**: GPU-resident CUDA, CPU fallback
- **License**: Apache-2.0 / MIT dual license

## Intended Use

- **Primary**: Pre-training and fine-tuning transformer language models
- **Users**: ML researchers and engineers using the sovereign AI stack
- **Out of scope**: Real-time inference (use realizar for inference)

## Training Data

- Pre-tokenized Parquet format (ByteLevel BPE tokenizer)
- Data pipeline: alimentar (dedup, filter, FIM, mix, curriculum)
- No PII in training data (quality filtering removes personal information)

## Evaluation

- Validation perplexity during training (per checkpoint)
- HumanEval pass@k for code models
- Contamination detection between train/eval sets
- Perplexity-benchmark correlation tracking

## Ethical Considerations

- Sovereign data governance: all training local, no cloud API calls
- Model weight encryption available (BLAKE3-derived keystream)
- Audit trail via SQLite experiment tracking
- Bias testing through fairness metric evaluation

## Limitations

- Single-GPU training primarily (DDP for multi-GPU)
- No FP8 support (BF16 is current precision floor)
- Code-focused tokenizer (not optimal for natural language)

## Quantitative Analyses

| Model | Steps | Loss | Perplexity | MFU |
|-------|-------|------|------------|-----|
| 50M (12L) | 5000 | ~4.5 | ~90 | ~30% |
| 350M (24L) | 50 | 5.92 | ~370 | ~25% |

## References

- Hu et al. (2021) "LoRA: Low-Rank Adaptation"
- Dettmers et al. (2023) "QLoRA: Efficient Finetuning"
- Touvron et al. (2023) "LLaMA: Open Foundation Models"
