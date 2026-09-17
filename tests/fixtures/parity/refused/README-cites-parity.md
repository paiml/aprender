# Fixture README (NOT the project README)

Used by scripts/check_model_parity.sh --self-test to exercise the RED arm of the
UNMEASURED-TOOL rule: a published GPU=CPU claim about a model the tool cannot
measure is a claim with no measurement behind it.

## Performance

| Model | Backend | Note |
|-------|---------|------|
| qwen3-coder-30b | CUDA | GPU=CPU parity verified over 64 positions |
