# determinism of the n=5 series (gx10, post-fix)

All five runs of each model are BYTE-IDENTICAL to each other and to the canonical
record beside this directory, so the five copies carried no information the sha256
below does not. `stdev 0` understates it: the whole file is the same file.

| model | sha256 of every run 1..5 | == canonical record |
|---|---|---|
| qwen2.5-coder-1.5b-instruct-q4_k_m | `af98dc311d341bd61b12d597d9f04bd66a582b9c5367cbceca806bc8a6deb17e` (1 distinct over 5) | yes |
| qwen2.5-coder-7b-instruct-q4_k_m | `ee804a3264fc2f5ae2d8849b0886b2f6a7448a9341f0545750ff24fa838b40f1` (1 distinct over 5) | yes |

The canonical records and the two `--per-op` summaries were then re-taken with the model
named RELATIVELY (`cd ~/models && apr parity ./<model>.gguf …`) on the same binary
(`apr 0.65.2 (ecbd8e4f2)`, sha256 `8cdb7c1a668127db`), so no record carries a
machine-specific path. Every `metrics`/`rows` array and every other key is
byte-identical to the absolute-path run — only `model` differs.
