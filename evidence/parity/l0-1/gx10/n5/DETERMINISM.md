# determinism of the n=5 series (gx10)

All five runs of each model are BYTE-IDENTICAL to each other and to the canonical
record beside this directory, so the five copies carried no information the sha256
below does not. `stdev 0` understates it: the whole file is the same file.

| model | sha256 of every run 1..5 | == canonical record |
|---|---|---|
| qwen2.5-coder-1.5b-instruct-q4_k_m | `9631f44b0034fe292866a8bf798e9cbeafb0d6da351a1659471c2d3cf66b19d1` (1 distinct over 5) | yes |
| qwen2.5-coder-7b-instruct-q4_k_m | `163aa8ecdd28e3de91032869151c2d226812b0422f8dbd65fa57e7db391f67e7` (1 distinct over 5) | yes |

The canonical records were re-taken with the model named RELATIVELY (`cd ~/models &&
apr parity ./<model>.gguf ...`) so no record carries a machine-specific path; the
`metrics` array and every other key are byte-identical to the absolute-path run —
only the `model` field differs. Same binary, same prompt, same numbers.
