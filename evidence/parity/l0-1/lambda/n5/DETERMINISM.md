# determinism of the n=5 series (lambda)

All five runs of each model are BYTE-IDENTICAL to each other and to the canonical
record beside this directory, so the five copies carried no information the sha256
below does not. `stdev 0` understates it: the whole file is the same file.

| model | sha256 of every run 1..5 | == canonical record |
|---|---|---|
| qwen2.5-coder-1.5b-instruct-q4_k_m | `e1d05ed6e3cf0addfea05ece589df360ddc63ed45376e81f25e1153bbbb8060d` (1 distinct over 5) | yes |
| qwen2.5-coder-7b-instruct-q4_k_m | `879b7f7715119cf310ba694a870d5de3097a45d4dcc149733ac33d167855ac96` (1 distinct over 5) | yes |

The canonical records were re-taken with the model named RELATIVELY (`cd ~/models &&
apr parity ./<model>.gguf ...`) so no record carries a machine-specific path; the
`metrics` array and every other key are byte-identical to the absolute-path run —
only the `model` field differs. Same binary, same prompt, same numbers.
