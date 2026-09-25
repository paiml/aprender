# apr parity-oracle cross-check against the recorded Python comparator (#4444)

The verb and `compare_raw_logits.py` are two implementations of one comparison.
The verb was run on the pair the Python comparator measured
(`evidence/parity/l0-1/intel/qwen35-apr-vs-reference/comparison.json`).

| input | sha256 |
|-------|--------|
| reference: `intel:~/parity-ref/qwen35-0.8b-q4km-d1d3c3396/qwen35-0.8b-q4km-d1d3c3396.rawlogits.bin` (llama.cpp d1d3c3396 producer) | `1d5eaf129545aa6087cd6c49792f9774b61caa874fccefdd79e6164afa15bb93` |
| subject: `intel:~/parity-ref/apr-3091-subject/apr-intel-run1.bin` (apr CPU, #3114 loop) | `f6f792649014dcef6eec1afaf96852a293cc95c6cfceb1c88d62e16ee36c0252` |

Binary: `apr 0.69.0 (2c8d8d797)`, release, built on lambda from this branch.

```
apr parity-oracle --reference <ref> --subject <sub> --threshold 0.99 \
  --threshold-basis 'cross-check only (#4444): not an admission threshold; ...' -o receipt.json
# rc 0
```

| field | Python (comparison.json) | apr parity-oracle |
|-------|--------------------------|-------------------|
| n_positions | 78 | 78 |
| min cosine | 0.995905 @ pos 4 | 0.9959052275724324 @ pos 4 |
| argmax mismatches | [2, 28, 36, 73] | [2, 28, 36, 73] |
| max abs diff | 1.342222 | 1.342221736907959 |

Control: the same pair at `--threshold 0.9999` exits **13 RED**, and the failure message names position 4.

**0.99 is NOT an admission threshold.** `evidence/parity/thresholds.yaml`
`qwen3.5-0.8b@cpu-vs-llama.cpp` still has no measured basis. A cell declaring
`parity.oracle: llama.cpp@d1d3c3396` needs that row measured first.
`receipt-crosscheck.json` sha256: `c716fd1dde22e3bfc5faa9917eb22f9ede4f37fb479db9c30c5d0d5ea8ed87cf`.
