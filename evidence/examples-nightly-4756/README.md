# examples-nightly reds — receipts (#4756)

Before: CI run 37169058144 (yoga-build, main @316dee2cd4), 36 rows fail 29 + timeout 7.

After, on intel, debug builds, `scripts/dogfood_examples.sh --timeout-secs 180`, load ~100:

- `after-intel.tsv`: the 36 rows on branch 51b6d69fe8 minus its last two fixes:
  pass 8, needs-args 11, needs-hardware 10, needs-data 4, needs-feature 1, fail 1, timeout 1.
- `rerun-parity_035.tsv`: parity_035 with the Ollama model probe. fail -> needs-data.
  Intel runs an Ollama server without phi2. CI has no server and hits the TCP probe.
- `performance_parity-cpu.txt`: no wgpu adapter (VK_ICD_FILENAMES=/nonexistent.json):
  rc 0 in 24 s, was 322 s. timeout -> pass.
  The GPU path (wgpu adapter present) still timed out at 1505 s in debug. CI's
  yoga-build has no adapter. A GPU runner would need its own fix.
- `rerun-bug_hunter.tsv`: bug_hunter_demo and design_by_contract after limiting the
  narrowed scan to debug builds (quorum round 1 finding): both pass.

Net for the 36 rows: pass 9, needs-args 11, needs-hardware 10, needs-data 5,
needs-feature 1, fail 0, timeout 0.

The CUDA rows read "cuInit failed" on intel (libcuda, no device). On CI they read
"CUDA driver not found (libcuda.so)", which the classifier's `libcuda\.so` pattern
matches. That CI row is inferred from the message and is not measured here.

## Notes for review (quorum round 2)

- The Q4_0/Q8_0 buffer change in performance_parity (20 -> 18, 36 -> 34 bytes per
  32-value block) fixes a bug and is not a workload reduction. Q4_0 is an f16
  scale plus 16 quant bytes (18), and Q8_0 is an f16 scale plus 32 int8 (34). The
  old sizes made dequantize return InvalidShape, which panicked the example in
  every build, release included.
- The `timeout` row for performance_parity in `after-intel.tsv` comes from the tree
  before the fused-attention debug reduction. Its after-fix measurement is
  `performance_parity-cpu.txt` (rc 0, 24 s), and it should be read in place of
  that row.
- performance_parity's GPU path in debug is not_measured as passing: 1505 s, rc 124
  on intel. That is a known gap on GPU runners, not part of the 36-row claim.
