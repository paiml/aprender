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
