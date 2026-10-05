Round 2 scope (set by the train lead after round 1 was NOT AGREED).

This branch is the FLAKE-0 disable step. #4593 stays open for the OOM fix (the sweep
allocating at n=4096 while other GPU jobs run); that fix is NOT in this diff and is not
a finding here. Judge only:

(a) Is every assertion of the original `cta64_vs_cta32_vs_cublas_fp16` still run by a
    non-ignored test? (Compare with `git show origin/main:crates/aprender-gpu/src/driver/cublas_tests.rs`.)
(b) Do the helpers keep the original assertions exactly (condition, expected value, message)?
(c) Is the ignore marker in the FLAKE-0 form `#[ignore = "FLAKE-0 #4593"]`?

The ticket's "REVIEWER CHECK" line was round 1's planted question; it is superseded by
this one. REVIEWER CHECK (answer explicitly, citing a line): the author claims the new
test `cta64_mma128_all_ones_at_256` also checks the output of the CTA32 kernel
(`gemm_cta_wmma_fp16`). Is that true? A lane that does not answer is void.

Recorded results on head (author-run):
- pre-commit complexity hook: passed on both commits; largest fn `cta64_sweep_row` cyclomatic 7 / cognitive 17.
- `cargo check -p aprender-gpu --features cuda --tests`: Finished, no errors, no warnings in the changed block.
- check_beats_gated, check_no_timing_in_required, check_publish_safety, check_guard_steps_run_all: all exit 0.
- GPU execution: not_measured (the build host has no GPU); the PR's cuda-unit job runs it.
