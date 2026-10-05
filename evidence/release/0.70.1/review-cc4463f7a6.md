# Review of cc4463f7a6 (fix(code): small-context models get an input budget)
Reviewer model: claude-sonnet-5-5 (author: Opus 5.5). Read-only review.
Verdict: PASS
Findings: 2 (both non-blocking, LOW)

## Checks
1. Cap correctness — OK. code.rs `code_output_reserve` = min(max_tokens, window/4); window 2048 -> 512 (1536 input), 4096 -> 1024, 262144 -> 4096 unchanged, smaller reserve never raised, u32 overflow falls back to u32::MAX (no-op). Explicit `context_window` returns early (reserve kept); `--manifest` skips the call entirely (code.rs ~L572 `manifest_path.is_none()`). Window comes from `code_driver_window` = same function the driver uses, so cap and window cannot disagree.
2. #4599 guarantee — kept. Over-budget newest message still yields AgentError::ContextOverflow; test 010 asserts it and a nonzero exit via map_error_to_exit_code.
3. Falsifiers — real. 009 references the new fn (fails to compile/assert on parent); 010 asserts the uncapped 4096 reserve reproduces `ContextOverflow{available:0}` (the exact 0.70.0 defect) and the capped one passes. Receipts: tinyllama-code-fixed.log rc=0 with real output.
4. Nothing weakened — diff is +112/-0 in two source files; no test/contract/gate/workflow touched.

## Findings
- F1 LOW crates/aprender-orchestrate/src/agent/code.rs:575 — the call site `if manifest_path.is_none() { cap_output_reserve_to_model_window(..) }` in cmd_code_with is not covered by any unit test (tests call the helper directly). Deleting the call passes all new tests; only the e2e receipt would catch it. Suggest a follow-up test or keep the e2e receipt as the gate.
- F2 LOW code.rs `code_output_reserve` — windows 1..3 give reserve 0 (max_tokens 0); unrealistic for real GGUFs, no test for window 0/tiny. Consider `.max(1)` or a note.

## Release-scope record: v0.70.0..cc4463f7a6 (40 paths)
| Class | Count | Paths |
|---|---|---|
| VERSION-BUMP | 38 | 36 Cargo.toml + 2 Cargo.lock; verified by diff: after excluding `version = "0.70.x"` lines (incl. inline dependency version pins) no other +/- line remains |
| EVIDENCE | 0 | none |
| SOURCE-FIX | 2 | crates/aprender-orchestrate/src/agent/code.rs (+31), code_tests_falsification.rs (+81) |
| OTHER (finding) | 0 | none |
