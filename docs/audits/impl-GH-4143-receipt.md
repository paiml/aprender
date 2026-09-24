# GH-4143 (#4117 follow-up): a relative receipt path needs its verdict's location

Stacked on `fix/4117-release-scope-callers` @ 0a6c7ef62 (3/3 PASS at 83645a21e). Source: aprender-6c's review of 215da7f79. The verdict was PASS, with one LOW note.

**Defect.** `nightly_admission.resolve()` with no `_path` computed `base = dirname(abspath(""))`, i.e. the cwd. A relative path in a verdict dict that says nowhere where it lives was therefore resolved against whatever directory the caller ran in: the host-dependence 215da7f79 removed. No current caller reaches it: `load_verdicts` always sets `_path`.

**Change.**
- A relative path with no `_path` returns `(None, "a relative receipt path … with no verdict location")`.
- `coherent()` now reports every resolve refusal except "no receipt recorded" by name, instead of the generic "unreadable".

**Measured.**
- `nightly_admission_cases.py --mutants`: 20 rows. New row `relative-path-without-verdict-location-refused`; its cwd holds a valid GREEN receipt at that relative path, so only this rule can refuse it.
- Mutants: 15/15 killed, including the new `no-path-from-cwd`, the old cwd fallback restored.
- `check_certify_nightly.sh`: PASS. `check_model_ladder.sh --self-test`: 155 cases, 0 bad.

**Quorum round 1** (judged against GH-4117 by mistake): lane 2 FAIL, correct. The diff and receipt did not match GH-4117's intent, so this change now has its own ticket, #4143, and this receipt.
