# Implementation receipt: GH-4086, the dogfood's declared ladder gate reads a recorded emergency scope

Branch `fix/4086-dogfood-ladder-scope-seam`, based on `origin/chore/0.69.1-merge-back` (#4046, 9d85c96fd). The base is the merge-back because `scripts/lib/crux_smoke_scope.py` and `ladder.emergency_scopes` exist only there, not on main (checked at origin/main 49fe19c28).

## The defect (issue #4086)

At 0.69.1 the release gate ran `check_model_ladder.sh --scope crux-smoke`, which was satisfied on both hosts. The pre-publish dogfood ran the same script as a declared gate. `dogfood.sh` runs each declared gate as `bash "$dg_path"` with no arguments, so it judged the full ladder and went RED. The row shipped known-red. Two gates judged one release and returned opposite verdicts.

## What changed

- `scripts/lib/crux_smoke_scope.py` adds `recorded_scope(L, version)`. It returns the scope name when exactly one `emergency_scopes` entry's `release` equals the version, compared as whole strings. It returns `(None, None)` when no entry matches, which means the full ladder applies. It returns a reason, which makes the gate RED, for two entries or a nameless entry.
- `scripts/check_model_ladder.sh`:
  - With no `--scope`, it applies the recorded scope and prints `SCOPED: <name> -- …` before the scope's own verdict line.
  - `--scope none` forces the full ladder.
  - An unusable record is `RED … neither the scope nor the ladder was judged`.
  - `MODEL_LADDER_CRUX_DIR` is the receipts seam for a caller that cannot pass `--crux`; `--crux` wins over it.
- `scripts/dogfood.sh` `classify_declared` carries a gate's `SCOPED:` line into its row note, on both PASS and FAIL. Only the scope name is carried, because `mark()` keeps 200 chars of a note and a red row's reason must survive.
- `contracts/model-capability-ladder-v1.yaml` adds FALSIFY-MCL-017.

## Measured

- `bash scripts/check_model_ladder.sh --self-test`: rc 0, "147 case(s), 0 bad".
  - The new select table (6 rows) passes, and 4 select mutants (any-release, prefix-match, first-wins, nameless-ok) are each killed.
  - The new end-to-end table (7 rows) runs the script itself and passes. 4 dispatch mutants (no-auto-scope, silent-scope, none-ignored, crux-env-unread) are each killed.
  - The must-RED row: with **no** recorded scope and CRUX smoke receipts that would satisfy a scope, the gate returns rc 1. It prints the full ladder's `no receipt at …/receipts/lambda.json` and no `SCOPED:` line.
- `bash scripts/check_dogfood_no_defer.sh --self-test`: rc 0.
  - 4 new `declnote` rows pass.
  - 3 new mutants (scope-dropped-pass, scope-dropped-fail, scope-anywhere) are each killed.
  - The scope-dropped-fail mutant first SURVIVED. The one-line fixture log put the `SCOPED:` line inside the 3-line tail that the FAIL note already quotes. The fixture is now a realistic multi-line red log, and the mutant is killed.
- Real data, the dogfood's own call (no arguments), `MODEL_LADDER_CRUX_DIR=/mnt/nvme-raid0/apr-release/d8a6df53a/crux-smoke` (the published 0.69.1 receipts):
  - With `--cut-commit d8a6df53a…3f` it returns rc 0, prints `SCOPED: crux-smoke -- …`, and gives the scope's "satisfied" line.
  - At this branch's HEAD it returns rc 1, "not the release binary". Those receipts are from d8a6df53a and bind only to it.
- `pv validate contracts/model-capability-ladder-v1.yaml`: valid.
- `check_dogfood_shim.sh`, `check_no_claim_literals.sh` and `check_dogfood_no_defer.sh`: rc 0.
- `bashrs lint --no-ignore --level error` on the 3 changed shell files: 0 SEC/DET/IDEM.
- Pre-push: `cargo fmt --all -- --check` rc 0; `cargo test -p aprender-contracts --lib` 1701 passed; `cargo deny check advisories` ok.

## Not done / limits

- The 148 existing case rows call the `judge` function directly, not the top-level dispatch. Auto-selection therefore cannot change them, and the new behaviour is covered only by the new end-to-end rows.
- The workspace version on this base is 0.69.1, so a no-argument run on this tree auto-scopes. That is the intent, and it lapses when the version moves to 0.70.x.
