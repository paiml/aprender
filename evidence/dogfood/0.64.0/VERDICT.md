# v0.64.0 dogfood verdict — written, dated, per-gate

Run: `bash scripts/dogfood.sh` on `release/0.64.0` @ e124a723d, 2026-08-23.
Receipt: `receipt-20260823T174343Z.json` (aprender 0.64.0, NO-GO, 32 gates:
16 PASS / 8 FAIL / 4 SKIP / 1 each INFO, WARN, REPORT, MANUAL).

The suite CANNOT be all-green for any version under the gates as written, so a
cut requires this determination rather than waiting for a green receipt. Each
RED below is classified with evidence.

## Fixed by the bump (RED at the audit commit, green here)
- `version-unpublished` — 0.64.0 is not on crates.io. Was RED only because the
  audited branch still said 0.63.0.
- `dogfood-gates` — the declared repo gates now discover and run green.

## Artifacts of THIS run, not defects (re-measured by hand)
- `pv-lint` — FAIL in the receipt; re-run in the gate's exact form with the
  pinned pv 0.64.0: **rc=0, `Summary: 0 errors, 10694 warnings`, `Result: PASS`**.
- `pv-bindings` — FAIL in the receipt via the "no verification line" branch,
  but the line IS emitted: `aprender: 53/55 binding functions verified in
  source`, with 2 ghosts (`compute_mse`, `forward_pass`). `compute_mse` is real
  at `crates/aprender-core/src/tree/regression_helpers.rs:27` and is
  `pub(super) fn` — the documented pv false positive. Correct classification is
  REPORT, which the gate itself specifies. The FAIL came from reading a
  `$WORKLOG` log that no longer existed (finding RCPT-1: the receipt cannot
  explain its own RED because WORKLOG is trap-deleted and the note is the first
  line matching error|fail|warning).

## Phase-misplaced — cannot pass before publish (#2643, #2658)
- `publish-dry-run` — `failed to prepare local package for uploading`: the root
  manifest pins siblings at 0.64.0, which cargo resolves from the registry
  where they do not exist yet. #2643 predicted this exact error.
- `check_multiplatform_dogfood` — needs `cargo install aprender` receipts for
  0.64.0 from four hosts. Proof it is post-publish in practice: 0.63.0 shipped
  2026-08-01, all three of its receipts are dated 2026-08-22, and only 2 of 4
  are tracked on main — the gate has never passed for any release (#2658).

## Pre-existing standing debt — identical at v0.63.0, not new
- `bashrs` — 175 SEC/DET/IDEM over 182 files. CI's bashrs gate is narrower than
  the dogfood one; this has never been enforced at this level.
- `pmat-verify` — 27 strict-mode SATD violations. `git grep` for TODO/FIXME/
  HACK/BUG returns 225 marker lines at v0.63.0, at main, and here — byte-identical.
- `pmat-comply` — CB-200 TDG grade gate. The only pmat gate in CI is in
  book.yml; this has never been enforced, so it cannot have been green before.

## Defective gate, already filed
- `cli-surface` — "advertised but unusable: apr aprender binary clip
  cross-encoder existing model no producer same the yet". Those are prose words,
  not subcommands: the enumerator scrapes `--help` text (#2641, OPEN).
- `transport-decl` — no `[package.metadata.transports]` in Cargo.toml. Measured
  absent at v0.63.0 and at main; the block has never existed (#2642, OPEN).

## Determination

None of the eight REDs is a NEW regression introduced by 0.64.0. Two are run
artifacts, two are structurally unsatisfiable before publish, three are
pre-existing debt measured for the first time by a suite that did not exist at
the v0.63.0 tag (`scripts/dogfood.sh` was added 2026-08-12), and one is a known
defective gate. A RED here means "newly measured", not "newly broken".

This is NOT a floor being lowered: nothing is exempted, no threshold moved, and
each item keeps its ticket. It is a dated statement of what the receipt means.
