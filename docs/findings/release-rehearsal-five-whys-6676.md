# Release Rehearsal Nightly red 4 of 4: five-whys (aprender#6676)

Evidence: run 42478090078 (2026-11-08, schedule) and the three runs before it (42369644585, 42222005269, 42034425992), all `failure`.
Read with `gh run view <id> --json jobs` and `--log-failed`. The `pick` job is green in the latest run; the red job is `rehearse`.

1. Why is the night red? `rehearse` exits 1: `RED stage bump exit 1: FAIL cargo set-version failed (is cargo-edit installed?)`; every later stage is `unreached`.
2. Why did `bump` fail? `scripts/bump-version.sh` shelled out to `cargo set-version`, a cargo-edit subcommand, not part of cargo.
3. Why was it missing? The rehearsal runner (intel-clean-room-11) has no cargo-edit, and nothing in the repo declares it as a dependency of the bump.
4. Why did no gate catch it? The bump was only exercised by the rehearsal itself, on that runner; unit gates ran where cargo-edit happened to be installed.
5. Why did it stay red 4 nights? The failure was read as a runner-provisioning issue, so no one owned a code fix.

Fix (code, merged): #6807 / PMAT-6806 makes `bump-version.sh` edit versions without cargo-edit and adds row 22 (no executable `cargo set-version` remains).
Next-night check: the stage `bump` must pass; a red there again reopens this ticket.

Second cause, visible after `bump` is fixed: `RED D1 NOT CLEARED ... preflight R4 requires origin/release/<V>, which no script creates` and
`RED D2 OPEN in t2: check_model_ladder exit=1 (no CRUX receipt dir)`. These are rehearsal-design gaps (the rehearsal does not create the release branch or hand down
CRUX receipts), tracked here, not fixed by this change.
