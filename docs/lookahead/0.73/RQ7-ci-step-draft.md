# RQ-7 draft: the P3 aarch64 parity step in the determinism job

Status: draft on the drafts branch, not a PR. RQ-7 is ruled (a), 3-0 by quorum on
2026-10-05, and it is operator-approved (C299 Q4, "all recommended"; cop ruling
2026-10-05 15:06Z). The edit needs no new check-in if it does only what RQ-7 says,
keeps every gate, carries red/green proof and passes a quorum on its diff. It lands
with P3 or after it, because its filter names P3's tests and an empty filter is RED.

## Where it goes, re-read at origin/main 11f844a772
R3 §16 cites .github/workflows/ci.yml:279 at 316dee2cd4. Since the fat-job split (#4433) the leg is:
- .github/workflows/ci.yml:388, job `determinism`, `runs-on: [self-hosted, Linux, clean-room, ARM64]`.
- .github/workflows/ci.yml:415, its first step's `--sections 'determinism[ARM64],determinism-compare'`.
- .github/workflows/ci.yml:774, `gate` needs `determinism`, so a failing section fails `gate`.
- Job bodies live in ci/sections.yml; fat_driver.py runs each as a section with its
  own clone, env and step outcomes.
- ci/sections.yml carries the operator ruling "Same test set as today (Σ executed
  unchanged)". This edit adds test ids and removes none; scripts/ci/sigma_executed.py
  compares sets in both directions, so the new section declares its own universe and
  is checked against it.

## The edit (two hunks)
ci/sections.yml, a new section next to `determinism`:
```yaml
  # RQ-7 (a), #3999 P3: the NEON parity tests are cfg(target_arch = "aarch64"), so
  # no x86 lane runs them. This section is listed only in the determinism fat job's
  # ARM64 leg. Its sigma step lists the ids itself, so an empty run is RED.
  neon-parity:
    name: neon parity (ARM64)
    runs-on: [self-hosted, Linux, clean-room, ARM64]
    timeout-minutes: 45
    steps:
      - uses: actions/checkout@v7
        with:
          persist-credentials: false
      - name: A target dir of this job's own
        run: echo "CARGO_TARGET_DIR=${RUNNER_TEMP}/neon-parity-target" >> "$GITHUB_ENV"
      - name: NEON parity tests
        run: cargo test -p aprender-serve --lib falsify_neon_q4k_
      - name: Sigma executed
        run: |
          set -euo pipefail
          sig="${RUNNER_TEMP}/neon-sigma"; mkdir -p "$sig"
          cargo test -p aprender-serve --lib falsify_neon_q4k_ -- --list > "$sig/list.txt"
          cargo test -p aprender-serve --lib falsify_neon_q4k_ -- --list --ignored > "$sig/ignored.txt"
          python3 scripts/ci/sigma_executed.py cargo --kind "neon parity" \
            --log "${FAT_SECTION_LOG:?the fat driver sets FAT_SECTION_LOG}" \
            --step "NEON parity tests" --list "$sig/list.txt" --ignored "$sig/ignored.txt"
```
.github/workflows/ci.yml:415 @11f844a772, one word added to the list:
```
--sections 'determinism[ARM64],determinism-compare,neon-parity'
```
Nothing else changes: no `needs:`, no `if:`, no runner label, no timeout of another job.

## Red/green proof the PR must carry
| # | Run | Expect |
|---|---|---|
| G1 | the PR as is, with P3's tests present | determinism green; the section log lists every `falsify_neon_q4k_*` id compiled on aarch64, count > 0 |
| R1 | plant a wrong NEON arm (R3-test-skeletons.md, the mutation at :220) | `falsify_neon_q4k_001_q4k_f32` RED, determinism RED, gate RED |
| R2 | change the filter to a name nothing matches | sigma_executed RED on an empty universe |
| R3 | drop `neon-parity` from the `--sections` list only | the section never runs; whatever lists unrun sections must go RED. Unchecked [U]: if nothing does, the PR adds that check |
| G2 | revert R1-R3 | green again, same id set as G1 |

## Open, for the PR author
- [U] The fat job's 120-minute budget: the ARM64 raster takes about a minute
  (.github/workflows/ci.yml:393 @11f844a772), and a cold aprender-serve lib build on the clean-room ARM64 runner is
  unmeasured. If it does not fit, the fix is a separate ARM64 fat job, which is not
  what RQ-7 says: that would be a RULING? line, not a quiet change.
- [U] Whether the power guard and NEON-000/006 (x86, ticket-bodies-P1-P5.md P3
  **Hosts**) match the filter on aarch64. They are cfg-gated per R3-test-skeletons.md;
  the G1 list settles it.
- Under C277 L3 opens no PR and runs no CI, so G1-G2 are run by P3's owner.
