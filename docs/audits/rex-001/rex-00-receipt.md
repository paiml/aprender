# REX-00 receipt — pre-registration (PMAT-4355, epic #4354)

- Tree: worktree `/mnt/nvme-raid0/tmp/rex`, branch `rex/001`, base `origin/main` @ `5fa13fe8c` (R-12).
- Spec: operator file sha256 `3e0b5594af99…`; in-tree copy differs only in lines 1 and 3
  (PROMETHEUS rename, cop relay 2026-09-25), both outside §2–§5.
- **prereg_sha = `ef51087dc79bab0ad160e8a14f5b13e2ea43986b30c05dafc63c84c2dc21cdc0`**
  (`cargo run -p aprender-review-experiment --example rex -- prereg-check`).
- Components: spec §2–§5 `dee8b66c…`, `stats.rs` `a892d50a…`, analysis plan `8ae9213b…`,
  prompt v1 `dd9d67c2…` (full values in `prereg.lock`).
- Prompt v1 = the G1 pilot prompt header byte-for-byte (compared against the pilot's
  `prompt.txt`, cop scratchpad `qdemo/`).
- Test-item manifest: placeholder `docs/audits/review-corpus/test-manifest-v1.txt` (sealed by REX-02).
- Ordering (S-1): committed before any REX measurement; no measurement has been taken.

## Falsifier, planted on disk (RED → GREEN)

Planted `any count > 0 rejects H1` → `any count > 1 rejects H1` in §3 of the spec file:

```
rex-prereg-v1 DRIFT spec_s2_s5: lock dee8b66c… != tree e4c1abd9…
rex-prereg-v1 DRIFT prereg_sha: lock ef51087d… != tree 42db9f76…
planted_rc=1
falsify_rex_prereg_001: test result: FAILED. 0 passed; 1 failed
```

Restored (cmp-identical) → `rex-prereg-v1 OK prereg_sha=ef51087d…`; 12/12 lib tests pass.

## Recalibration note (R-7)

§2.2 quotes a half-width of "≈ 0.117" for 70 defect items at p = 0.5. That is the Wald
half-width; the pre-registered Wilson interval gives **0.114** (`stats::wilson_half_width(35, 70)`).
The decision threshold (0.12) is unchanged.
