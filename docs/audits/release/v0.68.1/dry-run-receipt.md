# v0.68.1 — cascade dry-run receipt (APR-RELEASE-001 §4 T-4)

Committed **before** the real cascade starts. Nothing in this directory was uploaded.

| Fact | Value |
|---|---|
| tag / release commit | `v0.68.1` = `1661c7138705d6b0a1f40c42bfe2f8a140c91c62` (detached checkout, `HEAD == tag`, tree clean before and after) |
| clean-room B2-cpu on the tag | paiml/infra run **35255889257**, job `clean-room (aprender)` = success — A0 A1 A2 A3 A4, B0 B1 B2 B3 B4 B5 all PASSED; B2 = 74 test binaries, 78,794 passed, 0 failed, `--no-fail-fast` |
| clean-room B2-gpu on the tag | paiml/aprender run **35268563585** (`b2-gpu.yml`, yoga RTX 4060 sm_89) = success — tested-sha asserted, 2604 passed / 12 ignored, floor 2500 |
| repo cascade gate | `scripts/cascade-publish.sh --check` rc=0 → `CLEAN-ROOM PROCEED` on run 35255889257 (see `cascade-check.log`); 71 crates behind at 0.67.0, 3 facades already live at 0.4.0 |
| packaging dry-run | `cargo publish --workspace --dry-run --no-verify --locked` rc=**0**, 71/71 packaged, 2026-09-17T20:28:59Z → 2026-09-17T20:30:38Z, tree clean after (`publish-dryrun-noverify.summary.log`) |
| publish order | `publish-order.txt` — derived from `cargo metadata` at the tag over normal + build + versioned-dev edges, acyclic, re-proved by `publish_strict.sh` before the first upload. NOT `TIERS`: 47 non-dev order violations at this tag (#3462) |
| T-2 | full pre-publish dogfood GO on `1661c7138` (receipt-20260917T143837Z.json); parent-sha preflight GO on `ee684e94c` |
| assets / installer | 16/16 release assets (binary-release run 35238055998); `install.sh --version v0.68.1` rc=0 on intel (x86_64) and gx10 (aarch64), `apr 0.68.1` |
| publish preflight | R1–R5 ok at the tag. **R6 refused** on four versioned dev-deps of `aprender-compute` (#3307) that lie on no cycle; fixed gate-side in #3469 (R6 judges the cycle). With the fix, run against the unmoved tag tree: PASS. The cascade does not start until #3469 is on `main` |
| authorization | standing operator authorization 2026-09-17 (§3.7); `attended_min: 0 (operator-authorized unattended, 2026-09-17)` |

## Gate defects found by this train's first-ever complete chain (all fixed gate-side; the tag never moved)

1. infra#650 — the clone lacked `refs/tags/<ref>`; the assert could not resolve the tag
2. infra#652 — A1 crates.io deadlock on a bumped-unpublished tag → `[patch.crates-io]` overlay of the tag's own members
3. infra#654 — the overlay was a CLI flag; a test's child `cargo metadata` did not inherit it; B2 was fail-fast
4. infra#656 — `apr` not on PATH (A3 skipped); one test holds 32.6 GB (32g memcg) → 48g + 8 threads, measured; 96 CUDA tests in a CPU container → B2 split
5. infra#658 / aprender#3467 — the org GPU runner groups admit paiml/aprender only → B2-gpu runs from this repo
6. aprender#3469 — preflight R6 refused a shape, not a cycle
