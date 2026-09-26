# PRM-09 — ladder extension: the three planted reports (rex-promotion-ladder-v2 1.1.0)

**Outcome: acceptance met.** The PRM-001 §9 row reads "H4 only → tripwire; H4+H5+H7 → tie-breaker; lambda primary → refused". All three run through the real binary, `rex ladder`, and not only through unit tests.

The reports below are `prm-001-report-v1` documents (spec PRM-001, version 3). They carry the **live** prereg lock, `docs/audits/prm-001/prereg.lock` `prereg_sha` `f0b4e4ac…`, so they exercise the gate and not the stale-lock refusal. Every run is recorded verbatim in `runs.txt`.

| Planted report | sha256 (prefix) | `--current` | mode | why |
|---|---|---|---|---|
| `planted-h4-only.json`: H4 holds (Holm p 0.01), within budget, gx10-cpu primary | `3922acbb9040d0f3` | shadow | **tripwire** | H5, H7, H1 and H2 are not in the report |
| `planted-h4-h5-h7.json`: adds H5 (p 0.04), H7, H1, H2; availability 0.97 | `b4e225242cd0f0eb` | tripwire | **tie-breaker** | vote needs ≥ 20 decided tie-breaks, and it has 0 |
| `planted-lambda-primary.json`: the same report with primary `lambda-cuda` | `e9e875dc457aa891` | tripwire | **tripwire (refused)** | `S-4: primary cell lambda-cuda is lambda or unnamed; lambda is shadow-only (R-9)` |
| the same lambda report | ″ | tie-breaker | **tripwire** | S-4 caps at once. It is the one exception to the one-rung step |
| the H4+H5+H7 report | ″ | shadow | tripwire | the positive control for the step rule: one rung per report, Shadow → Tripwire, never straight to TieBreaker |

## Defect fixed on the way

The S-4 reason printed the Rust Debug rendering of the cell, `primary cell Some("lambda-cuda")`. That string would have reached the quorum gate and the arbiter's receipts. It now names the cell as written, or `(unnamed)`. The fix is FALSIFY-RXG-006 in contract 1.1.0: planted RED in `cf7bdbfbb`, then GREEN.

## Named follow-ups (not in this change)

- **pi#428**: the paiml-implement quorum gate reads `rex ladder … | .mode`.
- **infra#1095**: the arbiter reads the same `.mode`.

Both consume the `.mode` field and must never re-derive it (contract description).

## Gates

- `cargo test -p aprender-review-experiment --lib`: 167 passed.
- `cargo clippy -p aprender-review-experiment --all-targets -- -D warnings`: clean.
- `cargo fmt`: clean.
- `pv validate contracts/rex-promotion-ladder-v2.yaml`: valid.
