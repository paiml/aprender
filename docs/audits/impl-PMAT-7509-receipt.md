# PMAT-7509 (APR-MAP-001 MAP-1): implementation receipt

Branch `PMAT-7509-map1-geo`, first reviewed code head `1af0ecfe6`; later rounds below.

## Measured
- `cargo test -p aprender-viz --features geo --lib`: 1132 passed, 0 failed (orchestrator run).
- `cargo test -p aprender-contracts --lib`: 2434 passed, 0 failed, 5 ignored.
- `cargo fmt --all -- --check`: clean.

## Not measured locally (left to PR CI)
- clippy, `cargo deny check advisories`, line coverage.

## Review
- Round 1, two lanes of gemini-4-neon-high at `2bd802557`: both FAIL. Fixed in `a206b835b`.
- Round 2, same model at `a206b835b`: both PASS, one minor fixed in `1af0ecfe6`.
- Delta verdict by claude-opus-5-5 at `1af0ecfe6`: PASS.
- Each lane rejected its planted false claim.

## Deviations
- Phases ran on claude-haiku-5-5 workers, not the `agy-goal` route.
- Dispatches were declared with `goal.sh worker` for phase 5 only.
- Phases 1 and 3 have no recorded RED run.

## Open
- Contract bindings wait for the contract fix-forward (#7527) to merge.
- Under `coord_geo` the SVG holds the raster image and the vector paths, so shapes are painted twice.
- `visvalingam` is quadratic.
- AlbersUsa inset placements are hand-chosen.
- The polar-band round-trip test is a fixed grid and checks latitude only.

## Later rounds (2026-11-28)
- Independent review at `80825060f`: Mercator inverse slack and `wrap180` rounding reproduced and fixed (`d8dd63aff`, `6e13791b0`).
- Reviews at `6e13791b0` (FAIL, changes requested): polygon refusal off a map coord, libm math, no new public `SvgElement` variant, all-empty rings. Fixed in `4507e69f6`.
- `4507e69f6`: `--features geo --lib` 1138 passed; without the feature 1010 passed. agy refused (429 quota), so one claude-opus-5-5 lane: PASS, `degraded: same-family`.
- Clippy is not measured: the host wrapper refuses `cargo clippy`.
- The issue's acceptance boxes are not ticked in the roadmap entry; that is left to the merge-time completion receipt.
