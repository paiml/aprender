# PMAT-7509 (APR-MAP-001 MAP-1): implementation receipt

Branch `PMAT-7509-map1-geo`, code head `1af0ecfe6`.

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
- Contract bindings wait for MAP-0 to merge.
- Under `coord_geo` the SVG holds the raster image and the vector paths, so shapes are painted twice.
- `visvalingam` is quadratic.
- AlbersUsa inset placements are hand-chosen.
- The polar-band round-trip test is a fixed grid and checks latitude only.
