# PMAT-7511 (APR-MAP-001 MAP-3): implementation receipt

Branch `PMAT-7511-map3-spatial`, stacked on MAP-1 (`80825060f`), code head `ce2906ce4`.

## Measured
- `cargo test -p aprender-viz --features geo --lib`: 1186 passed, 0 failed (orchestrator run).
- `cargo fmt --all -- --check`: clean.
- The five planted mutations of the hex contract each turn their own `falsify_map_hex_00N` test red (orchestrator run at `ce2906ce4`).

## Not measured locally (left to PR CI)
- clippy, `cargo deny check advisories`, line coverage.

## Review
- Round 1, two lanes of gemini-4-neon-high at `bf3e1f029`: one FAIL; one timed out with a partial answer and one MAJOR. Fixed in `8350fa983` and `ce2906ce4`.
- Round 2, same model at `ce2906ce4`: both PASS, each rejected its planted false claim. The correctness lane read the non-test diff only.

## Deviations
- Phases ran on claude-haiku-5-5 workers, not the `agy-goal` route.
- The aspect-ratio test written in phase 3 could not fail; review caught it and phase 4 replaced it.
- The cut-cell fixture as first written equalled its own bounding box, so its mutation could not turn red; the orchestrator made it concave in `bf3e1f029`.

## Open
- Contract bindings wait for MAP-0 to merge.
