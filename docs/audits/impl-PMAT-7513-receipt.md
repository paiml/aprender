# PMAT-7513 (MAP-5, epic #7507): implementation receipt, slice 1 (partial)

Scope: `crates/aprender-present-core/src/map/`, the headless runtime core. Refs #7513; it does not close it.

Worker: claude-haiku-5-5. Orchestrator: claude-sonnet-5-5, which reran the acceptance command and wrote the review fixes.

Acceptance: `cargo test -p aprender-present-core --lib map::` : 54 passed on this host.

Reviews (gemini-4-neon-high, one planted false claim each, each caught):

| Round | Head | Angle | Verdict | Receipt |
|---|---|---|---|---|
| 1 | c30bea396 | contract conformance | PASS | quorum-aprender-map5-r1-agy-neon-lane1.md |
| 1 | c30bea396 | adversarial | FAIL | quorum-aprender-map5-r1-agy-neon-lane2.md |
| 2 | 2dd57406b | fixes for round 1 | PASS | quorum-aprender-map5-r2-agy-neon-lane1.md |

Defects found and fixed: a pre-painted raster was lost at the next pre-paint; FALSIFY-MAP-LC-001 could not fail
(its planted mutation now turns it red, run on this host); finite inputs could overflow (the numeric domain is now
bounded); a rejected animation step advanced the animation; a pan of exactly the margin missed the cache.

Not measured: clippy, coverage, wasm32.

Open in MAP-5: sprite atlas and golden, browser binding and one-second frame count, 10,530-cell timings.
FALSIFY-MAP-LC-006 does not assert the drawn value, because the core holds no cell values yet.

Spec gap: `aprender-viz-map-camera-v1` says the MAP-5 row states the level-of-detail boundaries and the epsilon.
The row does not. The code uses boundaries 10, 22 and 44 px (Far, Mid, Near, Close; a boundary value takes the
upper tier). The spec row needs these values.
