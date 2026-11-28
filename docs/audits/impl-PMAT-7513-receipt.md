# PMAT-7513 (MAP-5, epic #7507): implementation receipt, slice 1 (partial)

Scope: `crates/aprender-present-core/src/map/`, the headless runtime core. Refs #7513; it does not close it.

Worker: claude-haiku-5-5. Orchestrator: claude-sonnet-5-5, which reran the acceptance command and wrote the review fixes.

Acceptance: `cargo test -p aprender-present-core --lib map::` : 74 passed on this host (after the review-fix commit).

Reviews (gemini-4-neon-high, one planted false claim each, each caught):

| Round | Head | Angle | Verdict | Receipt |
|---|---|---|---|---|
| 1 | c30bea396 | contract conformance | PASS | quorum-aprender-map5-r1-agy-neon-lane1.md |
| 1 | c30bea396 | adversarial | FAIL | quorum-aprender-map5-r1-agy-neon-lane2.md |
| 2 | 2dd57406b | fixes for round 1 | PASS | quorum-aprender-map5-r2-agy-neon-lane1.md |

Defects found and fixed: a pre-painted raster was lost at the next pre-paint; FALSIFY-MAP-LC-001 could not fail
(its planted mutation now turns it red, run on this host); finite inputs could overflow (the numeric domain is now
bounded); a rejected animation step advanced the animation; a pan of exactly the margin missed the cache.

Review-fix pass (findings from a later review, each checked by a mutation that turns a test red and then reverted): the flyTo test now asserts quarter points and monotone progress (a reversed t, or a constant midpoint, is red); the take_frame request recorded by `handle` is tested; LC-002 covers pans past the margin in all four directions (a dropped `abs` on either axis is red); the sprite disc test checks orientation and area (an inverted or square disc is red); the atlas test varies the key (a key-blind cache is red). The layer overlay invalidate near `layers.rs:341` was already caught by the hover and LC-005 tests (red when removed). The casts and float comparisons in `map/` are allowed workspace-wide (`Cargo.toml` `[workspace.lints.clippy]`: cast_possible_truncation, cast_sign_loss, float_cmp), so no change was made for them. Added doc comments to the public fields and variants in `map/`.

Not measured: clippy (refused on this host, so the lint result is not known), coverage, wasm32.

Sprite atlas: implemented (`sprites.rs`, keyed by key, seed and size bucket) with a committed FNV golden for one sprite. Open in MAP-5: a golden against rendered output, browser binding, one-second frame count, 10,530-cell timings.
FALSIFY-MAP-LC-006 does not assert the drawn value, because the core holds no cell values yet.

Spec gap: `aprender-viz-map-camera-v1` says the MAP-5 row states the level-of-detail boundaries and the epsilon.
The row does not. The code uses boundaries 10, 22 and 44 px (Far, Mid, Near, Close; a boundary value takes the
upper tier). The spec row needs these values.
