# impl receipt — PMAT-3552 (aprender#3552)

**Ticket:** Facet/Coord have no runnable example; EV-15/EV-16 have nothing to build against.
**Acceptance:** one program under `examples/` going study data → `panels` → `Coord::apply` → SVG,
compiling against the **published crate only**, wired so it cannot rot.

## What changed

| File | Change |
|------|--------|
| `examples/viz-facet-coord/src/main.rs` | New program. 12-row study `DataFrame` (lineage/dose/response) → `panels(&Facet::wrap("lineage", 2))` → `apply_limits` + `apply(&Coord::cartesian().xlim(0,8).flip())` per point → `SvgEncoder` (panel rects, level labels, circles, polyline) → `render()`/`write_to_file`. It exits 1 if a law it relies on fails: 3 panels covering every row, the flip moving xlim to the vertical axis, every point inside its panel, and one point per row. |
| `examples/viz-facet-coord/Cargo.toml.in` | `aprender-viz = "=0.69.1"` (crates.io), its own `[workspace]`. The name is `.in` so it is never a workspace member or a nested package. This follows the precedent of `tests/fixtures/dogfood_examples` (FALSIFY-MONO-012 flat layout). |
| `scripts/check_viz_published_example.sh` | `--self-test` (network-free, and run FIRST on every check) is an 8-row case table for the two classifiers: `source_ok` (crates.io 0.69.1 accepted; a workspace path, a git source or another version are RED) and `svg_ok` (12 points + ≥3 rects + `</svg>`; missing points, one panel or an unclosed document are RED). Then the known-bad build: a `Facet::none` mutant must exit ≠0. Then the real build: `cargo pkgid aprender-viz` must equal the crates.io registry id (no python; round 3 lane 2), and the SVG must pass `svg_ok`. Each build deletes the binary first. A failed MUTANT build exits 2 (environment: nothing was judged). A failed REAL build exits 1, because that is the defect this gate exists for. |
| `Makefile` | `make check-viz-example` target. |
| `Cargo.toml` | Declared in `[package.metadata.dogfood].gates`, so `scripts/dogfood.sh` runs it on every release (network available there). That is the wiring `check_guards_are_wired.sh` accepts, and it closes quorum round 1 lane 2's finding (NEW: check_viz_published_example.sh). |

## Evidence: `make -s check-viz-example` at 062ef8ccc (lambda, private CARGO_TARGET_DIR, heavy-slot queue; `cargo-memcap` queue lines omitted)

```
ok   self-test: crates.io 0.69.1 accepted
ok   self-test: workspace path source RED
ok   self-test: git source RED
ok   self-test: other version RED
ok   self-test: well-formed SVG accepted
ok   self-test: missing points RED
ok   self-test: one panel RED
ok   self-test: unclosed document RED
ok   mutant (Facet::none) killed: viz-facet-coord: FAIL: facet law: 1 panels covering 12/12 rows
ok   aprender-viz resolved from crates.io: registry+https://github.com/rust-lang/crates.io-index#aprender-viz@0.69.1
ok   SVG shape (viz-facet-coord: panels=3 points=12 bytes=1815)
rc=0
```

Second mutant, run by hand (`.flip()` removed, build=0, fresh binary):
`FAIL: coord law: flip must move xlim to the vertical axis, got (1.5, 7.5)`, rc=1.

A DNS failure during an earlier mutant build left a stale binary. It "killed" a mutant it was never built from, so that run is discarded. That is why the script deletes the binary before each build, and a failed MUTANT rebuild exits 2 (not judged).

Self-test mutants (scratch copy): `source_ok` forced true → 3 rows RED; the `svg_ok` count check forced true → 2 rows RED.

## Not done / handed off

- **Per-PR CI step (optional, beyond the release gate)**: `.github/workflows` edits need a cop quorum;
  the step would be `bash scripts/check_viz_published_example.sh` (needs network to crates.io).
- Not in `make tier3`: it needs network and a full aprender-viz+trueno build from crates.io.
- EV-15 (byte-identical SVG across archs, `text-path`+`raster`) and EV-16 are downstream; this program is what they build on.
- The pin `=0.69.1` must be bumped when a newer aprender-viz is published. The check stays on the published crate by design.
