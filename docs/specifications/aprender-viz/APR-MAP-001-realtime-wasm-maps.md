# APR-MAP-001: Realtime WASM maps and dashboards

Status: Draft, revision 7 (2026-11-27; revision 6 plus two acceptance lines added after review round 6, in MAP-3 and MAP-6).
Tracking: epic #7507; rows MAP-0 to MAP-9 are issues #7508 to #7517.
Kind: CRUX review plus epic plan. No product code is specified as done here.

## 1. Why

A public reference, <https://karpathy.ai/usaciv/> ("the United States as a strategy game"), shows what a
single-page interactive map with linked dashboards looks like when it is done well: a hex-cell map of the
United States, 19 switchable choropleth "lenses", ten dashboard docks linked both ways with the map, and a
URL hash that restores the exact view. This document measures that reference, measures what this workspace
has today, and plans the rows that close the gap **as Rust compiled to WASM, with live data**.

We reproduce the *capabilities*. We do not copy its code, art or data files.

## 2. The reference, measured from fetched bytes (2026-11-27)

| Aspect | Measured |
|---|---|
| Stack | Plain ES modules and Canvas 2D. No framework, no build step, **no WASM, no WebGL, no Web Workers** |
| Live data | **None.** No WebSocket, no EventSource, no polling. All data is static JSON |
| Grid | Pointy-top hex grid, 117 columns x 90 rows = 10,530 cells, 4,807 of them land; world is 5002 x 3302 km |
| Per-cell data | Column arrays: owner, county, elevation, water, relief, cover, land-cover percentages |
| Entities | 3,109 counties, 524 cities, 60 rivers, 618 resource sites of 21 types, 49 state rows |
| Lenses | 19 choropleths, one value per cell, linear or log scale, legend is a histogram that can be brushed |
| Dashboards | 10 "advisor" docks plus a picker that switches between them: stat tiles, bar lists, histograms, sortable tables, a brushable scatter plot; selection is linked both ways with the map |
| Payload | About 2.86 MB eager JSON (574 KB gzip), about 6.7 MB in total; three entity tables load on demand |
| Render | One static-layer raster the size of the viewport plus a 160 px margin. Pan blits it. Zoom blits it scaled, then re-renders once when the camera settles |
| Level of detail | Four tiers by hex radius in pixels (under 10, under 22, under 44, 44 and over) |
| Sprites | Painted once per (key, quarter-octave zoom bucket) and stamped; seeded RNG so output is stable |
| Frame loop | Draw on demand: one queued requestAnimationFrame; camera steps only while moving; idle time pre-paints the next zoom step |
| Camera | World units in km; eased zoom around an anchor point; flyTo on the van Wijk and Nuij path |
| URL state | `#lens=..&city=..&state=..&at=x,y,zoom&advisor=..&layers=..`; `at` is the centre in km and log2 of scale over home scale; extensions register their own keys; lens and selection use pushState, camera uses a 350 ms debounced replaceState |
| Input | Drag, wheel, double-click, arrows and WASD, pinch, minimap, `/` or Ctrl-K search, lens cycling, a resource-layer toggle, a help overlay, Esc, mobile bottom sheets |
| Export | Copy link; download a PNG |
| Offline pipeline | Standard-library Python |

Not measured: frame times and per-frame draw counts. They were read from source, not profiled in a browser.

**Reading.** The reference is neither realtime nor WASM. Its quality comes from a small, well-chosen data
model (columnar hex cells), a cached raster with cheap pan and zoom, and consistent linked selection.
"Realtime WASM" is where this plan goes beyond the reference, so it must not cost the things the reference
gets right: a small eager payload (574 KB gzip there) and no work on an idle frame.

## 3. This workspace today

| Capability | State | Where | Missing |
|---|---|---|---|
| Canvas drawing in the browser | partial | `crates/aprender-present-lib/src/browser/canvas2d.rs` | Immediate mode only; no retained layer, no raster cache |
| GPU instanced drawing | partial | `crates/aprender-present-lib/src/webgpu.rs` | No polygon tessellation |
| Pointer, wheel, touch, pinch | partial | `crates/aprender-present-lib/src/browser/events.rs`, `crates/aprender-present-core/src/gesture.rs` | No camera or viewport transform |
| Tooltip, charts, table, tabs | partial | `crates/aprender-present-widgets/src/tooltip.rs`, `chart.rs`, `data_table.rs`, `tabs.rs` | No hit-testing, no brush, no histogram chart, no stat tile |
| Shared state | exists | `crates/aprender-present-core/src/state.rs` (`Store`) | n/a |
| Draw list | exists | `crates/aprender-present-core/src/draw.rs` (`DrawCommand`) | n/a |
| URL hash | partial | `crates/aprender-present-lib/src/browser/router.rs` | No state codec (reading the hash and pushing or replacing history entries exist) |
| Live stream client | partial | `crates/aprender-present-lib/src/browser/websocket.rs`, `crates/aprender-present-core/src/streaming.rs` | Client only. The server crate has SSE for token streaming, no WebSocket and no data-push endpoint |
| Columnar data in the browser | partial | `crates/aprender-db/src/wasm/streaming_parquet.rs` | No load from an in-memory buffer, no manifest hash check (loading by URL with HTTP range requests exists) |
| Colour scales, tile geom | partial | `crates/aprender-viz/src/scale.rs`, `crates/aprender-viz/src/grammar/geom.rs` | No polygon or hex geom |
| Coordinate systems | partial | `crates/aprender-viz/src/grammar/coord.rs` | Cartesian only |
| Polygon output | partial | the SVG encoder of `crates/aprender-viz` emits polygon and path elements | No polygon fill in the framebuffer; no even-odd fill of a polygon with holes |
| GeoJSON or TopoJSON reading | **absent** | n/a | all |
| Map projections | **absent** | n/a | all |
| Line simplification | **absent** | n/a | all |
| Hex grid | **absent** | n/a | all |
| 2D spatial index, polygon hit-testing | **absent** | n/a | all |
| Camera, raster cache, lens registry, minimap, search | **absent** | n/a | all |
| Bundle builder (shapes and tables in, columnar bundle out) | **absent** | n/a | all |
| CI that builds the viz, present or `aprender-db` crates for wasm32, or runs a browser test | **absent** | `scripts/check_wasm32_core_builds.sh` type-checks the core crate only | all |
| Browser end-to-end tooling | exists | `crates/aprender-test-lib`, `crates/aprender-test-cli` (probar, probador) | Not yet pointed at a map |

**The crux.** Two things are absent outright and gate the rest. The first is a **geo kernel**: geometry in,
projected polygons out (MAP-1). Nothing map-shaped, static or interactive, can be drawn without it, and the
hex grid (MAP-3) is built on it. The second is a **wasm32 build and browser test of the viz, present and
`aprender-db` crates** (MAP-2): no browser acceptance in this plan can run without it.

The rest of section 3 sorts into three groups. *Exists*: the store, the draw list, the browser test
tooling. *Partial*: canvas drawing, GPU instanced drawing, colour scales and tile geometry, coordinate systems, polygon output, input events and gestures, the router, the stream client, the columnar
reader, the widgets. *Absent, new code*: camera, raster cache, hit-testing, spatial index, lens registry,
minimap, search, stat-tile and histogram widgets, the bundle builder, and a data-push endpoint on the server.

Whether Canvas 2D is fast enough is **a hypothesis, not a finding**. The reference does it in plain
JavaScript at about ten thousand cells, but its frame times were not measured, and this plan adds a
WASM-to-canvas call boundary and live deltas. MAP-5 measures it.

## 4. Design

1. **The geo kernel lives in `aprender-viz` behind a `geo` feature**, not in a new crate. Static output
   (SVG, PNG) and the interactive runtime then share one projection and one fill implementation, and a
   downstream user of static figures gets choropleths from a version bump.
2. **The map runtime lives in the present crates**, Canvas 2D first, reusing `DrawCommand`, `Store`, the
   gesture recogniser, the router and the stream client.
3. **Three cached layers.** The *base layer* holds geometry that does not depend on data: cell or polygon
   outlines, borders, terrain, sprites. The *lens layer* holds the choropleth fill of the active lens. The
   *overlay* holds hover, selection and point or line entities. Each has its own raster cache. A camera
   settle, or a pan past the margin, re-renders each of the three once, except a layer whose raster for the new camera is already cached (as after the idle pre-paint). A lens switch or a delta to the active lens
   re-renders the lens layer only, once per switch or per delta message. A hover, a selection or a layer
   toggle re-renders the overlay only. A delta to a lens that is not active is stored and renders nothing until that lens is selected.
4. **Data bundles are parquet plus a sha256 manifest, built by a Rust tool.** The browser verifies the
   manifest before it shows a value. No Python enters the build. A bundle has an *eager* part (needed for
   first paint) and *lazy* tables fetched on first use.
5. **Realtime means columnar deltas pushed over WebSocket from the inference server crate.** Serving stays
   in that crate (the Realizar-first rule). WebSocket and not SSE, because the browser client and its
   reconnect logic already exist for WebSocket and nothing exists for SSE on the client.
6. **Thresholds follow measurement.** No frame-time, payload or `.wasm` size number in this document is a
   gate. MAP-5 (frame time), MAP-4 (eager payload) and MAP-2 (`.wasm` size) record baselines first; a
   threshold is proposed only from those records.

Out of scope: tile pyramids and slippy-map servers, 3D terrain, a general GIS, copying the reference's art,
and the reference's own offline data pipeline.

## 5. Epic rows

One epic, "Realtime WASM Maps and Dashboards". Each row is one ticket and one or more PRs.

| Row | Depends on | Scope | Acceptance (each one can fail) |
|---|---|---|---|
| MAP-0 | none | This spec, plus `pv` contracts for projection, hex grid, camera, layer cache and URL state | `pv validate` passes on each; each contract names a mutation that turns its test red |
| MAP-1 | MAP-0 | Geo kernel in `aprender-viz` (`geo` feature): GeoJSON and TopoJSON reader; Albers equal-area conic, composite Albers USA, Mercator, equirectangular; Douglas-Peucker and Visvalingam simplification; even-odd polygon fill in the framebuffer and as SVG paths; `Coord::Geo`; `Geom::polygon` | For each projection, forward then inverse returns the input within a stated epsilon **for points inside that projection's stated valid domain**, and a point outside it returns an error, not a number (property tests); each projection maps a committed table of reference points to committed reference coordinates within a stated epsilon (one table per projection); a committed GeoJSON fixture and the TopoJSON encoding of the same shapes decode to equal rings; each simplifier keeps the two end points, never adds a point, at tolerance 0 returns its input, and at a stated tolerance above 0 turns a committed fixture polyline into a committed reference polyline with fewer points (one reference per algorithm); a polygon with a hole leaves a pixel inside the hole unfilled and a pixel in the ring filled, in the framebuffer and in the SVG path; a three-shape golden SVG matches a committed file, and changing one coordinate of the fixture changes it |
| MAP-2 | MAP-1 | Browser harness: a script that builds `aprender-viz` (with `geo`), the present crates and `aprender-db` for `wasm32-unknown-unknown`; a probador smoke page that draws one projected polygon; a recorded, compressed `.wasm` size per build; and one non-blocking nightly CI job that runs the script, added by a reviewed workflow change and tracked by this row's ticket (decision D3) | The build fails on a planted `use std::os::unix::fs::PermissionsExt;` in a present crate; the smoke test fails on a planted one-colour change; the size record is a committed file that the script rewrites and a stale file fails the script |
| MAP-3 | MAP-1 | Spatial queries. Hex grid: axial coordinates, constant-time pixel-to-cell picking, polygon-to-cell rasterisation by **area-weighted sum**, per-cell column arrays, `Geom::hex`. Polygons and points: a 2D spatial index with point-in-polygon picking and nearest-point lookup | `pick(center(c)) == c` for every cell; for a fixture polygon of area `A` the overlap areas `a_c` of its cells sum to `A` within a stated epsilon, and a value `V` spread as `V * a_c / A` sums back to `V` within a stated epsilon; for a committed fixture polygon that cuts cells partly, each cell's `a_c` equals a committed reference area within a stated epsilon; a point on a shared hex edge resolves to one of the two cells that share that edge, and to the same one on every call; for random points the index's polygon pick equals a brute-force point-in-polygon scan and its nearest point equals a brute-force nearest scan (property tests); a `Geom::hex` golden SVG matches a committed file |
| MAP-4 | MAP-2, MAP-3 | Bundles. Builder as a registered CLI subcommand (`contracts/apr-cli-commands-v1.yaml`): shapes plus CSV in; parquet bundle (eager part and lazy tables, including hex-cell columns from MAP-3) plus sha256 manifest out. Browser side: load a bundle from an in-memory buffer as well as by URL, and verify every part against the manifest before use | Two builder runs give identical bytes; in probador the bundle loads by URL and from a buffer; a manifest with one flipped hash byte, and a data part with one flipped byte, are each refused in the browser with the part named; a lazy table is not requested before first use, and on first use is requested exactly once, hash-checked against the manifest, and then returns its rows (request log); the eager part's compressed size is recorded |
| MAP-5 | MAP-2, MAP-3 | Runtime: `Camera2D` (anchored eased zoom, flyTo), viewport culling, the three cached layers of design item 3 with a margin, pan by blit, zoom by scaled blit then one re-render, level-of-detail tiers, zoom-bucketed sprite atlas from a seeded generator, draw-on-demand loop, idle-time pre-paint of the next zoom step | With no input the frame callback count over one second is 0; a pan inside the margin performs 0 renders of any of the three layers, a pan past it performs exactly 1 of each; a zoom performs 0 renders while moving and exactly 1 of each layer on settle when no pre-painted raster exists; zooming about an anchor leaves the world point under the anchor fixed (property test); flyTo starts at the current camera and ends at the target centre and zoom within a stated epsilon; after the idle pre-paint has run, settling one zoom step performs 0 base renders; the culled draw list equals a brute-force visibility filter; each level-of-detail boundary selects the stated tier on both sides; a sprite for a stated key and seed matches a committed golden, and the same key in two zoom buckets gives two different sizes; on a 10,530-cell map the pan-blit, base-render and lens-render times are measured and recorded with the machine and browser named |
| MAP-6 | MAP-4, MAP-5 | Interaction: picking, tooltip, selection, minimap, search, keyboard, touch, layer toggles for point and line overlays, a help overlay, copy-link and PNG export, state codec with extension-registered keys, history policy (push for selection, debounced replace for camera) | `decode(encode(s)) == s` for every state and `encode(decode(encode(s))) == encode(s)` (property tests); a key no extension registered survives decode then encode; a camera drag adds 0 history entries and, after the debounce interval, the URL hash decodes to the new camera; a selection adds 1 history entry and the URL hash then decodes to the new selection; a hover, a selection and a layer toggle each perform 0 base renders, 0 lens renders and exactly 1 overlay render; probador drives each control in a committed control list and asserts a named state change for each, and a control present in the page but absent from the list fails the run; the exported PNG's pixels equal the canvas's |
| MAP-7 | MAP-6 | Lenses and linked dashboards: lens registry (linear and log scales, histogram legend with brush); new stat-tile and histogram widgets; a dock built from those and the existing chart, table and tooltip widgets, with a switcher between docks; the existing table column sort wired into the dock; two-way brushing through `Store`; a narrow-viewport layout that moves the dock to a bottom sheet | Brushing the scatter changes the map selection and brushing the legend changes the scatter, both asserted in probador; switching lens performs 0 base renders, 0 overlay renders and exactly 1 lens render, adds 1 history entry, and the URL hash then decodes to the new lens; sorting a column reorders rows to match a reference sort; at a stated narrow width the dock's bounding box is below the map's |
| MAP-8 | MAP-7 | Realtime: a WebSocket data-push endpoint in the server crate, client delta apply, per-region aggregation of cell values in WASM | One delta message that changes many cells of the active lens changes a pixel of a changed cell, leaves a pixel of an unchanged cell byte-identical, and performs exactly 1 lens render, 0 base renders and 0 overlay renders; a delta to a lens that is not active performs 0 renders of any layer, and selecting that lens afterwards shows the changed value; after a dropped connection during which the server changed the column, the client reconnects and its column equals the server's (hash compared); the WASM aggregate equals a reference sum over the same cells |
| MAP-9 | MAP-8 | Proof: line coverage, the module-only rule for target `cfg`, the wasm-only file list, pixel regression over the whole page, and a public-data demo that lives in the demo repository and is rendered from a commit merged there | Native line coverage of the target-neutral code added by MAP-1 to MAP-8 is 100%, and deleting one test drops it below; in the code MAP-1 to MAP-8 add, a `cfg` or `cfg_attr` attribute or `cfg!` macro whose predicate mentions `target_arch`, `target_family`, `target_os` or the `wasm` feature is allowed in one form only: `target_arch = "wasm32"` or its negation, on an out-of-line `mod name;` declaration. A planted one on a function, a block, a statement or an inline `mod name { }`, and a planted `target_os` predicate, each fail the check; so target-specific code is always a whole file, and the wasm-only set is the source files compiled into the wasm32 build minus the files the native coverage run instruments. That set must equal a committed list, each listed file is exercised by a named probador step, and a new file in the set but off the list fails the check; the pixel test fails on a planted one-colour change; the demo's receipt names the merged commit it was rendered from |

Order: MAP-0, MAP-1, then MAP-2 and MAP-3 together, then MAP-4 and MAP-5 together, then MAP-6, MAP-7,
MAP-8, MAP-9. The harness (MAP-2) exists before the first row that needs a browser.

Where each reference capability lands: geometry and choropleth fill MAP-1; hex cells and picking MAP-3;
columnar payload, eager and lazy loading MAP-4; raster cache, level of detail, sprites, camera, frame loop,
idle pre-paint MAP-5; input, minimap, search, URL state, history, layer toggles, help, share and export
MAP-6; lenses, docks, dock switcher, tiles, histograms, tables, scatter, mobile sheet MAP-7. Live data
(MAP-8) has no counterpart in the reference.

A WebGPU instanced tier is a follow-on row. It is opened only on a threshold proposed from the MAP-5
record; this document sets no number for it.

## 6. Open decisions

| # | Question | Recommendation |
|---|---|---|
| D1 | Geo kernel as an `aprender-viz` feature or as a new crate | Feature |
| D2 | A new epic, or rows under an existing one | New epic |
| D3 | Is the MAP-2 wasm32 and browser job a merge gate from day one | No. It starts as a non-blocking check with one ticket (MAP-2's) and one owner. It becomes merge-blocking only under the repository's rule for a new gate: the owner's explicit yes, three green nights on `main`, and one existing gate retired |
| D4 | Canvas 2D first or WebGPU first | Canvas 2D |
| D5 | Live transport: WebSocket endpoint in the server crate, or an SSE client in the present crates | WebSocket |

## 7. Risks

- **Scope creep into a GIS.** Guard: the out-of-scope list in section 4; four projections, two simplifiers.
- **WASM payload larger than the reference's JS.** Guard: MAP-2 records the compressed `.wasm` size on
  every build and MAP-4 records the eager data size; neither is a gate until a threshold is proposed.
- **A realtime path that redraws everything.** Guard: the render-count assertions in MAP-5 to MAP-8.
- **Coverage on browser-only code.** `cfg(target_arch = "wasm32")` code is not seen by the native coverage
  run, and neither is a file gated out by its parent module or by a feature. Guard: logic stays in
  target-neutral modules; MAP-9's module-only rule and computed list keep the wasm-only remainder small, named, and exercised
  in a browser.
