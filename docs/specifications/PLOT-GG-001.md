# PLOT-GG-001 — `plot`: ggplot2 grammar-of-graphics parity in pure Rust

| | |
|---|---|
| **Spec id** | PLOT-GG-001 |
| **Version** | 1.0.0 (2026-11-12) |
| **Status** | PROPOSED (2026-11-12) |
| **Ticket** | paiml/aprender#6909 |
| **Reference** | ggplot2 4.0.3 on R 4.6.2 |
| **Engine** | `crates/aprender-viz` (grammar module: `crates/aprender-viz/src/grammar/`) |
| **Matrix data** | `docs/specifications/plot-gg-001/matrix/` |

## 1. Goal and non-goals

Goal. The `plot` verb implements the ggplot2 grammar of graphics: data, aesthetic mappings, layers (geom, stat, position), scales, coords, facets, themes, guides and labels. It uses the same names, the same defaults and produces the same computed results as ggplot2.

The order that started this work reads: "for 'plot' I want to have exact replica of ggplot functionality and spec."

Non-goals.

- No R at runtime and none in CI. R only regenerates committed JSON fixtures (section 4).
- No second plot engine. All work builds on `crates/aprender-viz`.
- Pixel equality with R's graphics devices is not a goal. Parity is numeric (section 4).
- No `unwrap()` in the new code (banned by clippy).

## 2. Pinned reference

| Item | Pin |
|---|---|
| ggplot2 | 4.0.3 |
| R | 4.6.2 |
| Source of names | `getNamespaceExports("ggplot2")`; theme elements from `get_element_tree()` |

Every fixture records the R, ggplot2, scales and labeling versions, and the tests assert `ggplot2 == "4.0.3"`. Changing the pin is a spec change: regenerate every fixture and re-derive the matrix.

## 3. The grammar

A plot spec is one value that holds these parts. Names follow ggplot2.

| Part | Content |
|---|---|
| data | A table (CSV or Parquet in the CLI, a data frame in the builder API). |
| aes | Mappings from columns to aesthetics: x, y, colour/color, fill, size, shape, alpha, group, linetype, linewidth, label. |
| layers | Each layer is a geom, a stat and a position, plus its own data, aes and fixed parameters. |
| scales | One per mapped aesthetic: continuous, discrete, date, log/transform, manual, identity, binned, with breaks, labels, limits and expansion. |
| coords | `coord_cartesian`, `coord_flip`, `coord_fixed`, `coord_polar`, `coord_trans`, `coord_radial`, and the others in the matrix. |
| facets | `facet_wrap`, `facet_grid`, with labellers. |
| themes | Complete themes (`theme_grey`, `theme_minimal`, ...) and the `theme()` element tree. |
| guides | Axes and legends, `guide_*`, `draw_key_*`. |
| labs | Title, subtitle, caption, axis and legend titles. |

Each piece is a row in the conformance matrix (section 6). The layer pipeline follows ggplot2: map aesthetics, compute the stat, apply the position adjustment, train and map scales, apply the coord and facet layout, then draw.

## 4. The numeric oracle

Parity is numeric, not pixels. For a fixture set, R runs `ggplot_build()` and the result is exported to JSON. aprender computes the same plot and its values are compared with the JSON.

Exported per fixture today (`crates/aprender-viz/fixtures/ggplot2/oracle.R`, `write_fixture`):

- layer data from `ggplot_build()` (`b$data`), written by `layer_json` as one array per column (not one record per row). Every column except `PANEL` and `group` is kept. Only `colour` is also resolved to RGBA (`colour_rgba`, from `col2rgb`); `fill` and other colour columns stay R colour strings. `PANEL`, `group` and resolved `fill` are extensions, listed below,
- the first panel's x and y position scales from `b$layout$panel_params[[1]]`: limits, continuous_range (limits after expansion), breaks, labels and the count of censored breaks,
- the R, ggplot2, scales and labeling versions.

Required extensions, NOT yet in the oracle (each lands with the first row that needs it):

- the facet layout table (`b$layout$layout`, one record per panel) and `panel_params` for every panel, for the facet rows (`facet_wrap`, `facet_grid`, labellers);
- a non-finite encoding: `Inf`, `-Inf` and `NA` as the strings `"Inf"`, `"-Inf"` and `"NA"`. Today `json_scalar` writes `NA` as `null` and stops with an error on `Inf`/`-Inf`, so rows that produce infinite values (`stat_ecdf` with `pad = TRUE`, `annotate("rect", xmin = -Inf, ...)`, `geom_rect` with infinite bounds) cannot have a fixture yet;
- discrete and binned scale parameters, for the discrete, binned and date scale rows;
- the `PANEL` and `group` columns of the layer data, for facet and grouped rows, and `fill` (and any other colour column) resolved to RGBA, for the fill rows.

Layout of a row (from the first row, PR #6910):

    crates/aprender-viz/fixtures/ggplot2/oracle.R            shared base-R JSON writer (no jsonlite)
    crates/aprender-viz/fixtures/ggplot2/<row>/generate.R    the ggplot2 spec
    crates/aprender-viz/fixtures/ggplot2/<row>/data.csv      input; every value exact in f32
    crates/aprender-viz/fixtures/ggplot2/<row>/expected.json committed ggplot_build() output
    crates/aprender-viz/src/grammar/ggplot2_oracle.rs        #[cfg(test)] tests using include_str! of the fixture
    contracts/trueno-viz/ggplot2-<rowN>-<name>-v1.yaml       metadata.kind: pattern; one FALSIFY-* per guarded default

Tests live under `src/` so that `cargo test -p aprender-viz --lib`, which CI runs, picks them up.

Tolerance. Continuous values must satisfy `|a-b| <= 1e-9*max(|a|,|b|) + 1e-12`. Inputs are f32-exact, so honest differences are a few ulp, and 1e-9 is still orders below the effect of any default mutation. Labels, shapes, colours (RGBA from `col2rgb`) and counts must match exactly. A row that needs a looser bound (for example `stat_smooth` or density) states the reason in its module docs and its contract.

Collation. Requirement: every fixture is generated with `LC_ALL=C` (so `LC_COLLATE=C`), for example `LC_ALL=C Rscript generate.R`. Today neither `oracle.R` nor row 1's `generate.R` sets the locale, so the caller must; row 1 has no discrete scale, so its fixture does not depend on it. The first row with a discrete scale also sets the locale inside its script. Under C collation, R sorts strings by byte value, so the default level order of a discrete scale (and of a factor made from a character column) is byte order: all upper-case letters before all lower-case ones. aprender sorts facet levels the same way (byte order, `distinct_sorted` in `crates/aprender-viz/src/grammar/facet.rs`); there are no discrete scales yet. Any other order is specified only by explicit `limits`/`levels` in the row's spec, never by locale.

Fixture regeneration. Run the row's `generate.R` with `Rscript` on the oracle host, which has the pinned R and ggplot2. Then run `git diff --exit-code crates/aprender-viz/fixtures/`. The diff must be empty when the script is only re-run. Fixtures are committed JSON, so CI never needs R.

Falsifier rule. Each row guards its defaults (bins, breaks, expansion, dodge width, smoother span and so on). For each guarded default, one mutant (change the constant) is made, the row's tests are run, and the failing test names are recorded. A mutant default must turn the row red; a row where it stays green is not accepted. Each contract `prediction` is written to match what was measured. For example, row 1 found that changing the break count from 5 to 4 trips y only, and the prediction was corrected to say so.

## 5. Render targets and front ends

One plot spec renders to three targets. Pure Rust, no R at runtime.

| Target | Use |
|---|---|
| Terminal | Unicode and colour (`BuiltGGPlot::to_terminal`, `crates/aprender-viz/src/grammar/term.rs`); for terminal captures and editor-plus-plot demos. First target. |
| WASM | Browser and board/booth use. |
| SVG / PNG | Web demos and documents. |

`aprender-viz --features wasm` builds for wasm32 since PR #6913 (see section 8). Its wasm entry points draw: `ggplot` (`crates/aprender-viz/src/wasm.rs:318`; geoms point, line, bar and area) builds a `GGPlot`, renders it with `to_framebuffer` and returns PNG bytes, as do `scatter_plot`, `line_chart`, `histogram` and `heatmap`. No test renders through the wasm target, so wherever it draws a row, that row's WASM drawn state is `untested`.

Front ends:

- CLI: `aprender-present plot` (`crates/aprender-present-cli/src/plot.rs`) exists. It reads CSV with `--data`/`--aes`/`--geom` flags or a ggplot-like YAML spec (`--spec`), draws `geom_point` only (other geoms are refused), and writes terminal or SVG output. Tests: `crates/aprender-present-cli/tests/cli_surface_gate.rs` (`plot_row1_matches_the_golden_render`, `plot_svg_carries_the_oracle_break_labels`, `plot_refuses_what_it_cannot_draw`). Parquet input is not read yet. The `apr` binary has no `plot` subcommand; the release phases (section 7) bring the verb to it.
- Rust builder API on the same plot spec.

Both front ends produce the same plot spec value, so they cannot differ in results.

## 6. Conformance matrix

Scope. ggplot2 4.0.3 exports 643 names (`getNamespaceExports("ggplot2")`). The matrix has 525 rows:

| Part | Count | Where |
|---|---|---|
| Exported names that are rows | 365 | the nine export families (`<family>.exports.txt`) |
| Theme elements (from `get_element_tree()`, not exports) | 160 | `theme-elements` |
| **Rows** | **525** | `*.tsv` |
| Exported ggproto classes, left out on purpose (`GeomPoint`, `StatBin`, `Facet`, ...) | 128 | `ggproto-classes.txt` |
| Exported functions not yet in the matrix (`continuous_scale`, `discrete_scale`, `binned_scale`, `cut_width`, `cut_interval`, `cut_number`, `mean_se`, `mean_cl_boot`, `mean_cl_normal`, `mean_sdl`, `median_hilow`, `layer_data`, `get_theme`, `set_theme`, `update_theme`, `replace_theme`, `update_geom_defaults`, `arrow`, `unit`, `alpha`, ...) | 150 | `other-exports.txt` |
| **Exports** | **643** | 365 + 128 + 150 |

The selection rule: a row exists for every export whose name starts with a family prefix or is a named helper (README, "Deriving each export list"), plus every theme element. The ggproto classes are implementation objects behind the constructors that are rows. The 150 other exports (some are user-facing: the scale constructors, the `mean_se` family used as `stat_summary` defaults, theme setters, `layer_data`) are listed but have no status yet: they are the next family to add, and until they are rows the matrix understates the work. The README gives the command that recounts all four numbers.

Statuses read at main `a554d2f66`. Every row was re-read at that commit: each symbol and line in the `aprender_symbol` column of the 148 `implemented` and `partial` rows, and each test in `proving_test`, was checked against the tree there. The `LINE` in `aprender_symbol` is valid at `a554d2f66`; the symbol name is the key. For every family the row count equals the export count (theme-elements: the element-tree count), the name diff is empty, every row has six columns and no name repeats.

| Family | Rows | Implemented | Partial | Absent |
|---|---|---|---|---|
| geoms | 53 | 1 | 18 | 34 |
| stats | 36 | 0 | 11 | 25 |
| positions | 8 | 0 | 0 | 8 |
| scales (131) + scales-helper (9) | 140 | 2 | 43 | 95 |
| coords | 12 | 0 | 5 | 7 |
| facets + labellers | 12 | 0 | 3 | 9 |
| core (ggplot, aes, labs, annotate, ggsave ...) | 29 | 0 | 13 | 16 |
| themes + element_* | 40 | 0 | 10 | 30 |
| theme elements | 160 | 0 | 41 | 119 |
| guides + draw_key_* | 35 | 0 | 1 | 34 |
| **Total** | **525** | **3** | **145** | **377** |

Status definitions.

- `implemented`: ALL of the following hold for the row's default call. (1) The same name and defaults exist in aprender-viz. (2) An R fixture (committed `ggplot_build()` JSON) exists. (3) A Rust test compares the computed values with that fixture at the stated tolerance (section 4). (4) A contract under `contracts/` states the row and carries a falsifier whose mutant was measured to turn the row red (section 4). (5) The code is reachable from the public grammar path: `BuiltGGPlot::layout()` (builder API today, the `plot` verb later), not only from a private helper. Arguments the fixture does not exercise are listed in the notes; they are new fixtures, not a different status.
- `partial`: related code exists (the symbol is named, with file and line) that computes or draws part of the row's own behaviour, but the defaults or arguments differ, or no test proves the computed values. A symbol that is only a declared enum variant or constructor counts, as it always has in the stats and geoms families; a code path for a different feature (for example `Coord::flip` for `scale_x_reverse`) does not. When in doubt between implemented and partial, a row is partial.
- `absent`: no related code. A note that names code means the row is `partial`, unless the note says that code is a different feature.

`drawn` is a separate state, per target: terminal, SVG, framebuffer (PNG) and WASM. `implemented` says the numbers `layout()` computes match ggplot2; it does not say anything is drawn. Each target gets one value:

- `oracle`: a test renders that target and checks the drawn output against values from the row's ggplot2 fixture;
- `checked`: a test renders that target and checks the drawn output (counts, positions, a golden render), but not against fixture values;
- `smoke`: a test renders that target and checks only that it succeeds (and its size);
- `untested`: the target draws the row, but no test renders it;
- `-`: the target does not draw the row.

The three `implemented` rows carry a `[drawn: terminal=... svg=... framebuffer=... wasm=...]` tag at the end of the notes; rows without the tag claim nothing about drawing. The tag is a notes convention, not a seventh column, so every TSV keeps six columns.

Row 1 (PR #6910) made `geom_point`, `scale_x_continuous` and `scale_y_continuous` `implemented`: 3 of 525, 0.6%. Their drawn states at `a554d2f66`:

- `scale_x_continuous`, `scale_y_continuous`: `terminal=oracle svg=oracle framebuffer=smoke wasm=untested`. The terminal test `crates/aprender-viz/src/grammar/term.rs::tests::draws_every_point_and_the_oracle_break_labels` asserts ggplot2 4.0.3's break labels for row 1's data (x: 5, 10; y: -2.5 to 7.5); `crates/aprender-present-cli/tests/cli_surface_gate.rs::plot_svg_carries_the_oracle_break_labels` does the same for the SVG. `crates/aprender-viz/src/grammar/ggplot.rs::tests::test_ggplot_basic` renders the framebuffer and checks its size only.
- `geom_point`: `terminal=checked svg=smoke framebuffer=smoke wasm=untested`. The terminal test counts all 10 points; no test checks point positions in the SVG or the framebuffer.

The rows are committed as data in `docs/specifications/plot-gg-001/matrix/` (columns, derivation and recount commands in the README there).
## 7. Release phases

1. 0.72.6, demo set only. The `plot` subcommand exists, with the layers the demo set uses, at oracle parity, terminal target first. It does not wait for the full matrix.
2. Then the rest of the matrix in release trains. Every release note reports the percentage of matrix rows that are `implemented`, and lists the rows that went green. 100% is the goal.

The matrix work does not wait for 0.72.6, and 0.72.6 does not wait for the matrix.

Demo set. Evidence grades matter here. Reads of existing demo source code (R, Julia and Kotlin plotting code in existing teaching demos) support: `geom_point`, a fitted line (`geom_smooth(method="lm", se=FALSE)`, `geom_abline` or `geom_line`), `facet_wrap`, `coord_cartesian` limits, `labs`, `theme_minimal(base_size=18)`, continuous scales and discrete or manual colour scales. A further set of 25 demos names only a plot subject in words (14 of 25 have one); the mapping to `geom_col`, `geom_line`, `geom_step`, `geom_hline`/`geom_vline`, `coord_flip`, `position_dodge`/`position_stack`, `scale_x_date` and `scale_fill_manual` is INFERRED, a proposal to confirm with the demo authors. There is no evidence in any demo for log scales, histogram, boxplot, density, violin or text geoms. The brief expected histogram, boxplot and log scales; they stay in the matrix and ship when their rows go green. The exact 0.72.6 layer list is fixed after the demo authors name the plot for each demo.

A downstream consumer pins the 0.64.0 family of aprender and has no presentar dependency, so the 0.72.6 consumer proof needs a version-bump PR there. It is part of the estimate below.

## 8. Findings from the source reads (at `a554d2f66`)

- The `Stat` enum values are never computed and no position adjustment exists; `render_layer` draws 6 geoms only (point, line, bar, area, hline, vline).
- `layout()` uses extended breaks (`PositionScale`, PR #6910). The framebuffer draws one grid line per break of each position scale (`draw_grid`).
- Tick marks and break labels are drawn in SVG (`svg_axis_text`) and in the terminal target (`to_terminal`); the framebuffer has no font, so it draws no axis text, title or axis titles.
- There are no discrete scales, no aesthetic-to-colour mapping, and no legends; the only guide drawn is the bottom and left axis.
- `aprender-viz --features wasm` builds for wasm32 since PR #6913. The wasm entry points draw through `to_framebuffer` (section 5); no test renders through them.

## 9. Estimate for 100% (dated 2026-11-12; a basis, not a promise)

Basis.

- 525 rows collapse to about 210 work units (spelling aliases such as colour/color and bin2d/bin_2d, scale families, the theme element tree).
- Measured so far: 1 unit (row 1), 19 minutes of author time and 9 minutes of review, before CI.
- Assumed: an average unit costs 3 times that (stats, text and legends are harder), about 1.5 hours of combined author and review time.
- Capacity: 4 authors and 1 reviewer working in parallel, about 40 hours of work a day, giving about 8 days.
- Multiplier of 1.5 for the hard tail (sf/map coords and geoms, hex, contour, exact loess, legends on 3 targets) and merge-flow waits, giving 12 days.

Date: **2026-11-26**, range 2026-11-24 to 2026-12-03.

Not yet measured: merged PRs per day for this work. The date is re-based after the first 3 trains merge.

## 10. Falsification of this spec

- If `implemented` rows exceed the rows that have an oracle fixture and a falsifier-checked test, the matrix is overstating. Count both.
- If a family's matrix row count differs from its export list, the matrix is stale; re-derive it.
- If any release note omits the percentage of green rows, phase 2 is not being followed.
