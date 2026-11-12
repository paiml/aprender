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

Exported per fixture:

- layer data (one record per computed row, with every aesthetic column),
- scale breaks, labels and limits for each position scale,
- facet layout (panel table),
- the R, ggplot2, scales and labeling versions.

Layout of a row (from the first row, PR #6910):

    crates/aprender-viz/fixtures/ggplot2/oracle.R            shared base-R JSON writer (no jsonlite)
    crates/aprender-viz/fixtures/ggplot2/<row>/generate.R    the ggplot2 spec
    crates/aprender-viz/fixtures/ggplot2/<row>/data.csv      input; every value exact in f32
    crates/aprender-viz/fixtures/ggplot2/<row>/expected.json committed ggplot_build() output
    crates/aprender-viz/src/grammar/ggplot2_oracle.rs        #[cfg(test)] tests using include_str! of the fixture
    contracts/trueno-viz/ggplot2-<rowN>-<name>-v1.yaml       metadata.kind: pattern; one FALSIFY-* per guarded default

Tests live under `src/` so that `cargo test -p aprender-viz --lib`, which CI runs, picks them up.

Tolerance. Continuous values must satisfy `|a-b| <= 1e-9*max(|a|,|b|) + 1e-12`. Inputs are f32-exact, so honest differences are a few ulp, and 1e-9 is still orders below the effect of any default mutation. Labels, shapes, colours (RGBA from `col2rgb`) and counts must match exactly. A row that needs a looser bound (for example `stat_smooth` or density) states the reason in its module docs and its contract.

Fixture regeneration. Run the row's `generate.R` with `Rscript` on the oracle host, which has the pinned R and ggplot2. Then run `git diff --exit-code crates/aprender-viz/fixtures/`. The diff must be empty when the script is only re-run. Fixtures are committed JSON, so CI never needs R.

Falsifier rule. Each row guards its defaults (bins, breaks, expansion, dodge width, smoother span and so on). For each guarded default, one mutant (change the constant) is made, the row's tests are run, and the failing test names are recorded. A mutant default must turn the row red; a row where it stays green is not accepted. Each contract `prediction` is written to match what was measured. For example, row 1 found that changing the break count from 5 to 4 trips y only, and the prediction was corrected to say so.

## 5. Render targets and front ends

One plot spec renders to three targets. Pure Rust, no R at runtime.

| Target | Use |
|---|---|
| Terminal | Unicode and colour, through presentar; for terminal captures and editor-plus-plot demos. First target. |
| WASM | Browser and board/booth use. |
| SVG / PNG | Web demos and documents. |

Today `aprender-viz --features wasm` does not build for wasm32 (see section 8), so the WASM target starts with that fix.

Front ends:

- CLI: a `plot` subcommand that reads a ggplot-like spec plus CSV or Parquet. It does not exist today; terminal plot widgets are library code only.
- Rust builder API on the same plot spec.

Both front ends produce the same plot spec value, so they cannot differ in results.

## 6. Conformance matrix

Rows are the exported ggplot2 4.0.3 names (theme elements come from the element tree). The aprender side was read at origin/main `53fa483e0` on 2026-11-12, 09:45Z. For every family the row count equals the export count, the name diff is empty, every row has six columns and no name repeats.

| Family | Rows | Implemented | Partial | Absent |
|---|---|---|---|---|
| geoms | 53 | 0 | 15 | 38 |
| stats | 36 | 0 | 11 | 25 |
| positions | 8 | 0 | 0 | 8 |
| scales (+9 helpers) | 140 | 0 | 18 | 122 |
| coords | 12 | 0 | 5 | 7 |
| facets + labellers | 12 | 0 | 3 | 9 |
| core (ggplot, aes, labs, annotate, ggsave ...) | 29 | 0 | 11 | 18 |
| themes + element_* | 40 | 0 | 9 | 31 |
| theme elements | 160 | 0 | 20 | 140 |
| guides + draw_key_* | 35 | 0 | 0 | 35 |
| **Total** | **525** | **0** | **92** | **433** |

Status definitions.

- `implemented`: the same name and defaults exist in aprender-viz AND a test checks the computed values against the R oracle fixture. No row met this at the read commit, because no oracle existed on main.
- `partial`: related code exists (the symbol is named, with file and line) but the defaults or arguments differ, or no test proves the computed values. When in doubt between implemented and partial, a row is partial.
- `absent`: no code.

PR #6910 (row 1: `geom_point`, `scale_x_continuous`, `scale_y_continuous`, with oracle, contract and falsifier) turns 3 rows green when it merges: 3 of 525, 0.6%.

The rows are committed as data in `docs/specifications/plot-gg-001/matrix/` (columns and derivation commands in the README there).

## 7. Release phases

1. 0.72.6, demo set only. The `plot` subcommand exists, with the layers the demo set uses, at oracle parity, terminal target first. It does not wait for the full matrix.
2. Then the rest of the matrix in release trains. Every release note reports the percentage of matrix rows that are `implemented`, and lists the rows that went green. 100% is the goal.

The matrix work does not wait for 0.72.6, and 0.72.6 does not wait for the matrix.

Demo set. Evidence grades matter here. Reads of existing demo source code (R, Julia and Kotlin plotting code in the data-science course demos) support: `geom_point`, a fitted line (`geom_smooth(method="lm", se=FALSE)`, `geom_abline` or `geom_line`), `facet_wrap`, `coord_cartesian` limits, `labs`, `theme_minimal(base_size=18)`, continuous scales and discrete or manual colour scales. The 25 agile-practices demos name only a plot subject in words (14 of 25 have one); the mapping to `geom_col`, `geom_line`, `geom_step`, `geom_hline`/`geom_vline`, `coord_flip`, `position_dodge`/`position_stack`, `scale_x_date` and `scale_fill_manual` is INFERRED, a proposal to confirm with the demo owner. There is no evidence in any demo for log scales, histogram, boxplot, density, violin or text geoms. The brief expected histogram, boxplot and log scales; they stay in the matrix and ship when their rows go green. The exact 0.72.6 layer list is fixed after the demo owner names the plot for each demo.

The rmedia consumer pins the 0.64.0 family of aprender and has no presentar dependency, so the 0.72.6 consumer proof needs a version-bump PR in that repository. It is part of the estimate below.

## 8. Findings from the matrix lanes (source reads)

- The `Stat` enum values are never computed and no position adjustment exists; `render_layer` draws 6 geoms only.
- Extended breaks existed but the grammar did not call them (fixed 6 grid lines). PR #6910 wires them into `layout()`.
- There are no discrete scales, no aesthetic-to-colour mapping, no legends or guides, and no axis text on the framebuffer path.
- `aprender-viz --features wasm` does not build for wasm32: the wasm dependencies sit under the linux target table (exit 101).

## 9. Estimate for 100% (dated 2026-11-12; a basis, not a promise)

Basis.

- 525 rows collapse to about 210 work units (spelling aliases such as colour/color and bin2d/bin_2d, scale families, the theme element tree).
- Measured so far: 1 unit (row 1), 19 minutes of author time and 9 minutes of review, before CI.
- Assumed: an average unit costs 3 times that (stats, text and legends are harder), about 1.5 lane-hours.
- Capacity: 4 author lanes and 1 review lane (the measured subagent cap is 5), about 40 lane-hours a day, giving about 8 days.
- Multiplier of 1.5 for the hard tail (sf/map coords and geoms, hex, contour, exact loess, legends on 3 targets) and merge-flow waits, giving 12 days.

Date: **2026-11-26**, range 2026-11-24 to 2026-12-03.

Not yet measured: merged PRs per day for this work. The date is re-based after the first 3 trains merge.

## 10. Falsification of this spec

- If `implemented` rows exceed the rows that have an oracle fixture and a falsifier-checked test, the matrix is overstating. Count both.
- If a family's matrix row count differs from its export list, the matrix is stale; re-derive it.
- If any release note omits the percentage of green rows, phase 2 is not being followed.
