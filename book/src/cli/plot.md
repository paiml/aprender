# apr plot

Draw a plot from a CSV file, restricted to what the ggplot2 conformance contracts prove

**Category**: Tools

## Synopsis

```text
apr plot --data <CSV> --aes <MAPPING> [--geom <GEOM>] [--width <N>] [--height <N>]
         [--output term|svg] [--no-color]
```

`apr plot` draws only what the ggplot2 contracts prove. The set of drawable
rows is derived at build time from the `ggplot2_row:` blocks under
`contracts/trueno-viz/ggplot2-*.yaml` (committed as
`crates/aprender-viz/src/plot_cli/proven_rows.rs`), not typed by hand.

## What it draws today

One row is proven: `geom_point` (stat `identity`, position `identity`,
continuous x and y, cartesian coord, no facet, grey theme), from
`ggplot2-row1-point-v1`. Its break labels are checked against the R oracle's
`expected.json`. The scales are trained on the `f64` values, as ggplot2 trains
on doubles.

## Refusals

Anything outside the proven set is refused by name, with a non-zero exit, an
empty stdout and the supported list on stderr:

- `--geom line` fails with `unsupported geom 'line'; supported: point`.
- `--aes x=a,y=b,colour=c` fails with `unsupported aesthetic 'colour'; supported: x, y`,
  and a mapping without `x` or `y` fails with `aes needs a mapping for 'x' (e.g. x=column)`.
- An infinite value (`Inf`, `-inf`, `infinity`, `1e400`) is refused by column and line.
  Only empty cells, `NA` and NaN are dropped, with ggplot2's warning
  (`Removed 1 row containing missing values ...`).
- A value whose magnitude is outside the compared range (zero, or 1e-5 up to but
  excluding 1e16) is refused by column and line.
- A column the engine cannot store without changing a label is refused by line and column:
  the engine holds `f32`, ggplot2 holds doubles, and a value that moves by more than 1e-4 of
  its column's range would print other labels than R. The message names the value, what the
  engine stores (`binary32`) and the bound.
- A column with one value on every row is refused by name: ggplot2 widens that
  zero-width range, and the engine does not yet.
- The global `--json` and `--quiet` are refused (`plot: --json is not supported`);
  the plot is the output.

## Examples

<!-- example-cost: trivial -->
```bash
apr plot --help
```

These invocations are covered by `crates/apr-cli/tests/cli_commands.rs`
(`apr_plot_draws_row1_on_stdout_with_the_oracle_breaks`,
`apr_plot_writes_svg_with_the_oracle_break_labels`,
`apr_plot_refuses_geom_line_by_name`):

```text
apr plot --data points.csv --aes x=x,y=y
apr plot --data points.csv --aes x=x,y=y --output svg
apr plot --data points.csv --aes x=x,y=y --geom line
```

## Full help

Run `apr plot --help` for the complete option list.

## See also

- Source: [`crates/aprender-viz/src/plot_cli/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-viz/src/plot_cli/mod.rs)
- Contract: [`contracts/trueno-viz/ggplot2-row1-point-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/trueno-viz/ggplot2-row1-point-v1.yaml)
