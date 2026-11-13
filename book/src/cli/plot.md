# apr plot

Draw a plot from a CSV file or a YAML spec, restricted to what the ggplot2 conformance contracts prove

**Category**: Tools

## Synopsis

```text
apr plot (--spec <FILE> | --data <CSV> --aes <MAPPING>) [--geom <GEOM>] [--title <T>]
         [--xlab <T>] [--ylab <T>] [--width <N>] [--height <N>]
         [--output term|svg] [--out <FILE>] [--no-color]
```

`apr plot` draws only what the ggplot2 contracts prove. The set of drawable
rows is derived at build time from the `ggplot2_row:` blocks under
`contracts/trueno-viz/ggplot2-*.yaml` (committed as
`crates/aprender-viz/src/plot_cli/proven_rows.rs`), not typed by hand.

## What it draws today

One row is proven: `geom_point` (stat `identity`, position `identity`,
continuous x and y, cartesian coord, no facet, grey theme), from
`ggplot2-row1-point-v1`. Its break labels are checked against the R oracle's
`expected.json`.

## Refusals

Anything outside the proven set is refused by name, with a non-zero exit, an
empty stdout and the supported list on stderr:

- `--geom line` fails with `unsupported geom 'line'; supported: point`.
- A spec with `scales: {x: log10}` fails with
  `unsupported scale 'log10'; supported: continuous`.
- A stat, position, coord or theme outside the proven set is refused the same way
  (`unsupported <kind> '<value>'; supported: <list>`).
- A facet is refused as `facets are not supported yet (got 'wrap'); supported: null`.
- A `coord:` with `xlim` or `ylim` is refused as
  `coord limits (xlim) are not proven by any ggplot2 row`: ggplot2 sets the panel's
  range and breaks from them, and no row proves that.
- More than one layer is refused: each proven row draws one.
- The global `--json` and `--quiet` are refused (`plot: --json is not supported`);
  the plot is the output.
- A combination of values that no single row proves is refused, even when each
  value is proven by some row.

## Examples

<!-- example-cost: trivial -->
```bash
apr plot --help
```

These invocations are covered by `crates/apr-cli/tests/cli_commands.rs`
(`apr_plot_draws_row1_on_stdout_with_the_oracle_breaks`,
`apr_plot_writes_svg_with_the_oracle_break_labels`,
`apr_plot_refuses_geom_line_by_name`,
`apr_plot_refuses_an_unproven_scale_by_name`):

```text
apr plot --data points.csv --aes x=x,y=y
apr plot --data points.csv --aes x=x,y=y --output svg
apr plot --data points.csv --aes x=x,y=y --geom line
apr plot --spec log.yaml
```

## Full help

Run `apr plot --help` for the complete option list.

## See also

- Source: [`crates/aprender-viz/src/plot_cli/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-viz/src/plot_cli/mod.rs)
- Contract: [`contracts/trueno-viz/ggplot2-row1-point-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/trueno-viz/ggplot2-row1-point-v1.yaml)
