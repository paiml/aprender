# Fixture generator for the bars_companions conformance row (PLOT-GG-001, paiml/aprender#6909).
# Regenerate (R + ggplot2 4.0.3 only):  Rscript crates/aprender-viz/fixtures/ggplot2/bars_companions/generate.R
# Writes the expected_*.json files beside this script. Commit them; tests never run R.
#
# Bar layers over a discrete x beside reference lines and points, the group order of a layer-level
# fill, and position_fill over a zero total.
here <- (function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
})()
source(file.path(here, "..", "oracle.R"))
suppressPackageStartupMessages(library(ggplot2))
invisible(Sys.setlocale("LC_COLLATE", "C"))
stopifnot(Sys.getlocale("LC_COLLATE") == "C")
# Every numeric value is exactly representable in binary32 (aprender-viz stores f32).
df <- utils::read.csv(file.path(here, "data.csv"), stringsAsFactors = FALSE)
emit <- function(plot, spec, name) {
  write_fixture(plot, row = paste("bars_companions", name), spec = spec, data_file = "data.csv",
                out_file = file.path(here, paste0("expected_", name, ".json")), drop = character(),
                collate = Sys.getlocale("LC_COLLATE"))
}
# geom_vline trains the continuous part of the discrete x scale (range_c), on either side.
emit(ggplot(df, aes(x, y)) + geom_col() + geom_vline(xintercept = 5),
     "geom_col() + geom_vline(xintercept = 5)", "col_vline_right")
emit(ggplot(df, aes(x, y)) + geom_col() + geom_vline(xintercept = -1.5),
     "geom_col() + geom_vline(xintercept = -1.5)", "col_vline_left")
emit(ggplot(df, aes(x)) + geom_bar() + geom_hline(yintercept = 4) + geom_vline(xintercept = 0.25),
     "geom_bar() + geom_hline(4) + geom_vline(0.25)", "bar_hline_vline")
emit(ggplot(df, aes(x, y)) + geom_bar(stat = "identity", width = 0.5) + geom_vline(xintercept = 3.25),
     "geom_bar(width = 0.5) + geom_vline(3.25) inside the bars", "col_width_vline_inside")
emit(ggplot(df, aes(x, y)) + geom_col() + geom_hline(yintercept = -5) + geom_hline(yintercept = 7),
     "geom_col() + two geom_hline", "col_two_hlines")
emit(ggplot(df, aes(x, y)) + geom_hline(yintercept = -5) + geom_col(),
     "geom_hline() before geom_col()", "hline_then_col")
# A text x on a point layer adds levels to the discrete scale (d and e are not bar levels).
emit(ggplot(df, aes(x, y)) + geom_col() + geom_point(aes(x = xl, y = z)),
     "geom_col() + geom_point(aes(x = xl, y = z))", "col_point_new_level")
emit(ggplot(df) + geom_point(aes(x = xl, y = z)) + geom_bar(aes(x = x)),
     "geom_point(aes(x = xl)) before geom_bar(aes(x))", "point_then_bar")
emit(ggplot(df, aes(x, y)) + geom_col() + geom_point(aes(x = xl, y = z)) + geom_vline(xintercept = 6),
     "geom_col() + geom_point(xl) + geom_vline(6)", "col_point_vline")
# add_group(): the discrete columns are ranked in computed_mapping order, the layer's own
# aesthetics first, then the plot's. A layer-level fill under a plot-level x ranks fill first.
emit(ggplot(df, aes(x)) + geom_bar(aes(fill = g)) + geom_col(aes(y = y, fill = g), position = "dodge"),
     "aes(x) + geom_bar(aes(fill = g)) + geom_col(aes(y, fill = g), dodge)", "layer_fill_two_layers")
emit(ggplot(df, aes(x, y)) + geom_col(aes(fill = g), position = "dodge"),
     "aes(x, y) + geom_col(aes(fill = g), dodge)", "layer_fill_dodge")
emit(ggplot(df, aes(x, y)) + geom_col(aes(fill = g)),
     "aes(x, y) + geom_col(aes(fill = g))", "layer_fill_stack")
emit(ggplot(df, aes(fill = g)) + geom_bar(aes(x = x), position = "dodge"),
     "aes(fill = g) + geom_bar(aes(x), dodge)", "plot_fill_layer_x")
emit(ggplot(df) + geom_bar(aes(x = x, fill = g), position = "dodge"),
     "geom_bar(aes(x, fill = g), dodge)", "layer_x_and_fill")
# position_fill over a column whose total is zero leaves the heights unscaled (pos_stack guard).
emit(ggplot(df, aes(x, y0, fill = g)) + geom_col(position = "fill"),
     "geom_col(position = \"fill\") with zero totals", "fill_zero_sum")
emit(ggplot(df, aes(x, ys, fill = g)) + geom_col(position = "fill"),
     "geom_col(position = \"fill\") with negative bars", "fill_signed")
# Reference lines under facet_wrap: one row per panel.
emit(ggplot(df, aes(x, fill = g)) + geom_bar() + geom_hline(yintercept = 2) + facet_wrap(~f),
     "geom_bar() + geom_hline(2) + facet_wrap(~f)", "facet_bar_hline")
emit(ggplot(df, aes(x, y, fill = g)) + geom_col(position = "dodge") + geom_vline(xintercept = 3.75) + facet_wrap(~f),
     "geom_col(dodge) + geom_vline(3.75) + facet_wrap(~f)", "facet_col_vline")
emit(ggplot(df, aes(x, y, fill = g)) + geom_col(position = position_dodge(preserve = "single")) + geom_vline(xintercept = 0),
     "geom_col(dodge single) + geom_vline(0)", "dodge_single_vline")
# One level only: the discrete scale is (1, 1) and still not degenerate.
emit(ggplot(df[df$x == "a", ], aes(x, y)) + geom_col() + geom_hline(yintercept = 1),
     "geom_col() on one level + geom_hline(1)", "single_level_hline")
# A NaN y: stat_identity keeps the row in ggplot_build().
emit(ggplot(df, aes(x, yn)) + geom_col(),
     "geom_col() with a NaN y", "col_nan_y")
