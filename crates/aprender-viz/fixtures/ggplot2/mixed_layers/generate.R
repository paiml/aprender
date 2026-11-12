# Fixture generator for the mixed_layers conformance row (PLOT-GG-001, paiml/aprender#6909).
# Regenerate (R + ggplot2 4.0.3 only):  Rscript crates/aprender-viz/fixtures/ggplot2/mixed_layers/generate.R
# Writes the expected*.json files beside this script. Commit them; tests never run R.
here <- (function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
})()
source(file.path(here, "..", "oracle.R"))
suppressPackageStartupMessages(library(ggplot2))
# The oracle is pinned to LC_COLLATE=C (bytewise: uppercase before lowercase); recorded in each JSON.
invisible(Sys.setlocale("LC_COLLATE", "C"))
stopifnot(Sys.getlocale("LC_COLLATE") == "C")
# Every numeric value is exactly representable in binary32 (aprender-viz stores f32).
df <- utils::read.csv(file.path(here, "data.csv"), stringsAsFactors = FALSE)
emit <- function(plot, row, spec, file) {
  write_fixture(plot, row = row, spec = spec, data_file = "data.csv",
                out_file = file.path(here, file), drop = character(),
                collate = Sys.getlocale("LC_COLLATE"))
}
# Position scales are trained over ALL layers: y over every layer's y (and ymin/ymax of bars and
# hline intercepts), x over the union of the layers' levels plus the continuous range (range_c).
emit(ggplot(df, aes(x, y)) + geom_col() + geom_point(aes(y = z)),
     "mixed_layers col+point(y=z)", "geom_col() + geom_point(aes(y = z))", "expected_col_point.json")
emit(ggplot(df, aes(x, y)) + geom_col() + geom_hline(yintercept = 30),
     "mixed_layers col+hline", "geom_col() + geom_hline(yintercept = 30)", "expected_col_hline.json")
# Points whose numeric x lies outside the bars (-1 and 4.5 vs bars in 0.55..3.45).
emit(ggplot(df, aes(x, y)) + geom_col() + geom_point(aes(x = xn, y = z)),
     "mixed_layers col+point(x=xn)", "geom_col() + geom_point(aes(x = xn, y = z))", "expected_col_point_x.json")
# Two bar layers with different levels (a,b and b,c): the discrete x scale unions them.
d1 <- df[df$x %in% c("a", "b"), ]
d2 <- df[df$x %in% c("b", "c"), ]
emit(ggplot() + geom_bar(data = d1, aes(x)) + geom_col(data = d2, aes(x, y, fill = g), position = "dodge"),
     "mixed_layers bar+col", "geom_bar(data = d1) + geom_col(data = d2, position = \"dodge\")", "expected_two_bars.json")
