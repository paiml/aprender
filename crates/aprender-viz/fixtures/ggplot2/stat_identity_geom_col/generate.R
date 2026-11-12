# Fixture generator for the stat_identity_geom_col conformance row (PLOT-GG-001, paiml/aprender#6909).
# Regenerate (R + ggplot2 4.0.3 only):  Rscript crates/aprender-viz/fixtures/ggplot2/stat_identity_geom_col/generate.R
# Writes the expected*.json files beside this script. Commit them; tests never run R.
here <- (function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
})()
source(file.path(here, "..", "oracle.R"))
suppressPackageStartupMessages(library(ggplot2))
# Every numeric value is exactly representable in binary32 (aprender-viz stores f32).
# Discrete levels are lowercase ASCII so R's locale-dependent sort and a byte sort agree.
df <- utils::read.csv(file.path(here, "data.csv"), stringsAsFactors = FALSE)
# Group and PANEL are kept: stacking and dodging order by group.
emit <- function(plot, row, spec, file) {
  write_fixture(plot, row = row, spec = spec, data_file = "data.csv",
                out_file = file.path(here, file), drop = character())
}
# Default position is stack; negative values stack downward from 0 separately.
emit(ggplot(df, aes(x, y, fill = g)) + geom_col(),
     "stat_identity+geom_col", "ggplot(df, aes(x, y, fill = g)) + geom_col()", "expected.json")
# One row per x: no fill, nothing to stack.
one <- df[df$g == "g1", ]
emit(ggplot(one, aes(x, y)) + geom_col(),
     "stat_identity+geom_col plain", "ggplot(df[df$g == \"g1\", ], aes(x, y)) + geom_col()", "expected_plain.json")
