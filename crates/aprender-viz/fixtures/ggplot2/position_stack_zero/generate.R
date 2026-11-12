# Fixture generator for the position_stack_zero conformance row (PLOT-GG-001, paiml/aprender#6909).
# Regenerate (R + ggplot2 4.0.3 only):  Rscript crates/aprender-viz/fixtures/ggplot2/position_stack_zero/generate.R
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
# g2 at x=a and g1 at x=b have y == 0: PositionStack puts a zero-height bar in the positive stack.
emit(ggplot(df, aes(x, y, fill = g)) + geom_col(),
     "position_stack_zero", "ggplot(df, aes(x, y, fill = g)) + geom_col()", "expected.json")
