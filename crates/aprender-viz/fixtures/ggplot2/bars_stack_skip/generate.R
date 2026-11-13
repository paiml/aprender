# Fixture generator for the bars_stack_skip conformance row (PLOT-GG-001, paiml/aprender#6909).
# Regenerate (R + ggplot2 4.0.3 only):  Rscript crates/aprender-viz/fixtures/ggplot2/bars_stack_skip/generate.R
# Writes the expected_*.json files beside this script. Commit them; tests never run R.
#
# PositionStack$setup_params leaves the data alone when no x repeats in the whole layer, fill is off
# and vjust is 1: a negative bar keeps y = value. Any one of the three conditions stacks it.
here <- (function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
})()
source(file.path(here, "..", "oracle.R"))
suppressPackageStartupMessages(library(ggplot2))
invisible(Sys.setlocale("LC_COLLATE", "C"))
stopifnot(Sys.getlocale("LC_COLLATE") == "C")
rd <- function(f) utils::read.csv(file.path(here, f), stringsAsFactors = FALSE)
emit <- function(plot, spec, name, data_file) {
  write_fixture(plot, row = paste("bars_stack_skip", name), spec = spec, data_file = data_file,
                out_file = file.path(here, paste0("expected_", name, ".json")), drop = character(),
                collate = Sys.getlocale("LC_COLLATE"))
}
df <- rd("data.csv"); one <- rd("one.csv"); cross <- rd("cross.csv")
emit(ggplot(df, aes(x, y)) + geom_col(), "geom_col(), unique x, negative y", "unique", "data.csv")
emit(ggplot(one, aes(x, y)) + geom_col(), "geom_col(), one negative row", "one_row", "one.csv")
emit(ggplot(df, aes(x, y)) + geom_col() + facet_wrap(~f), "geom_col() + facet_wrap, unique x", "unique_facet", "data.csv")
emit(ggplot(df, aes(x, y)) + geom_col(position = position_stack(vjust = 0.5)),
     "position_stack(vjust = 0.5) stacks", "vjust05", "data.csv")
emit(ggplot(df, aes(x, y)) + geom_col(position = "fill"), "position_fill stacks", "fill", "data.csv")
emit(ggplot(cross, aes(x, y)) + geom_col() + facet_wrap(~f),
     "x repeats only across panels: stacks", "cross_panel", "cross.csv")
