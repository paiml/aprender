# Fixture generator for the position_dodge_dup conformance row (PLOT-GG-001, paiml/aprender#6909).
# Regenerate (R + ggplot2 4.0.3 only):  Rscript crates/aprender-viz/fixtures/ggplot2/position_dodge_dup/generate.R
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
# a/g1 appears twice: n is the number of DISTINCT groups at the x, not the number of rows.
emit(ggplot(df, aes(x, y, fill = g)) + geom_col(position = "dodge"),
     "position_dodge_dup", "geom_col(position = \"dodge\")", "expected.json")
emit(ggplot(df, aes(x, y, fill = g)) + geom_col(position = position_dodge(preserve = "single")),
     "position_dodge_dup preserve=single", "geom_col(position = position_dodge(preserve = \"single\"))", "expected_single.json")
