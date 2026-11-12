# Fixture generator for the stat_count_case conformance row (PLOT-GG-001, paiml/aprender#6909).
# Regenerate (R + ggplot2 4.0.3 only):  Rscript crates/aprender-viz/fixtures/ggplot2/stat_count_case/generate.R
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
# C collation: levels are B, a, c (not a, B, c as en_US would give).
emit(ggplot(df, aes(x)) + geom_bar(),
     "stat_count_case", "ggplot(df, aes(x)) + geom_bar()", "expected.json")
# Fill levels Z < h under C: group ids and the stacking order follow it.
emit(ggplot(df, aes(x, fill = g)) + geom_bar(),
     "stat_count_case fill", "ggplot(df, aes(x, fill = g)) + geom_bar()", "expected_fill.json")
