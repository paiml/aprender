# Fixture generator for the stat_count_geom_bar conformance row (PLOT-GG-001, paiml/aprender#6909).
# Regenerate (R + ggplot2 4.0.3 only):  Rscript crates/aprender-viz/fixtures/ggplot2/stat_count_geom_bar/generate.R
# Writes the expected*.json files beside this script. Commit them; tests never run R.
here <- (function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
})()
source(file.path(here, "..", "oracle.R"))
suppressPackageStartupMessages(library(ggplot2))
invisible(Sys.setlocale("LC_COLLATE", "C"))
stopifnot(Sys.getlocale("LC_COLLATE") == "C")
# Every numeric value is exactly representable in binary32 (aprender-viz stores f32).
# The oracle is pinned to LC_COLLATE=C: R sorts factor levels by the locale, and C is bytewise
# (uppercase before lowercase), the order aprender-viz implements. The value is recorded in each JSON.
df <- utils::read.csv(file.path(here, "data.csv"), stringsAsFactors = FALSE)
# Group and PANEL are kept: stacking and dodging order by group.
emit <- function(plot, row, spec, file, data_file = "data.csv") {
  write_fixture(plot, row = row, spec = spec, data_file = data_file,
                out_file = file.path(here, file), drop = character(),
                collate = Sys.getlocale("LC_COLLATE"))
}
emit(ggplot(df, aes(x)) + geom_bar(),
     "stat_count+geom_bar", "ggplot(df, aes(x)) + geom_bar()", "expected.json")
# group = 1 makes the four bars one group, so prop = count / total (default groups give prop = 1).
emit(ggplot(df, aes(x, group = 1)) + geom_bar(),
     "stat_count+geom_bar group=1", "ggplot(df, aes(x, group = 1)) + geom_bar()", "expected_group1.json")
# width = 0.5 overrides the 0.9 default: bars are 0.5 wide, centred on the level.
emit(ggplot(df, aes(x)) + geom_bar(width = 0.5),
     "stat_count+geom_bar width=0.5", "ggplot(df, aes(x)) + geom_bar(width = 0.5)", "expected_width05.json")
