# Fixture generator for the stat_count_geom_bar conformance row (PLOT-GG-001, paiml/aprender#6909).
# Regenerate (R + ggplot2 4.0.3 only):  Rscript crates/aprender-viz/fixtures/ggplot2/stat_count_geom_bar/generate.R
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
emit(ggplot(df, aes(x)) + geom_bar(),
     "stat_count+geom_bar", "ggplot(df, aes(x)) + geom_bar()", "expected.json")
# group = 1 makes the four bars one group, so prop = count / total (default groups give prop = 1).
emit(ggplot(df, aes(x, group = 1)) + geom_bar(),
     "stat_count+geom_bar group=1", "ggplot(df, aes(x, group = 1)) + geom_bar()", "expected_group1.json")
