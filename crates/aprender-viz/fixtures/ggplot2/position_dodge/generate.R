# Fixture generator for the position_dodge conformance row (PLOT-GG-001, paiml/aprender#6909).
# Regenerate (R + ggplot2 4.0.3 only):  Rscript crates/aprender-viz/fixtures/ggplot2/position_dodge/generate.R
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
emit(ggplot(df, aes(x, fill = g)) + geom_bar(position = "dodge"),
     "position_dodge", "geom_bar(position = \"dodge\")", "expected.json")
emit(ggplot(df, aes(x, fill = g)) + geom_bar(position = position_dodge(preserve = "single")),
     "position_dodge preserve=single", "geom_bar(position = position_dodge(preserve = \"single\"))", "expected_single.json")
emit(ggplot(df, aes(x, fill = g)) + geom_bar(position = position_dodge(width = 0.5)),
     "position_dodge width=0.5", "geom_bar(position = position_dodge(width = 0.5))", "expected_width05.json")
# A dodge width wider than the bars (2 > 1.2) pushes bars outside the discrete scale's own range,
# so the panel range is the union of the expanded levels and the bars' extent.
emit(ggplot(df, aes(x, fill = g)) + geom_bar(position = position_dodge(width = 2)),
     "position_dodge width=2", "geom_bar(position = position_dodge(width = 2))", "expected_width2.json")
