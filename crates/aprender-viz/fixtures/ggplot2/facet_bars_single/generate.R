# Fixture generator for the facet_bars_single conformance row (PLOT-GG-001, paiml/aprender#6909).
# Regenerate (R + ggplot2 4.0.3 only):  Rscript crates/aprender-viz/fixtures/ggplot2/facet_bars_single/generate.R
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
# preserve = "single": n is the largest number of distinct groups at any (PANEL, xmin) cell, taken
# over ALL panels (PositionDodge$setup_params), so panel q's bars are 1/3 wide although q never has
# more than one group at an x. Panel q has no rows for g3; panel p has none at b or c.
emit(ggplot(df, aes(x, y, fill = g)) + geom_col(position = position_dodge(preserve = "single")) + facet_wrap(~f),
     "facet_bars_single dodge single", "geom_col(position = position_dodge(preserve = \"single\")) + facet_wrap(~f)", "expected_single.json")
emit(ggplot(df, aes(x, y, fill = g)) + geom_col(position = "dodge") + facet_wrap(~f),
     "facet_bars_single dodge total", "geom_col(position = \"dodge\") + facet_wrap(~f)", "expected_total.json")
emit(ggplot(df, aes(x, fill = g)) + geom_bar() + facet_wrap(~f),
     "facet_bars_single bar", "geom_bar() + facet_wrap(~f)", "expected_bar.json")
