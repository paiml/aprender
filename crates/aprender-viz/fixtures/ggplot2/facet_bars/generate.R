# Fixture generator for the facet_bars conformance row (PLOT-GG-001, paiml/aprender#6909).
# Regenerate (R + ggplot2 4.0.3 only):  Rscript crates/aprender-viz/fixtures/ggplot2/facet_bars/generate.R
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
# Stat and position are computed PER PANEL (group ids are global, from add_group before the facet
# split); the position scales are trained across panels (scales = "fixed").
emit(ggplot(df, aes(x, fill = g)) + geom_bar() + facet_wrap(~f),
     "facet_bars bar", "geom_bar() + facet_wrap(~f)", "expected_bar.json")
emit(ggplot(df, aes(x, y, fill = g)) + geom_col(position = "dodge") + facet_wrap(~f),
     "facet_bars dodge", "geom_col(position = \"dodge\") + facet_wrap(~f)", "expected_dodge.json")
emit(ggplot(df, aes(x, fill = g)) + geom_bar(position = "fill") + facet_wrap(~f),
     "facet_bars fill", "geom_bar(position = \"fill\") + facet_wrap(~f)", "expected_fill.json")
emit(ggplot(df, aes(x, y, fill = g)) + geom_col() + facet_wrap(~f),
     "facet_bars stack", "geom_col() + facet_wrap(~f)", "expected_stack.json")
