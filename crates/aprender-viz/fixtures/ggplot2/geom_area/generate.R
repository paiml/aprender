# Row: geom_area (PLOT-GG-001, paiml/aprender#6956, master #6909): stat_align + position_stack.
# Regenerate (R + ggplot2 4.0.3 only, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_area/generate.R
# Writes expected_<case>.json beside this file. Commit them; tests never run R.
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "filled_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

row <- "geom_area"
out <- function(case) file.path(here, paste0("expected_", case, ".json"))
w <- function(case, data, spec, mapping = aes(x, y, group = g), extra = NULL) {
  d <- read_case(here, data)
  p <- ggplot(d, mapping) + geom_area()
  if (!is.null(extra)) p <- p + extra
  write_filled_fixture(p, row, case, spec, out(case))
}
s <- "ggplot(d, aes(x, y, group = g)) + geom_area()"

w("equal", "data_equal.csv", s)
w("unequal", "data_unequal.csv", s)
w("cross", "data_cross.csv", s)
w("single", "data_single.csv", "ggplot(d, aes(x, y)) + geom_area()", aes(x, y))
w("dupx_single", "data_dupx_single.csv", "ggplot(d, aes(x, y)) + geom_area()", aes(x, y))
w("dupx_groups", "data_dupx_groups.csv", s)
w("facet", "data_facet.csv", paste0(s, " + facet_wrap(~f)"), extra = facet_wrap(~f))
w("onerow", "data_onerow.csv", s)
w("na", "data_na.csv", s)
w("unsorted", "data_unsorted.csv", s)
