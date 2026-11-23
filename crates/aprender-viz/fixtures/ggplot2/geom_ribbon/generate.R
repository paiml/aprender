# Row: geom_ribbon (PLOT-GG-001, paiml/aprender#6957, master #6909): identity stat and position.
# Regenerate (R + ggplot2 4.0.3 only, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_ribbon/generate.R
# Writes expected_<case>.json beside this file. Commit them; tests never run R.
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "filled_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

row <- "geom_ribbon"
out <- function(case) file.path(here, paste0("expected_", case, ".json"))
w <- function(case, plot, spec) write_filled_fixture(plot, row, case, spec, out(case))
m <- aes(x, ymin = lo, ymax = hi)
mg <- aes(x, ymin = lo, ymax = hi, group = g)
rb <- function(file, mapping = m) ggplot(read_case(here, file), mapping) + geom_ribbon()

w("plain", rb("data.csv"), "ggplot(d, aes(x, ymin = lo, ymax = hi)) + geom_ribbon()")
w("unsorted", rb("data_unsorted.csv"), "ggplot(d, aes(x, ymin = lo, ymax = hi)) + geom_ribbon()")
w("na", rb("data_na.csv"), "ggplot(d, aes(x, ymin = lo, ymax = hi)) + geom_ribbon()")
w("groups", rb("data_groups.csv", mg), "ggplot(d, aes(x, ymin = lo, ymax = hi, group = g)) + geom_ribbon()")
w("facet", rb("data_facet.csv", mg) + facet_wrap(~f), "ggplot(d, aes(x, ymin = lo, ymax = hi, group = g)) + geom_ribbon() + facet_wrap(~f)")
w("inverted", rb("data_inverted.csv"), "ggplot(d, aes(x, ymin = lo, ymax = hi)) + geom_ribbon()")
w("negative", rb("data_negative.csv"), "ggplot(d, aes(x, ymin = lo, ymax = hi)) + geom_ribbon()")
w("layer_data", ggplot(read_case(here, "data_point.csv"), aes(x, y)) + geom_point() +
    geom_ribbon(data = read_case(here, "data_layer.csv"), aes(x, ymin = lo, ymax = hi), inherit.aes = FALSE),
  "ggplot(pts, aes(x, y)) + geom_point() + geom_ribbon(data = d, aes(x, ymin = lo, ymax = hi))")
