# Row: geom_rect (PLOT-GG-001, paiml/aprender#6909). GeomRect$setup_data returns the data as it is when
# xmin, xmax, ymin and ymax are all mapped; the scales train on all four, whatever their order.
# Regenerate (R + ggplot2 4.0.3 only, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_rect/generate.R
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "rects_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

d <- read_case(here, "data.csv")
dr <- read_case(here, "data_reversed.csv")
dz <- read_case(here, "data_zero_area.csv")
ds <- read_case(here, "data_single.csv")
dsz <- read_case(here, "data_single_zero.csv")
dn <- read_case(here, "data_negative.csv")
dna <- read_case(here, "data_na.csv")
dpt <- read_case(here, "data_point.csv")
row <- "geom_rect"
out <- function(case) file.path(here, paste0("expected_", case, ".json"))
w <- function(case, plot, spec) write_rects_fixture(plot, row, case, spec, out(case))
m <- aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)

w("default", ggplot(d, m) + geom_rect(), "ggplot(d, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)) + geom_rect()")
w("reversed", ggplot(dr, m) + geom_rect(), "ggplot(dr, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)) + geom_rect()")
w("zero_area", ggplot(dz, m) + geom_rect(), "ggplot(dz, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)) + geom_rect()")
w("single", ggplot(ds, m) + geom_rect(), "ggplot(ds, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)) + geom_rect()")
w("single_zero_area", ggplot(dsz, m) + geom_rect(), "ggplot(dsz, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)) + geom_rect()")
w("negative", ggplot(dn, m) + geom_rect(), "ggplot(dn, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)) + geom_rect()")
w("na_rows", ggplot(dna, m) + geom_rect(), "ggplot(dna, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)) + geom_rect()")
w("fill_colour_alpha", ggplot(d, m) + geom_rect(fill = "red", colour = "blue", alpha = 0.5),
  "ggplot(d, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)) + geom_rect(fill = 'red', colour = 'blue', alpha = 0.5)")
# The point (7, -2) lies outside the rect envelope (x -1..5, y 0..6), so the shared scales widen to
# x [-1, 7] and y [-2, 6]: a layer that did not train the point would leave them at [-1, 5], [0, 6].
w("with_point", ggplot(d) + geom_rect(m) + geom_point(data = dpt, aes(x, y)),
  "ggplot(d) + geom_rect(aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)) + geom_point(data = dpt, aes(x, y))")
