# Row: geom_freqpoly / stat_bin + pad (PLOT-GG-001, paiml/aprender#6909). Regenerate (R + ggplot2 4.0.3, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_freqpoly/generate.R
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "dist_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

df <- read_case(here, "data.csv")
dna <- read_case(here, "data_na.csv")
dc <- read_case(here, "data_const.csv")
row <- "geom_freqpoly"
w <- function(case, plot, spec) {
  write_dist_fixture(plot, row, case, spec, file.path(here, paste0("expected_", case, ".json")))
}

w("default", ggplot(df, aes(x)) + geom_freqpoly(), "ggplot(df, aes(x)) + geom_freqpoly()")
w("bins5", ggplot(df, aes(x)) + geom_freqpoly(bins = 5), "ggplot(df, aes(x)) + geom_freqpoly(bins = 5)")
w("binwidth_boundary", ggplot(df, aes(x)) + geom_freqpoly(binwidth = 2.5, boundary = 0),
  "ggplot(df, aes(x)) + geom_freqpoly(binwidth = 2.5, boundary = 0)")
w("breaks", ggplot(df, aes(x)) + geom_freqpoly(breaks = c(0, 2, 4, 8, 16)),
  "ggplot(df, aes(x)) + geom_freqpoly(breaks = c(0, 2, 4, 8, 16))")
w("closed_left", ggplot(df, aes(x)) + geom_freqpoly(binwidth = 2, boundary = 0, closed = "left"),
  "ggplot(df, aes(x)) + geom_freqpoly(binwidth = 2, boundary = 0, closed = 'left')")
w("pad_false", ggplot(df, aes(x)) + geom_freqpoly(bins = 5, pad = FALSE),
  "ggplot(df, aes(x)) + geom_freqpoly(bins = 5, pad = FALSE)")
w("colour_two_groups", ggplot(df, aes(x, colour = g)) + geom_freqpoly(bins = 4),
  "ggplot(df, aes(x, colour = g)) + geom_freqpoly(bins = 4)")
w("colour_small_groups", ggplot(df, aes(x, colour = h)) + geom_freqpoly(binwidth = 2.5, boundary = 0),
  "ggplot(df, aes(x, colour = h)) + geom_freqpoly(binwidth = 2.5, boundary = 0)")
w("na_x", ggplot(dna, aes(x)) + geom_freqpoly(binwidth = 2, boundary = 0),
  "ggplot(dna, aes(x)) + geom_freqpoly(binwidth = 2, boundary = 0)")
w("const_x", ggplot(dc, aes(x)) + geom_freqpoly(), "ggplot(dc, aes(x)) + geom_freqpoly()")
