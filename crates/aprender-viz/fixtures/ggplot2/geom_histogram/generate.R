# Row: geom_histogram / stat_bin (PLOT-GG-001, paiml/aprender#6909). Regenerate (R + ggplot2 4.0.3, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_histogram/generate.R
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "dist_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

df <- read_case(here, "data.csv")
dna <- read_case(here, "data_na.csv")
dc <- read_case(here, "data_const.csv")
row <- "geom_histogram"
w <- function(case, plot, spec) {
  write_dist_fixture(plot, row, case, spec, file.path(here, paste0("expected_", case, ".json")))
}

w("default", ggplot(df, aes(x)) + geom_histogram(), "ggplot(df, aes(x)) + geom_histogram()")
w("bins5", ggplot(df, aes(x)) + geom_histogram(bins = 5), "ggplot(df, aes(x)) + geom_histogram(bins = 5)")
w("binwidth_boundary", ggplot(df, aes(x)) + geom_histogram(binwidth = 2.5, boundary = 0),
  "ggplot(df, aes(x)) + geom_histogram(binwidth = 2.5, boundary = 0)")
w("binwidth_center", ggplot(df, aes(x)) + geom_histogram(binwidth = 3, center = 0),
  "ggplot(df, aes(x)) + geom_histogram(binwidth = 3, center = 0)")
w("breaks", ggplot(df, aes(x)) + geom_histogram(breaks = c(0, 2, 4, 8, 16)),
  "ggplot(df, aes(x)) + geom_histogram(breaks = c(0, 2, 4, 8, 16))")
w("closed_right", ggplot(df, aes(x)) + geom_histogram(binwidth = 2, boundary = 0),
  "ggplot(df, aes(x)) + geom_histogram(binwidth = 2, boundary = 0)")
w("closed_left", ggplot(df, aes(x)) + geom_histogram(binwidth = 2, boundary = 0, closed = "left"),
  "ggplot(df, aes(x)) + geom_histogram(binwidth = 2, boundary = 0, closed = 'left')")
w("fill_two_groups", ggplot(df, aes(x, fill = g)) + geom_histogram(bins = 4),
  "ggplot(df, aes(x, fill = g)) + geom_histogram(bins = 4)")
w("fill_small_groups", ggplot(df, aes(x, fill = h)) + geom_histogram(binwidth = 2.5, boundary = 0),
  "ggplot(df, aes(x, fill = h)) + geom_histogram(binwidth = 2.5, boundary = 0)")
w("na_x", ggplot(dna, aes(x)) + geom_histogram(binwidth = 2, boundary = 0),
  "ggplot(dna, aes(x)) + geom_histogram(binwidth = 2, boundary = 0)")
w("const_x", ggplot(dc, aes(x)) + geom_histogram(),
  "ggplot(dc, aes(x)) + geom_histogram()")
w("colour_alpha", ggplot(df, aes(x)) + geom_histogram(bins = 5, colour = "blue", alpha = 0.25),
  "ggplot(df, aes(x)) + geom_histogram(bins = 5, colour = 'blue', alpha = 0.25)")
