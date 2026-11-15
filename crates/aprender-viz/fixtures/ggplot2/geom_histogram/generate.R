# Row: geom_histogram (PLOT-GG-001, paiml/aprender#6909): stat_bin, stacked from 0.
# Regenerate (R + ggplot2 4.0.3 only):
#   Rscript crates/aprender-viz/fixtures/ggplot2/geom_histogram/generate.R
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
source(file.path(here, "..", "oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

df <- utils::read.csv(file.path(here, "data.csv"))
dc <- utils::read.csv(file.path(here, "data_constant.csv"))
row <- "geom_histogram"
w <- function(case, plot, spec, data_file = "data.csv") {
  write_fixture(plot, row, spec, data_file, file.path(here, paste0("expected_", case, ".json")))
}

w("default", ggplot(df, aes(x)) + geom_histogram(), "ggplot(df, aes(x)) + geom_histogram()")
w("bins10", ggplot(df, aes(x)) + geom_histogram(bins = 10), "ggplot(df, aes(x)) + geom_histogram(bins = 10)")
w("bins1", ggplot(df, aes(x)) + geom_histogram(bins = 1), "ggplot(df, aes(x)) + geom_histogram(bins = 1)")
w("binwidth2", ggplot(df, aes(x)) + geom_histogram(binwidth = 2), "ggplot(df, aes(x)) + geom_histogram(binwidth = 2)")
w("binwidth1_boundary0", ggplot(df, aes(x)) + geom_histogram(binwidth = 1, boundary = 0),
  "ggplot(df, aes(x)) + geom_histogram(binwidth = 1, boundary = 0)")
w("binwidth5_center0", ggplot(df, aes(x)) + geom_histogram(binwidth = 5, center = 0),
  "ggplot(df, aes(x)) + geom_histogram(binwidth = 5, center = 0)")
w("bins5_boundary2", ggplot(df, aes(x)) + geom_histogram(bins = 5, boundary = 2),
  "ggplot(df, aes(x)) + geom_histogram(bins = 5, boundary = 2)")
w("constant", ggplot(dc, aes(x)) + geom_histogram(), "ggplot(dc, aes(x)) + geom_histogram()", "data_constant.csv")
