# Row: geom_freqpoly (PLOT-GG-001, paiml/aprender#6909): stat_bin with pad = TRUE, drawn as a path.
# Regenerate (R + ggplot2 4.0.3 only):
#   Rscript crates/aprender-viz/fixtures/ggplot2/geom_freqpoly/generate.R
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
source(file.path(here, "..", "oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

df <- utils::read.csv(file.path(here, "data.csv"))
dc <- utils::read.csv(file.path(here, "data_constant.csv"))
row <- "geom_freqpoly"
w <- function(case, plot, spec, data_file = "data.csv") {
  write_fixture(plot, row, spec, data_file, file.path(here, paste0("expected_", case, ".json")))
}

w("default", ggplot(df, aes(x)) + geom_freqpoly(), "ggplot(df, aes(x)) + geom_freqpoly()")
w("bins10", ggplot(df, aes(x)) + geom_freqpoly(bins = 10), "ggplot(df, aes(x)) + geom_freqpoly(bins = 10)")
w("binwidth2", ggplot(df, aes(x)) + geom_freqpoly(binwidth = 2), "ggplot(df, aes(x)) + geom_freqpoly(binwidth = 2)")
w("binwidth1_boundary0", ggplot(df, aes(x)) + geom_freqpoly(binwidth = 1, boundary = 0),
  "ggplot(df, aes(x)) + geom_freqpoly(binwidth = 1, boundary = 0)")
w("binwidth5_center0", ggplot(df, aes(x)) + geom_freqpoly(binwidth = 5, center = 0),
  "ggplot(df, aes(x)) + geom_freqpoly(binwidth = 5, center = 0)")
w("binwidth5_center1", ggplot(df, aes(x)) + geom_freqpoly(binwidth = 5, center = 1),
  "ggplot(df, aes(x)) + geom_freqpoly(binwidth = 5, center = 1)")
w("bins10_center0", ggplot(df, aes(x)) + geom_freqpoly(bins = 10, center = 0),
  "ggplot(df, aes(x)) + geom_freqpoly(bins = 10, center = 0)")
w("bins5_boundary2", ggplot(df, aes(x)) + geom_freqpoly(bins = 5, boundary = 2),
  "ggplot(df, aes(x)) + geom_freqpoly(bins = 5, boundary = 2)")
w("constant", ggplot(dc, aes(x)) + geom_freqpoly(), "ggplot(dc, aes(x)) + geom_freqpoly()", "data_constant.csv")
