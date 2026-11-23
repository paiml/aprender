# Row: geom_abline (PLOT-GG-001, paiml/aprender#6909). An abline does NOT train either scale
# (unlike hline/vline) and is replicated once per facet panel.
# Regenerate (R + ggplot2 4.0.3 only, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_abline/generate.R
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "lines_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

df <- read_case(here, "data.csv")
row <- "geom_abline"
out <- function(case) file.path(here, paste0("expected_", case, ".json"))
w <- function(case, plot, spec) write_lines_fixture(plot, row, case, spec, out(case))

base <- ggplot(df, aes(x, y)) + geom_point()
w("plain", base + geom_abline(slope = 1.5, intercept = 0.5), "ggplot(df, aes(x, y)) + geom_point() + geom_abline(slope = 1.5, intercept = 0.5)")
w("negative", base + geom_abline(slope = -2, intercept = 4), "ggplot(df, aes(x, y)) + geom_point() + geom_abline(slope = -2, intercept = 4)")
w("far_outside", base + geom_abline(slope = 1000, intercept = 5000), "ggplot(df, aes(x, y)) + geom_point() + geom_abline(slope = 1000, intercept = 5000)")
w("zero_slope", base + geom_abline(slope = 0, intercept = -50), "ggplot(df, aes(x, y)) + geom_point() + geom_abline(slope = 0, intercept = -50)")
w("facet", base + geom_abline(slope = 1, intercept = 0) + facet_wrap(~f), "ggplot(df, aes(x, y)) + geom_point() + geom_abline(slope = 1, intercept = 0) + facet_wrap(~f)")
w("colour_alpha", base + geom_abline(slope = 1, intercept = 1, colour = "red", alpha = 0.5), "ggplot(df, aes(x, y)) + geom_point() + geom_abline(slope = 1, intercept = 1, colour = 'red', alpha = 0.5)")
