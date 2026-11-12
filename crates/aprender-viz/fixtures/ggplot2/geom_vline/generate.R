# Row: geom_vline (PLOT-GG-001, paiml/aprender#6909). A reference line trains the x scale
# (it is an x position) and is replicated once per facet panel.
# Regenerate (R + ggplot2 4.0.3 only, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_vline/generate.R
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "lines_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

df <- read_case(here, "data.csv")
row <- "geom_vline"
out <- function(case) file.path(here, paste0("expected_", case, ".json"))
w <- function(case, plot, spec) write_lines_fixture(plot, row, case, spec, out(case))

base <- ggplot(df, aes(x, y)) + geom_point()
w("inside", base + geom_vline(xintercept = 3), "ggplot(df, aes(x, y)) + geom_point() + geom_vline(xintercept = 3)")
w("above", base + geom_vline(xintercept = 30), "ggplot(df, aes(x, y)) + geom_point() + geom_vline(xintercept = 30)")
w("below", base + geom_vline(xintercept = -20), "ggplot(df, aes(x, y)) + geom_point() + geom_vline(xintercept = -20)")
w("two_layers", base + geom_vline(xintercept = 30) + geom_vline(xintercept = -20, colour = "red"),
  "ggplot(df, aes(x, y)) + geom_point() + geom_vline(xintercept = 30) + geom_vline(xintercept = -20, colour = 'red')")
w("facet", base + geom_vline(xintercept = 30) + facet_wrap(~f), "ggplot(df, aes(x, y)) + geom_point() + geom_vline(xintercept = 30) + facet_wrap(~f)")
w("colour_alpha", base + geom_vline(xintercept = 3, colour = "red", alpha = 0.5), "ggplot(df, aes(x, y)) + geom_point() + geom_vline(xintercept = 3, colour = 'red', alpha = 0.5)")
w("only_ref", ggplot() + geom_vline(xintercept = 2) + geom_hline(yintercept = 5), "ggplot() + geom_vline(xintercept = 2) + geom_hline(yintercept = 5)")
