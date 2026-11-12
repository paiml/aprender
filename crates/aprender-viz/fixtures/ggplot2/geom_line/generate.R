# Row: geom_line (PLOT-GG-001, paiml/aprender#6909). Sorts by x within PANEL and group.
# Regenerate (R + ggplot2 4.0.3 only, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_line/generate.R
# Writes expected_<case>.json beside this file. Commit them; tests never run R.
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "lines_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

df <- read_case(here, "data.csv")
dna <- read_case(here, "data_na.csv")
dlayer <- read_case(here, "data_layer.csv")
dsingle <- read_case(here, "data_single.csv")
row <- "geom_line"
out <- function(case) file.path(here, paste0("expected_", case, ".json"))
w <- function(case, plot, spec) write_lines_fixture(plot, row, case, spec, out(case))

w("plain", ggplot(df, aes(x, y)) + geom_line(), "ggplot(df, aes(x, y)) + geom_line()")
w("group_num", ggplot(df, aes(x, y, group = gn)) + geom_line(), "ggplot(df, aes(x, y, group = gn)) + geom_line()")
w("group_text", ggplot(df, aes(x, y, group = g)) + geom_line(), "ggplot(df, aes(x, y, group = g)) + geom_line()")
w("na", ggplot(dna, aes(x, y)) + geom_line(), "ggplot(dna, aes(x, y)) + geom_line()")
w("facet", ggplot(df, aes(x, y)) + geom_line() + facet_wrap(~f), "ggplot(df, aes(x, y)) + geom_line() + facet_wrap(~f)")
w("facet_group", ggplot(df, aes(x, y, group = gn)) + geom_line() + facet_wrap(~f), "ggplot(df, aes(x, y, group = gn)) + geom_line() + facet_wrap(~f)")
w("colour_alpha", ggplot(df, aes(x, y)) + geom_line(colour = "red", alpha = 0.5), "ggplot(df, aes(x, y)) + geom_line(colour = 'red', alpha = 0.5)")
w("layer_data", ggplot(df, aes(x, y)) + geom_point() + geom_line(data = dlayer), "ggplot(df, aes(x, y)) + geom_point() + geom_line(data = dlayer)")
w("single", ggplot(dsingle, aes(x, y)) + geom_line(), "ggplot(dsingle, aes(x, y)) + geom_line()")
