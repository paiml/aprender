# Row: geom_polygon (PLOT-GG-001, paiml/aprender#6909). Layer data keeps the input rows and their
# order, one group per polygon; no vertex is added, sorted or closed at build time, so a group of
# fewer than three vertices is still its rows (it is only drawn differently). fill is the theme's
# 20% mix of ink on paper, "#333333FF"; colour is NA.
# Regenerate (R + ggplot2 4.0.3 only, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_polygon/generate.R
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "lines_oracle.R"))
source(file.path(here, "..", "pack_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

df <- read_case(here, "data.csv")
row <- "geom_polygon"
out <- function(case) file.path(here, paste0("expected_", case, ".json"))
w <- function(case, plot, spec) write_pack_fixture(plot, row, case, spec, out(case))

w("one_group", ggplot(df, aes(x, y)) + geom_polygon(), "ggplot(df, aes(x, y)) + geom_polygon()")
w("group_num", ggplot(df, aes(x, y, group = gn)) + geom_polygon(), "ggplot(df, aes(x, y, group = gn)) + geom_polygon()")
w("group_text", ggplot(df, aes(x, y, group = g)) + geom_polygon(), "ggplot(df, aes(x, y, group = g)) + geom_polygon()")
w("single", ggplot(read_case(here, "data_single.csv"), aes(x, y)) + geom_polygon(), "ggplot(ds, aes(x, y)) + geom_polygon()")
w("degenerate", ggplot(read_case(here, "data_degenerate.csv"), aes(x, y, group = gn)) + geom_polygon(), "ggplot(dd, aes(x, y, group = gn)) + geom_polygon()")
w("negative", ggplot(read_case(here, "data_negative.csv"), aes(x, y, group = gn)) + geom_polygon(), "ggplot(dn, aes(x, y, group = gn)) + geom_polygon()")
w("facet", ggplot(df, aes(x, y, group = gn)) + geom_polygon() + facet_wrap(~f), "ggplot(df, aes(x, y, group = gn)) + geom_polygon() + facet_wrap(~f)")
w("colour_alpha", ggplot(df, aes(x, y, group = gn)) + geom_polygon(colour = "red", alpha = 0.5), "ggplot(df, aes(x, y, group = gn)) + geom_polygon(colour = 'red', alpha = 0.5)")
w("na", ggplot(read_case(here, "data_na.csv"), aes(x, y, group = gn)) + geom_polygon(), "ggplot(dna, aes(x, y, group = gn)) + geom_polygon()")
