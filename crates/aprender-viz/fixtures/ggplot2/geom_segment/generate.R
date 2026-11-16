# Row: geom_segment (PLOT-GG-001, paiml/aprender#6909). Every segment is one row of layer data with
# x, y, xend and yend (xend or yend alone leaves the other column out of the built data); the
# position scales train on the start and the end points.
# Regenerate (R + ggplot2 4.0.3 only, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_segment/generate.R
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "lines_oracle.R"))
source(file.path(here, "..", "pack_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

df <- read_case(here, "data.csv")
row <- "geom_segment"
out <- function(case) file.path(here, paste0("expected_", case, ".json"))
w <- function(case, plot, spec) write_pack_fixture(plot, row, case, spec, out(case))
full <- aes(x, y, xend = xend, yend = yend)

w("plain", ggplot(df, full) + geom_segment(), "ggplot(df, aes(x, y, xend = xend, yend = yend)) + geom_segment()")
w("reversed", ggplot(read_case(here, "data_reversed.csv"), full) + geom_segment(), "ggplot(dr, aes(x, y, xend = xend, yend = yend)) + geom_segment()")
w("zero_length", ggplot(read_case(here, "data_zero.csv"), full) + geom_segment(), "ggplot(dz, aes(x, y, xend = xend, yend = yend)) + geom_segment()")
w("negative", ggplot(read_case(here, "data_negative.csv"), full) + geom_segment(), "ggplot(dn, aes(x, y, xend = xend, yend = yend)) + geom_segment()")
w("ends_far", ggplot(read_case(here, "data_ends_far.csv"), full) + geom_segment(), "ggplot(de, aes(x, y, xend = xend, yend = yend)) + geom_segment()")
w("xend_only", ggplot(df, aes(x, y, xend = xend)) + geom_segment(), "ggplot(df, aes(x, y, xend = xend)) + geom_segment()")
w("yend_only", ggplot(df, aes(x, y, yend = yend)) + geom_segment(), "ggplot(df, aes(x, y, yend = yend)) + geom_segment()")
w("group_facet", ggplot(df, aes(x, y, xend = xend, yend = yend, group = gn)) + geom_segment() + facet_wrap(~f), "ggplot(df, aes(x, y, xend = xend, yend = yend, group = gn)) + geom_segment() + facet_wrap(~f)")
w("group_text", ggplot(df, aes(x, y, xend = xend, yend = yend, group = g)) + geom_segment(), "ggplot(df, aes(x, y, xend = xend, yend = yend, group = g)) + geom_segment()")
w("colour_alpha", ggplot(df, full) + geom_segment(colour = "blue", alpha = 0.25), "ggplot(df, aes(x, y, xend = xend, yend = yend)) + geom_segment(colour = 'blue', alpha = 0.25)")
w("na", ggplot(read_case(here, "data_na.csv"), full) + geom_segment(), "ggplot(dna, aes(x, y, xend = xend, yend = yend)) + geom_segment()")
