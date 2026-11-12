# Row: geom_segment (PLOT-GG-001, paiml/aprender#6909).
# Regenerate (R + ggplot2 4.0.3 only, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_segment/generate.R
# Writes expected_<case>.json beside this file. Commit them; tests never run R.
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "g3_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))
row <- "geom_segment"
out <- function(case) file.path(here, paste0("expected_", case, ".json"))
w <- function(case, plot, spec) g3_write(plot, row, case, spec, out(case))
df <- g3_read(here, "data.csv"); dna <- g3_read(here, "data_na.csv"); dl <- g3_read(here, "data_layer.csv")
dlo <- g3_read(here, "data_label_lo.csv"); dhi <- g3_read(here, "data_label_hi.csv"); dpt <- g3_read(here, "data_pt.csv")
m <- function(d) aes(x, y, xend = xe, yend = ye)
w("plain", ggplot(df, aes(x, y, xend = xe, yend = ye)) + geom_segment(), "ggplot(df, aes(x, y, xend = xe, yend = ye)) + geom_segment()")
w("group_num", ggplot(df, aes(x, y, xend = xe, yend = ye, group = gn)) + geom_segment(), "... + aes(group = gn)")
w("group_text", ggplot(df, aes(x, y, xend = xe, yend = ye, group = g)) + geom_segment(), "... + aes(group = g)")
w("na", ggplot(dna, aes(x, y, xend = xe, yend = ye)) + geom_segment(), "ggplot(dna, ...) + geom_segment()")
w("facet", ggplot(df, aes(x, y, xend = xe, yend = ye)) + geom_segment() + facet_wrap(~f), "... + facet_wrap(~f)")
w("colour_alpha", ggplot(df, aes(x, y, xend = xe, yend = ye)) + geom_segment(colour = "red", alpha = 0.5), "geom_segment(colour = 'red', alpha = 0.5)")
w("layer_data", ggplot(dpt, aes(x, y)) + geom_point() + geom_segment(data = dl, aes(x, y, xend = xe, yend = ye)), "geom_point() + geom_segment(data = dl)")
w("label_lo", ggplot(dlo, aes(x, y, xend = xe, yend = ye)) + geom_segment(), "ggplot(dlo, ...) + geom_segment()")
w("label_hi", ggplot(dhi, aes(x, y, xend = xe, yend = ye)) + geom_segment(), "ggplot(dhi, ...) + geom_segment()")
