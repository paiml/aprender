# Row: geom_polygon (PLOT-GG-001, paiml/aprender#6909).
# Regenerate (R + ggplot2 4.0.3 only, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_polygon/generate.R
# Writes expected_<case>.json beside this file. Commit them; tests never run R.
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "g3_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))
row <- "geom_polygon"
out <- function(case) file.path(here, paste0("expected_", case, ".json"))
w <- function(case, plot, spec) g3_write(plot, row, case, spec, out(case))
df <- g3_read(here, "data.csv"); dna <- g3_read(here, "data_na.csv"); dl <- g3_read(here, "data_layer.csv")
dlo <- g3_read(here, "data_label_lo.csv"); dhi <- g3_read(here, "data_label_hi.csv"); dtwo <- g3_read(here, "data_two.csv"); dpt <- g3_read(here, "data_pt.csv")
w("plain", ggplot(df, aes(x, y)) + geom_polygon(), "ggplot(df, aes(x, y)) + geom_polygon()")
w("group_num", ggplot(df, aes(x, y, group = gn)) + geom_polygon(), "... aes(group = gn)")
w("group_text", ggplot(df, aes(x, y, group = g)) + geom_polygon(), "... aes(group = g)")
w("two_groups", ggplot(dtwo, aes(x, y, group = gn)) + geom_polygon(), "ggplot(dtwo, aes(x, y, group = gn)) + geom_polygon()")
w("na", ggplot(dna, aes(x, y)) + geom_polygon(), "ggplot(dna, aes(x, y)) + geom_polygon()")
w("facet", ggplot(df, aes(x, y)) + geom_polygon() + facet_wrap(~f), "... + facet_wrap(~f)")
w("colour_alpha", ggplot(df, aes(x, y)) + geom_polygon(colour = "red", alpha = 0.5), "geom_polygon(colour = 'red', alpha = 0.5)")
w("layer_data", ggplot(dpt, aes(x, y)) + geom_point() + geom_polygon(data = dl), "geom_point() + geom_polygon(data = dl)")
w("label_lo", ggplot(dlo, aes(x, y)) + geom_polygon(), "ggplot(dlo, aes(x, y)) + geom_polygon()")
w("label_hi", ggplot(dhi, aes(x, y)) + geom_polygon(), "ggplot(dhi, aes(x, y)) + geom_polygon()")
