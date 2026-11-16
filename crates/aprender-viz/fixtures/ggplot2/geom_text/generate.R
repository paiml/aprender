# Row: geom_text (PLOT-GG-001, paiml/aprender#6909). The numeric layer data only: x and y (after
# position_nudge, which the scales are retrained on), nudge_x, nudge_y, angle, hjust, vjust, size,
# alpha, colour, group and panel, and the labels passed through. Font metrics, family, fontface and
# lineheight are not layout and are not recorded.
# Regenerate (R + ggplot2 4.0.3 only, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_text/generate.R
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "lines_oracle.R"))
source(file.path(here, "..", "pack_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

df <- read_case(here, "data.csv")
row <- "geom_text"
out <- function(case) file.path(here, paste0("expected_", case, ".json"))
w <- function(case, plot, spec) write_pack_fixture(plot, row, case, spec, out(case))
lab <- aes(x, y, label = lab)

w("plain", ggplot(df, lab) + geom_text(), "ggplot(df, aes(x, y, label = lab)) + geom_text()")
w("nudge", ggplot(df, lab) + geom_text(nudge_x = 0.5, nudge_y = -1), "ggplot(df, aes(x, y, label = lab)) + geom_text(nudge_x = 0.5, nudge_y = -1)")
w("nudge_x_only", ggplot(df, lab) + geom_text(nudge_x = 0.25), "ggplot(df, aes(x, y, label = lab)) + geom_text(nudge_x = 0.25)")
w("hjust_vjust_angle", ggplot(df, lab) + geom_text(hjust = 0, vjust = 1, angle = 45), "ggplot(df, aes(x, y, label = lab)) + geom_text(hjust = 0, vjust = 1, angle = 45)")
w("size_colour_alpha", ggplot(df, lab) + geom_text(size = 5, colour = "blue", alpha = 0.5), "ggplot(df, aes(x, y, label = lab)) + geom_text(size = 5, colour = 'blue', alpha = 0.5)")
w("facet", ggplot(df, lab) + geom_text(nudge_y = 0.5) + facet_wrap(~f), "ggplot(df, aes(x, y, label = lab)) + geom_text(nudge_y = 0.5) + facet_wrap(~f)")
w("group_num", ggplot(df, aes(x, y, label = lab, group = gn)) + geom_text(), "ggplot(df, aes(x, y, label = lab, group = gn)) + geom_text()")
w("group_text", ggplot(df, aes(x, y, label = lab, group = g)) + geom_text(), "ggplot(df, aes(x, y, label = lab, group = g)) + geom_text()")
w("negative", ggplot(read_case(here, "data_negative.csv"), lab) + geom_text(nudge_x = -0.5, nudge_y = 0.25), "ggplot(dn, aes(x, y, label = lab)) + geom_text(nudge_x = -0.5, nudge_y = 0.25)")
w("na", ggplot(read_case(here, "data_na.csv"), lab) + geom_text(), "ggplot(dna, aes(x, y, label = lab)) + geom_text()")
