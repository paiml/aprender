# Row: geom_raster (PLOT-GG-001, paiml/aprender#6909).
# Regenerate (R + ggplot2 4.0.3 only, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_raster/generate.R
# Writes expected_<case>.json beside this file. Commit them; tests never run R.
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "g3_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))
row <- "geom_raster"
out <- function(case) file.path(here, paste0("expected_", case, ".json"))
w <- function(case, plot, spec) g3_write(plot, row, case, spec, out(case))
df <- g3_read(here, "data.csv"); dna <- g3_read(here, "data_na.csv"); dl <- g3_read(here, "data_layer.csv")
dfr <- g3_read(here, "data_frac.csv"); dsi <- g3_read(here, "data_single.csv"); dneg <- g3_read(here, "data_neg.csv")
dun <- g3_read(here, "data_uneven.csv"); dpt <- g3_read(here, "data_pt.csv")
w("plain", ggplot(df, aes(x, y)) + geom_raster(), "ggplot(df, aes(x, y)) + geom_raster()")
w("group_num", ggplot(df, aes(x, y, group = gn)) + geom_raster(), "... aes(group = gn)")
w("na", ggplot(dna, aes(x, y)) + geom_raster(), "ggplot(dna, aes(x, y)) + geom_raster()")
w("facet", ggplot(df, aes(x, y)) + geom_raster() + facet_wrap(~f), "... + facet_wrap(~f)")
w("colour_alpha", ggplot(df, aes(x, y)) + geom_raster(alpha = 0.5), "geom_raster(alpha = 0.5)")
w("layer_data", ggplot(dpt, aes(x, y)) + geom_point() + geom_raster(data = dl), "geom_point() + geom_raster(data = dl)")
w("fractional", ggplot(dfr, aes(x, y)) + geom_raster(), "ggplot(dfr, aes(x, y)) + geom_raster()")
w("single", ggplot(dsi, aes(x, y)) + geom_raster(), "ggplot(dsi, aes(x, y)) + geom_raster()")
w("negative", ggplot(dneg, aes(x, y)) + geom_raster(), "ggplot(dneg, aes(x, y)) + geom_raster()")
w("uneven", ggplot(dun, aes(x, y)) + geom_raster(), "ggplot(dun, aes(x, y)) + geom_raster()  # R warns and shifts; refused here")
