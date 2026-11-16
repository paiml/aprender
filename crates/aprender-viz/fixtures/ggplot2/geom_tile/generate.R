# Row: geom_tile (PLOT-GG-001, paiml/aprender#6909; ggplot2 core row 14). GeomTile$setup_data gives
# xmin/xmax = x -+ width/2 and ymin/ymax = y -+ height/2, where width and height default to the
# data resolution (per panel) and the scales train on the bounds too.
# Regenerate (R + ggplot2 4.0.3 only, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_tile/generate.R
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "rects_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

d <- read_case(here, "data.csv")
du <- read_case(here, "data_uneven.csv")
ds <- read_case(here, "data_single.csv")
dn <- read_case(here, "data_negative.csv")
dq <- read_case(here, "data_quarter.csv")
dna <- read_case(here, "data_na.csv")
row <- "geom_tile"
out <- function(case) file.path(here, paste0("expected_", case, ".json"))
w <- function(case, plot, spec) write_rects_fixture(plot, row, case, spec, out(case))

w("default", ggplot(d, aes(x, y)) + geom_tile(), "ggplot(d, aes(x, y)) + geom_tile()")
w("width_height", ggplot(d, aes(x, y)) + geom_tile(width = 0.5, height = 0.25),
  "ggplot(d, aes(x, y)) + geom_tile(width = 0.5, height = 0.25)")
w("width_only", ggplot(d, aes(x, y)) + geom_tile(width = 0.5), "ggplot(d, aes(x, y)) + geom_tile(width = 0.5)")
w("height_only", ggplot(d, aes(x, y)) + geom_tile(height = 3), "ggplot(d, aes(x, y)) + geom_tile(height = 3)")
w("uneven", ggplot(du, aes(x, y)) + geom_tile(), "ggplot(du, aes(x, y)) + geom_tile()")
w("single", ggplot(ds, aes(x, y)) + geom_tile(), "ggplot(ds, aes(x, y)) + geom_tile()")
w("single_sized", ggplot(ds, aes(x, y)) + geom_tile(width = 4, height = 0.5),
  "ggplot(ds, aes(x, y)) + geom_tile(width = 4, height = 0.5)")
w("negative", ggplot(dn, aes(x, y)) + geom_tile(), "ggplot(dn, aes(x, y)) + geom_tile()")
w("quarter", ggplot(dq, aes(x, y)) + geom_tile(), "ggplot(dq, aes(x, y)) + geom_tile()")
w("fill_colour_alpha", ggplot(d, aes(x, y)) + geom_tile(fill = "red", colour = "blue", alpha = 0.5),
  "ggplot(d, aes(x, y)) + geom_tile(fill = 'red', colour = 'blue', alpha = 0.5)")
w("na_rows", ggplot(dna, aes(x, y)) + geom_tile(), "ggplot(dna, aes(x, y)) + geom_tile()")
w("with_point", ggplot(du, aes(x, y)) + geom_tile() + geom_point(data = ds),
  "ggplot(du, aes(x, y)) + geom_tile() + geom_point(data = ds)")
