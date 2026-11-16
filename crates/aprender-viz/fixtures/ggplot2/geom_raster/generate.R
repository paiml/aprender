# Row: geom_raster (PLOT-GG-001, paiml/aprender#6909). GeomRaster$setup_data sizes a cell by the
# spacing of the distinct x (and y) values and places it with hjust/vjust: xmin = x - w * (1 - hjust),
# xmax = x + w * hjust. One distinct value gives a unit cell; an uneven grid warns and uses the
# smallest gap. The scales train on the bounds too.
# Regenerate (R + ggplot2 4.0.3 only, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_raster/generate.R
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "rects_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

d <- read_case(here, "data.csv")
dsr <- read_case(here, "data_single_row.csv")
dsc <- read_case(here, "data_single_column.csv")
dcell <- read_case(here, "data_single_cell.csv")
dux <- read_case(here, "data_uneven_x.csv")
duy <- read_case(here, "data_uneven_y.csv")
dh <- read_case(here, "data_half.csv")
dn <- read_case(here, "data_negative.csv")
row <- "geom_raster"
out <- function(case) file.path(here, paste0("expected_", case, ".json"))
w <- function(case, plot, spec) write_rects_fixture(plot, row, case, spec, out(case))

w("regular", ggplot(d, aes(x, y)) + geom_raster(), "ggplot(d, aes(x, y)) + geom_raster()")
w("hjust0_vjust0", ggplot(d, aes(x, y)) + geom_raster(hjust = 0, vjust = 0),
  "ggplot(d, aes(x, y)) + geom_raster(hjust = 0, vjust = 0)")
w("hjust1_vjust1", ggplot(d, aes(x, y)) + geom_raster(hjust = 1, vjust = 1),
  "ggplot(d, aes(x, y)) + geom_raster(hjust = 1, vjust = 1)")
w("hjust_quarter", ggplot(d, aes(x, y)) + geom_raster(hjust = 0.25, vjust = 0.75),
  "ggplot(d, aes(x, y)) + geom_raster(hjust = 0.25, vjust = 0.75)")
w("single_row", ggplot(dsr, aes(x, y)) + geom_raster(), "ggplot(dsr, aes(x, y)) + geom_raster()")
w("single_column", ggplot(dsc, aes(x, y)) + geom_raster(), "ggplot(dsc, aes(x, y)) + geom_raster()")
w("single_cell", ggplot(dcell, aes(x, y)) + geom_raster(), "ggplot(dcell, aes(x, y)) + geom_raster()")
w("uneven_x", ggplot(dux, aes(x, y)) + geom_raster(), "ggplot(dux, aes(x, y)) + geom_raster()")
w("uneven_y", ggplot(duy, aes(x, y)) + geom_raster(), "ggplot(duy, aes(x, y)) + geom_raster()")
w("half_steps", ggplot(dh, aes(x, y)) + geom_raster(), "ggplot(dh, aes(x, y)) + geom_raster()")
w("negative_unsorted", ggplot(dn, aes(x, y)) + geom_raster(hjust = 0, vjust = 1),
  "ggplot(dn, aes(x, y)) + geom_raster(hjust = 0, vjust = 1)")
w("fill_alpha", ggplot(d, aes(x, y)) + geom_raster(fill = "red", alpha = 0.5),
  "ggplot(d, aes(x, y)) + geom_raster(fill = 'red', alpha = 0.5)")
