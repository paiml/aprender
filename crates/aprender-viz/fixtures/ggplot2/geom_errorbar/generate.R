# Row: geom_errorbar (PLOT-GG-001, paiml/aprender#6909). Regenerate (R + ggplot2 4.0.3, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_errorbar/generate.R
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "dist_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

df <- read_case(here, "data.csv")
dna <- read_case(here, "data_na.csv")
d1 <- read_case(here, "data_one.csv")
dt <- read_case(here, "data_ties.csv")
row <- "geom_errorbar"
w <- function(case, plot, spec) {
  write_dist_fixture(plot, row, case, spec, file.path(here, paste0("expected_", case, ".json")))
}

w("default", ggplot(df, aes(x, ymin = lo, ymax = hi)) + geom_errorbar(),
  "ggplot(df, aes(x, ymin = lo, ymax = hi)) + geom_errorbar()")
w("width_param", ggplot(df, aes(x, ymin = lo, ymax = hi)) + geom_errorbar(width = 0.3),
  "ggplot(df, aes(x, ymin = lo, ymax = hi)) + geom_errorbar(width = 0.3)")
w("colour_groups", ggplot(df, aes(x, ymin = lo, ymax = hi, colour = g)) + geom_errorbar(),
  "ggplot(df, aes(x, ymin = lo, ymax = hi, colour = g)) + geom_errorbar()")
w("linewidth_colour", ggplot(df, aes(x, ymin = lo, ymax = hi)) + geom_errorbar(linewidth = 1, colour = "red"),
  "ggplot(df, aes(x, ymin = lo, ymax = hi)) + geom_errorbar(linewidth = 1, colour = 'red')")
w("na_bounds", ggplot(dna, aes(x, ymin = lo, ymax = hi)) + geom_errorbar(),
  "ggplot(dna, aes(x, ymin = lo, ymax = hi)) + geom_errorbar()")
w("one_row", ggplot(d1, aes(x, ymin = lo, ymax = hi)) + geom_errorbar(),
  "ggplot(d1, aes(x, ymin = lo, ymax = hi)) + geom_errorbar()")
w("ties_groups", ggplot(dt, aes(x, ymin = lo, ymax = hi, colour = g)) + geom_errorbar(),
  "ggplot(dt, aes(x, ymin = lo, ymax = hi, colour = g)) + geom_errorbar()")
w("inverted", ggplot(dt, aes(x, ymin = hi, ymax = lo)) + geom_errorbar(),
  "ggplot(dt, aes(x, ymin = hi, ymax = lo)) + geom_errorbar()")

# #6976 review F1/F8: the default width comes from the LAYER's own x, not the plot's; x close to 0.
dn0 <- read_case(here, "data_near0.csv")
db <- read_case(here, "data_b.csv")
dh <- read_case(file.path(here, "..", "geom_histogram"), "data.csv")
w("near_zero", ggplot(dn0, aes(x, ymin = lo, ymax = hi)) + geom_errorbar(),
  "ggplot(dn0, aes(x, ymin = lo, ymax = hi)) + geom_errorbar()")
w("two_layers",
  ggplot() + geom_errorbar(data = df, aes(x, ymin = lo, ymax = hi)) + geom_errorbar(data = db, aes(x, ymin = lo, ymax = hi)),
  "ggplot() + geom_errorbar(data = df, aes(x, ymin = lo, ymax = hi)) + geom_errorbar(data = db, aes(x, ymin = lo, ymax = hi))")
w("hist_plus_errorbar",
  ggplot() + geom_histogram(data = dh, aes(x)) + geom_errorbar(data = db, aes(x, ymin = lo, ymax = hi)),
  "ggplot() + geom_histogram(data = dh, aes(x)) + geom_errorbar(data = db, aes(x, ymin = lo, ymax = hi))")
