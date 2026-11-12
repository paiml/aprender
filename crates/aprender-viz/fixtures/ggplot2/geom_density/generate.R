# Row: geom_density / stat_density (PLOT-GG-001, paiml/aprender#6909). Regenerate (R + ggplot2 4.0.3, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_density/generate.R
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "dist_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

df <- read_case(here, "data.csv")
dna <- read_case(here, "data_na.csv")
row <- "geom_density"
w <- function(case, plot, spec) {
  write_dist_fixture(plot, row, case, spec, file.path(here, paste0("expected_", case, ".json")))
}

w("default", ggplot(df, aes(x)) + geom_density(), "ggplot(df, aes(x)) + geom_density()")
w("bw_number", ggplot(df, aes(x)) + geom_density(bw = 0.75), "ggplot(df, aes(x)) + geom_density(bw = 0.75)")
w("adjust2", ggplot(df, aes(x)) + geom_density(adjust = 2), "ggplot(df, aes(x)) + geom_density(adjust = 2)")
w("n64", ggplot(df, aes(x)) + geom_density(n = 64), "ggplot(df, aes(x)) + geom_density(n = 64)")
w("fill_two_groups", ggplot(df, aes(x, fill = g)) + geom_density(alpha = 0.5),
  "ggplot(df, aes(x, fill = g)) + geom_density(alpha = 0.5)")
w("na_x", ggplot(dna, aes(x)) + geom_density(), "ggplot(dna, aes(x)) + geom_density()")
w("trim", ggplot(df, aes(x)) + geom_density(trim = TRUE), "ggplot(df, aes(x)) + geom_density(trim = TRUE)")
