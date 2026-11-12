# Row: geom_boxplot / stat_boxplot (PLOT-GG-001, paiml/aprender#6909). Regenerate (R + ggplot2 4.0.3, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_boxplot/generate.R
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "dist_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

df <- read_case(here, "data.csv")
dna <- read_case(here, "data_na.csv")
d2 <- read_case(here, "data_two.csv")
dxv <- read_case(here, "data_xvar.csv")
row <- "geom_boxplot"
w <- function(case, plot, spec) {
  write_dist_fixture(plot, row, case, spec, file.path(here, paste0("expected_", case, ".json")))
}

w("y_only", ggplot(df, aes(y = v)) + geom_boxplot(), "ggplot(df, aes(y = v)) + geom_boxplot()")
w("fill_two_groups", ggplot(df, aes(y = v, fill = g)) + geom_boxplot(),
  "ggplot(df, aes(y = v, fill = g)) + geom_boxplot()")
w("fill_dodge_small", ggplot(df, aes(y = v, fill = k)) + geom_boxplot(),
  "ggplot(df, aes(y = v, fill = k)) + geom_boxplot()")
w("x_continuous_group", ggplot(df, aes(x, v, group = g)) + geom_boxplot(),
  "ggplot(df, aes(x, v, group = g)) + geom_boxplot()")
w("coef_3", ggplot(df, aes(y = v)) + geom_boxplot(coef = 3), "ggplot(df, aes(y = v)) + geom_boxplot(coef = 3)")
w("coef_0", ggplot(df, aes(y = v)) + geom_boxplot(coef = 0), "ggplot(df, aes(y = v)) + geom_boxplot(coef = 0)")
w("width_param", ggplot(df, aes(y = v)) + geom_boxplot(width = 0.4), "ggplot(df, aes(y = v)) + geom_boxplot(width = 0.4)")
w("na_y", ggplot(dna, aes(y = v)) + geom_boxplot(), "ggplot(dna, aes(y = v)) + geom_boxplot()")
w("na_y_groups", ggplot(dna, aes(y = v, fill = g)) + geom_boxplot(),
  "ggplot(dna, aes(y = v, fill = g)) + geom_boxplot()")
w("two_rows", ggplot(d2, aes(y = v, fill = g)) + geom_boxplot(), "ggplot(d2, aes(y = v, fill = g)) + geom_boxplot()")
w("x_varies_in_group", ggplot(dxv, aes(x, v, group = g)) + geom_boxplot(), "ggplot(dxv, aes(x, v, group = g)) + geom_boxplot()")
