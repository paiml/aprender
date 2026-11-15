# Row: geom_density (PLOT-GG-001, paiml/aprender#6909): stat_density, bw.nrd0, n = 512, gaussian.
# Regenerate (R + ggplot2 4.0.3 only; LC_COLLATE is pinned by oracle.R):
#   Rscript crates/aprender-viz/fixtures/ggplot2/geom_density/generate.R
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
source(file.path(here, "..", "oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

row <- "geom_density"
for (case in list(
  list("default", "data.csv"),
  list("three", "data_three.csv"),
  list("outlier", "data_outlier.csv"),
  list("negative", "data_negative.csv")
)) {
  df <- utils::read.csv(file.path(here, case[[2]]))
  spec <- sprintf("ggplot(df, aes(x)) + geom_density()  # df = %s", case[[2]])
  write_fixture(ggplot(df, aes(x)) + geom_density(), row, spec, case[[2]],
    file.path(here, paste0("expected_", case[[1]], ".json")))
}
