# Row: geom_violin (PLOT-GG-001, paiml/aprender#6909).
# Regenerate (R + ggplot2 4.0.3 only, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_violin/generate.R
# Writes expected_<case>.json beside this file. Commit them; tests never run R.
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "g3_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))
row <- "geom_violin"
out <- function(case) file.path(here, paste0("expected_", case, ".json"))
w <- function(case, plot, spec) g3_write(plot, row, case, spec, out(case))
df <- g3_read(here, "data.csv"); dsm <- g3_read(here, "data_small.csv"); dl <- g3_read(here, "data_layer.csv")
# geom_violin: only the *shape* of R's answer is recorded (layer rows, groups, rows per group,
# columns, panel y scale). aprender-viz refuses the row with a typed error; the fixtures prove the
# refused computation (stat_ydensity: 512 density points per group) is not a trivial one.
vw <- function(case, plot, spec) {
  b <- suppressWarnings(suppressMessages(ggplot2::ggplot_build(plot)))
  pp <- b$layout$panel_params[[1]]
  ld <- b$data[[length(b$data)]]
  per <- as.integer(table(ld$group)[order(as.integer(names(table(ld$group))))])
  fixture <- list(
    row = row, case = case, spec = spec, collation = Sys.getlocale("LC_COLLATE"),
    versions = versions_json(),
    layer = list(geom = class(plot$layers[[length(plot$layers)]]$geom)[1], n_rows = nrow(ld),
                 n_groups = length(unique(ld$group)), rows_per_group = I(per),
                 columns = I(names(ld)), has_density = "density" %in% names(ld)),
    panel = list(y = axis_json(pp$y))
  )
  writeLines(to_json(fixture), out(case), useBytes = TRUE)
}
vw("plain", ggplot(df[df$g == "a", ], aes(g, y)) + geom_violin(), "ggplot(df[g == 'a'], aes(g, y)) + geom_violin()")
vw("two_groups", ggplot(df[df$g != "c", ], aes(g, y)) + geom_violin(), "ggplot(df[g != 'c'], aes(g, y)) + geom_violin()")
vw("three_groups", ggplot(df, aes(g, y)) + geom_violin(), "ggplot(df, aes(g, y)) + geom_violin()")
vw("group_num", ggplot(df, aes(g, y, group = gn)) + geom_violin(), "ggplot(df, aes(g, y, group = gn)) + geom_violin()")
vw("facet", ggplot(df, aes(g, y)) + geom_violin() + facet_wrap(~f), "ggplot(df, aes(g, y)) + geom_violin() + facet_wrap(~f)")
vw("small", ggplot(dsm, aes(g, y)) + geom_violin(), "ggplot(dsm, aes(g, y)) + geom_violin()")
vw("layer_data", ggplot(dsm, aes(g, y)) + geom_point() + geom_violin(data = dl), "geom_point() + geom_violin(data = dl)")
vw("untrimmed", ggplot(df, aes(g, y)) + geom_violin(trim = FALSE), "ggplot(df, aes(g, y)) + geom_violin(trim = FALSE)")
