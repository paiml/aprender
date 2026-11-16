# Row: position_dodge() on geom_col / geom_bar (PLOT-GG-001, paiml/aprender#6930).
# Regenerate (needs R + ggplot2 4.0.3; nothing else):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/position_dodge/generate.R
# Writes expected.json beside this file. Commit both; tests never run R.
here <- (function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
})()
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "oracle_scales.R"))
suppressPackageStartupMessages(library(ggplot2))

rd <- function(f) utils::read.csv(file.path(here, f), stringsAsFactors = FALSE)
df <- rd("data.csv"); de <- rd("data_equal.csv"); dn <- rd("data_neg.csv")
dc <- rd("data_count.csv"); d1 <- rd("data_one.csv"); dfc <- rd("data_facet.csv")

col <- function(d, ...) ggplot(d, aes(x, y, fill = g)) + geom_col(...)
pd <- position_dodge
cases <- list(
  list(name = "col_dodge_total_unequal", data = "data.csv", stat = "identity",
       spec = "ggplot(df, aes(x, y, fill = g)) + geom_col(position = 'dodge')",
       plot = col(df, position = "dodge")),
  list(name = "col_dodge_equal", data = "data_equal.csv", stat = "identity",
       spec = "ggplot(de, aes(x, y, fill = g)) + geom_col(position = 'dodge')",
       plot = col(de, position = "dodge")),
  list(name = "col_dodge_width_half", data = "data_equal.csv", stat = "identity",
       spec = "ggplot(de, aes(x, y, fill = g)) + geom_col(position = position_dodge(width = 0.5))",
       plot = col(de, position = pd(width = 0.5))),
  list(name = "col_dodge_width_one", data = "data_equal.csv", stat = "identity",
       spec = "ggplot(de, aes(x, y, fill = g)) + geom_col(position = position_dodge(width = 1))",
       plot = col(de, position = pd(width = 1))),
  list(name = "col_dodge_single", data = "data.csv", stat = "identity",
       spec = "ggplot(df, aes(x, y, fill = g)) + geom_col(position = position_dodge(preserve = 'single'))",
       plot = col(df, position = pd(preserve = "single"))),
  list(name = "col_dodge_reverse", data = "data.csv", stat = "identity",
       spec = "ggplot(df, aes(x, y, fill = g)) + geom_col(position = position_dodge(reverse = TRUE))",
       plot = col(df, position = pd(reverse = TRUE))),
  list(name = "col_dodge_single_reverse", data = "data.csv", stat = "identity",
       spec = "ggplot(df, aes(x, y, fill = g)) + geom_col(position = position_dodge(preserve = 'single', reverse = TRUE))",
       plot = col(df, position = pd(preserve = "single", reverse = TRUE))),
  list(name = "col_dodge_negative", data = "data_neg.csv", stat = "identity",
       spec = "ggplot(dn, aes(x, y, fill = g)) + geom_col(position = 'dodge')",
       plot = col(dn, position = "dodge")),
  list(name = "col_dodge_narrow_bars", data = "data_equal.csv", stat = "identity",
       spec = "ggplot(de, aes(x, y, fill = g)) + geom_col(width = 0.5, position = 'dodge')",
       plot = col(de, width = 0.5, position = "dodge")),
  list(name = "col_dodge_no_fill", data = "data_one.csv", stat = "identity",
       spec = "ggplot(d1, aes(x, y)) + geom_col(position = 'dodge')",
       plot = ggplot(d1, aes(x, y)) + geom_col(position = "dodge")),
  list(name = "bar_dodge", data = "data_count.csv", stat = "count",
       spec = "ggplot(dc, aes(x, fill = g)) + geom_bar(position = 'dodge')",
       plot = ggplot(dc, aes(x, fill = g)) + geom_bar(position = "dodge")),
  list(name = "bar_dodge_single", data = "data_count.csv", stat = "count",
       spec = "ggplot(dc, aes(x, fill = g)) + geom_bar(position = position_dodge(preserve = 'single'))",
       plot = ggplot(dc, aes(x, fill = g)) + geom_bar(position = pd(preserve = "single"))),
  list(name = "col_dodge_facet_total", data = "data_facet.csv", stat = "identity",
       spec = "ggplot(dfc, aes(x, y, fill = g)) + geom_col(position = 'dodge') + facet_wrap(~f)",
       plot = col(dfc, position = "dodge") + facet_wrap(~f)),
  list(name = "col_dodge_facet_single", data = "data_facet.csv", stat = "identity",
       spec = "ggplot(dfc, aes(x, y, fill = g)) + geom_col(position = position_dodge(preserve = 'single')) + facet_wrap(~f)",
       plot = col(dfc, position = pd(preserve = "single")) + facet_wrap(~f))
)

num <- function(ld, nm) if (nm %in% names(ld)) I(as.numeric(ld[[nm]])) else NULL
out <- lapply(cases, function(cs) {
  b <- ggplot2::ggplot_build(cs$plot)
  pp <- b$layout$panel_params[[1]]
  ld <- b$data[[1]]
  layer <- list(geom = class(cs$plot$layers[[1]]$geom)[1])
  for (nm in c("x", "y", "ymin", "ymax", "xmin", "xmax", "width", "count", "prop")) {
    v <- num(ld, nm)
    if (!is.null(v)) layer[[nm]] <- v
  }
  layer$group <- I(as.integer(ld$group))
  layer$panel <- I(as.integer(ld$PANEL))
  list(name = cs$name, spec = cs$spec, data_file = cs$data, stat = cs$stat,
       layer = layer,
       panel = list(x = axis_scale_json(pp$x), y = axis_scale_json(pp$y)))
})
fixture <- list(row = "position_dodge", versions = versions_json(), cases = out)
writeLines(to_json(fixture), file.path(here, "expected.json"), useBytes = TRUE)
