# Row: position_fill() on geom_col / geom_bar (PLOT-GG-001, paiml/aprender#6931).
# Regenerate (needs R + ggplot2 4.0.3; nothing else):
#   Rscript crates/aprender-viz/fixtures/ggplot2/position_fill/generate.R
# Writes expected.json beside this file. Commit both; tests never run R.
here <- (function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
})()
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "oracle_scales.R"))
suppressPackageStartupMessages(library(ggplot2))

rd <- function(f) utils::read.csv(file.path(here, f), stringsAsFactors = FALSE)
df <- rd("data.csv"); dc <- rd("data_count.csv"); dn <- rd("data_neg.csv")
du <- rd("data_unique.csv"); dz <- rd("data_zero.csv")

col <- function(d, ...) ggplot(d, aes(x, y, fill = g)) + geom_col(...)
cases <- list(
  list(name = "col_fill", data = "data.csv", stat = "identity",
       spec = "ggplot(df, aes(x, y, fill = g)) + geom_col(position = 'fill')",
       plot = col(df, position = "fill")),
  list(name = "col_fill_reverse", data = "data.csv", stat = "identity",
       spec = "ggplot(df, aes(x, y, fill = g)) + geom_col(position = position_fill(reverse = TRUE))",
       plot = col(df, position = position_fill(reverse = TRUE))),
  list(name = "col_fill_vjust_half", data = "data.csv", stat = "identity",
       spec = "ggplot(df, aes(x, y, fill = g)) + geom_col(position = position_fill(vjust = 0.5))",
       plot = col(df, position = position_fill(vjust = 0.5))),
  list(name = "col_fill_vjust_zero_reverse", data = "data.csv", stat = "identity",
       spec = "ggplot(df, aes(x, y, fill = g)) + geom_col(position = position_fill(vjust = 0, reverse = TRUE))",
       plot = col(df, position = position_fill(vjust = 0, reverse = TRUE))),
  list(name = "col_fill_width", data = "data.csv", stat = "identity",
       spec = "ggplot(df, aes(x, y, fill = g)) + geom_col(width = 0.5, position = 'fill')",
       plot = col(df, width = 0.5, position = "fill")),
  list(name = "bar_fill", data = "data_count.csv", stat = "count",
       spec = "ggplot(dc, aes(x, fill = g)) + geom_bar(position = 'fill')",
       plot = ggplot(dc, aes(x, fill = g)) + geom_bar(position = "fill")),
  list(name = "col_fill_negative", data = "data_neg.csv", stat = "identity",
       spec = "ggplot(dn, aes(x, y, fill = g)) + geom_col(position = 'fill')",
       plot = col(dn, position = "fill")),
  list(name = "col_fill_unique_x", data = "data_unique.csv", stat = "identity",
       spec = "ggplot(du, aes(x, y, fill = g)) + geom_col(position = 'fill')",
       plot = col(du, position = "fill")),
  list(name = "col_fill_zero_total", data = "data_zero.csv", stat = "identity",
       spec = "ggplot(dz, aes(x, y, fill = g)) + geom_col(position = 'fill')",
       plot = col(dz, position = "fill"))
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
fixture <- list(row = "position_fill", versions = versions_json(), cases = out)
writeLines(to_json(fixture), file.path(here, "expected.json"), useBytes = TRUE)
