# Row: geom_violin (PLOT-GG-001, paiml/aprender#6909): stat_ydensity, GeomViolin setup_data and
# position_dodge, ggplot2 4.0.3 under R 4.6.2. Data: make_data.R (committed CSVs).
# Regenerate (R + ggplot2 4.0.3 only; LC_COLLATE is pinned by oracle.R):
#   Rscript crates/aprender-viz/fixtures/ggplot2/geom_violin/generate.R
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
source(file.path(here, "..", "oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

row <- "geom_violin"

# The x position scale as ggplot2 resolved it. A discrete axis records its levels (`limits`), the
# positions of its breaks (the `pos` attribute of get_breaks()) and their labels; a continuous
# one uses the shared axis_json() fields.
violin_axis <- function(view) {
  if (is.character(view$limits)) {
    br <- view$get_breaks()
    return(list(
      kind = "discrete",
      limits = I(view$limits),
      continuous_range = I(as.numeric(view$continuous_range)),
      breaks = I(as.numeric(attr(br, "pos"))),
      labels = I(as.character(view$get_labels())),
      n_breaks_censored = 0L
    ))
  }
  c(list(kind = "continuous"), axis_json(view))
}

write_violin <- function(plot, spec, data_file, out_file) {
  b <- suppressWarnings(ggplot2::ggplot_build(plot))
  pp <- b$layout$panel_params[[1]]
  fixture <- list(
    row = row,
    spec = spec,
    data_file = data_file,
    versions = versions_json(),
    layers = list(layer_json(b$data[[1]], class(plot$layers[[1]]$geom)[1])),
    panel = list(x = violin_axis(pp$x), y = axis_json(pp$y))
  )
  writeLines(to_json(fixture), out_file, useBytes = TRUE)
}

read <- function(file) {
  df <- utils::read.csv(file.path(here, file), stringsAsFactors = FALSE)
  # resolution() of an integer x is 1 in ggplot2; aprender-viz holds every number as a double.
  for (col in names(df)) if (is.integer(df[[col]])) df[[col]] <- as.numeric(df[[col]])
  df
}

cases <- list(
  list("single", "data_single.csv", "aes(g, y)", aes(g, y), geom_violin()),
  list("three", "data_three.csv", "aes(g, y)", aes(g, y), geom_violin()),
  list("count", "data_three.csv", "aes(g, y)", aes(g, y), geom_violin(scale = "count")),
  list("width", "data_three.csv", "aes(g, y)", aes(g, y), geom_violin(scale = "width")),
  list("trim_false", "data_three.csv", "aes(g, y)", aes(g, y), geom_violin(trim = FALSE)),
  list("trim_adjust", "data_three.csv", "aes(g, y)", aes(g, y), geom_violin(trim = FALSE, adjust = 0.5)),
  list("adjust_half", "data_three.csv", "aes(g, y)", aes(g, y), geom_violin(adjust = 0.5)),
  list("width_half", "data_three.csv", "aes(g, y)", aes(g, y), geom_violin(width = 0.5)),
  list("width_wide", "data_three.csv", "aes(g, y)", aes(g, y), geom_violin(width = 2)),
  list("fill", "data_three.csv", "aes(g, y, fill = f)", aes(g, y, fill = f), geom_violin()),
  list("dodge3", "data_dodge3.csv", "aes(g, y, fill = f)", aes(g, y, fill = f), geom_violin()),
  list("numeric_x", "data_three.csv", "aes(xn, y, group = g)", aes(xn, y, group = g), geom_violin()),
  list("na", "data_na.csv", "aes(g, y)", aes(g, y), geom_violin()),
  list("tiny", "data_tiny.csv", "aes(g, y)", aes(g, y), geom_violin()),
  list("constant", "data_constant.csv", "aes(g, y)", aes(g, y), geom_violin()),
  list("constant_only", "data_constant_only.csv", "aes(g, y)", aes(g, y), geom_violin())
)
for (case in cases) {
  df <- read(case[[2]])
  layer <- case[[5]]
  call <- paste(deparse(layer$constructor %||% quote(geom_violin())), collapse = "")
  spec <- sprintf("ggplot(df, %s) + %s  # df = %s", case[[3]], call, case[[2]])
  write_violin(ggplot(df, case[[4]]) + layer, spec, case[[2]],
    file.path(here, paste0("expected_", case[[1]], ".json")))
}
