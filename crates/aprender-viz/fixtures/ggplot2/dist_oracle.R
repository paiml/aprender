# Shared helpers for the distribution rows (PLOT-GG-001, paiml/aprender#6909):
# geom_histogram, geom_freqpoly, geom_density, geom_boxplot, geom_errorbar.
#
# Sourced by each row's generate.R after ../oracle.R. R is needed ONLY to regenerate the committed
# fixtures; no test, build or CI job runs R. Run with LC_ALL=C: group ids sort text by C
# collation, and the fixtures record the collation they were made under.

# Read a case CSV. `NA` cells are missing values; text columns stay character.
read_case <- function(here, file) {
  utils::read.csv(file.path(here, file), stringsAsFactors = FALSE, na.strings = "NA")
}

# RGBA (0..255) of a colour as ggplot2 resolved it, or NULL for NA.
rgba_json <- function(col) {
  if (all(is.na(col))) return(NULL)
  m <- grDevices::col2rgb(col, alpha = TRUE)
  lapply(seq_len(ncol(m)), function(i) I(as.integer(m[, i])))
}

# One layer of ggplot_build() output. Every numeric column is written as it is (NA -> null);
# `outliers` (a list column) becomes an array of arrays; colour and fill become RGBA rows.
dist_layer_json <- function(ld, geom) {
  out <- list(geom = geom)
  skip <- c("PANEL", "group", "colour", "fill", "outliers", "flipped_aes")
  for (col in setdiff(names(ld), skip)) {
    v <- ld[[col]]
    if (is.numeric(v) || is.logical(v)) out[[col]] <- I(as.numeric(v))
  }
  out$panel <- I(as.integer(ld$PANEL))
  out$group <- I(as.integer(ld$group))
  if (!is.null(ld$outliers)) out$outliers <- lapply(ld$outliers, function(o) I(as.numeric(o)))
  if ("colour" %in% names(ld)) out$colour_rgba <- rgba_json(ld$colour)
  if ("fill" %in% names(ld)) out$fill_rgba <- rgba_json(ld$fill)
  out
}

# Build `plot`; write layers + panel 1's x/y scales.
write_dist_fixture <- function(plot, row, case, spec, out_file) {
  b <- suppressMessages(suppressWarnings(ggplot2::ggplot_build(plot)))
  pp <- b$layout$panel_params[[1]]
  geoms <- vapply(plot$layers, function(l) class(l$geom)[1], character(1))
  fixture <- list(
    row = row,
    case = case,
    spec = spec,
    collation = Sys.getlocale("LC_COLLATE"),
    versions = versions_json(),
    layers = lapply(seq_along(b$data), function(i) dist_layer_json(b$data[[i]], geoms[i])),
    panel = list(x = axis_json(pp$x), y = axis_json(pp$y))
  )
  writeLines(to_json(fixture), out_file, useBytes = TRUE)
  invisible(fixture)
}
