# Shared helpers for the rectangle rows (PLOT-GG-001, paiml/aprender#6909): geom_tile, geom_rect,
# geom_raster. Sourced by each row's generate.R after ../oracle.R. R is needed ONLY to regenerate the
# committed fixtures; no test, build or CI job runs R. Run with LC_ALL=C.
#
# Every numeric column of the layer data is written as ggplot_build() has it (x, y, xmin, xmax, ymin,
# ymax, linewidth, linetype, alpha); fill and colour are written as RGBA triples, `null` where the
# column is NA (ggplot2's default tile/rect outline). The warnings ggplot_build() raised are recorded.
# Numeric CSV columns are read as DOUBLE: in R an integer column makes resolution() return 1
# whatever its spacing, and the Rust frame has no integer type.

read_case <- function(here, file) {
  df <- utils::read.csv(file.path(here, file), stringsAsFactors = FALSE, na.strings = "NA")
  df[] <- lapply(df, function(c) if (is.numeric(c)) as.numeric(c) else c)
  df
}

rgba_or_null <- function(v) {
  lapply(seq_along(v), function(i) {
    if (is.na(v[i])) return(NULL)
    I(as.integer(grDevices::col2rgb(v[i], alpha = TRUE)[, 1]))
  })
}

rects_layer_json <- function(ld, geom) {
  out <- list(geom = geom)
  skip <- c("PANEL", "group", "colour", "fill", "flipped_aes")
  for (col in setdiff(names(ld), skip)) out[[col]] <- I(as.numeric(ld[[col]]))
  out$panel <- I(as.integer(ld$PANEL))
  out$group <- I(as.integer(ld$group))
  out$fill_rgba <- rgba_or_null(ld$fill)
  if ("colour" %in% names(ld)) out$colour_rgba <- rgba_or_null(ld$colour)
  out
}

write_rects_fixture <- function(plot, row, case, spec, out_file) {
  ws <- character(0)
  b <- withCallingHandlers(
    ggplot2::ggplot_build(plot),
    warning = function(w) {
      ws <<- c(ws, strsplit(conditionMessage(w), "\n", fixed = TRUE)[[1]][1])
      invokeRestart("muffleWarning")
    }
  )
  pp <- b$layout$panel_params[[1]]
  geoms <- vapply(plot$layers, function(l) class(l$geom)[1], character(1))
  fixture <- list(
    row = row,
    case = case,
    spec = spec,
    collation = Sys.getlocale("LC_COLLATE"),
    versions = versions_json(),
    warnings = I(ws),
    layers = lapply(seq_along(b$data), function(i) rects_layer_json(b$data[[i]], geoms[i])),
    panel = list(x = axis_json(pp$x), y = axis_json(pp$y))
  )
  writeLines(to_json(fixture), out_file, useBytes = TRUE)
  invisible(fixture)
}
