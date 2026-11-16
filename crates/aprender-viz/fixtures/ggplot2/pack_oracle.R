# Shared helpers for the segment, polygon and text rows (PLOT-GG-001, paiml/aprender#6909):
# geom_segment, geom_polygon, geom_text. Sourced by each row's generate.R after ../oracle.R and
# ../lines_oracle.R. R is needed ONLY to regenerate the committed fixtures; no test, build or CI job
# runs R. Run with LC_ALL=C (group ids and facet panels sort text by C collation).

# One layer of ggplot_build() output. Numeric columns are written as numbers (NA is null), the
# character `label` column as strings, colours (and a polygon's fill) as 0..255 RGBA. Font columns
# (family, fontface, lineheight) are not numeric layout and are left out.
pack_layer_json <- function(ld, geom) {
  out <- list(geom = geom)
  skip <- c("PANEL", "group", "colour", "fill", "family", "fontface", "lineheight", "flipped_aes")
  for (col in setdiff(names(ld), skip)) {
    v <- ld[[col]]
    out[[col]] <- if (is.character(v)) I(as.character(v)) else I(as.numeric(v))
  }
  out$panel <- I(as.integer(ld$PANEL))
  out$group <- I(as.integer(ld$group))
  rgba <- function(col) {
    m <- grDevices::col2rgb(col, alpha = TRUE)
    lapply(seq_len(ncol(m)), function(i) I(as.integer(m[, i])))
  }
  out$colour_rgba <- rgba(ld$colour)
  if ("fill" %in% names(ld)) out$fill_rgba <- rgba(ld$fill)
  out
}

# Build `plot`; write every layer and panel 1's x/y scales (facets here use fixed scales).
write_pack_fixture <- function(plot, row, case, spec, out_file) {
  b <- ggplot2::ggplot_build(plot)
  pp <- b$layout$panel_params[[1]]
  geoms <- vapply(plot$layers, function(l) class(l$geom)[1], character(1))
  fixture <- list(
    row = row,
    case = case,
    spec = spec,
    collation = Sys.getlocale("LC_COLLATE"),
    versions = versions_json(),
    layers = lapply(seq_along(b$data), function(i) pack_layer_json(b$data[[i]], geoms[i])),
    panel = list(x = axis_json(pp$x), y = axis_json(pp$y))
  )
  writeLines(to_json(fixture), out_file, useBytes = TRUE)
  invisible(fixture)
}
