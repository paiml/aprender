# Shared helpers for the segment, polygon, smooth, violin and raster rows (PLOT-GG-001,
# paiml/aprender#6909). Sourced by each row's generate.R after ../oracle.R. R is needed ONLY to
# regenerate the committed fixtures; no test, build or CI job runs R. Run with LC_ALL=C.

# Read a case CSV. `NA` cells are missing values; text columns stay character. Every numeric
# column is made double: R's read.csv gives whole numbers an integer type, and stat_smooth builds
# a different x grid for an integer column than for a double one.
g3_read <- function(here, file) {
  d <- utils::read.csv(file.path(here, file), stringsAsFactors = FALSE, na.strings = "NA")
  for (n in names(d)) if (is.numeric(d[[n]])) d[[n]] <- as.numeric(d[[n]])
  d
}

# One resolved colour/fill column as 0..255 RGBA, or NULL when every cell is NA. ggplot_build
# gives every row the same value here (nothing is data-mapped), which is asserted.
g3_rgba <- function(v) {
  if (all(is.na(v))) return(NULL)
  stopifnot(length(unique(v)) == 1L)
  I(as.integer(grDevices::col2rgb(v[1], alpha = TRUE)[, 1]))
}

# One layer of ggplot_build() output: every numeric column as it is (NA -> null), PANEL and group
# as 1-based / ggplot2 ids, colour and fill as RGBA, and the scalar columns once.
g3_layer_json <- function(ld, geom) {
  out <- list(geom = geom)
  skip <- c("PANEL", "group", "colour", "fill", "flipped_aes", "linewidth", "linetype", "alpha", "weight")
  for (col in setdiff(names(ld), skip)) out[[col]] <- I(as.numeric(ld[[col]]))
  out$panel <- I(as.integer(ld$PANEL))
  out$group <- I(as.integer(ld$group))
  out$colour_rgba <- if ("colour" %in% names(ld)) g3_rgba(ld$colour) else NULL
  out$fill_rgba <- if ("fill" %in% names(ld)) g3_rgba(ld$fill) else NULL
  for (col in c("linewidth", "linetype", "alpha")) {
    if (col %in% names(ld)) {
      stopifnot(length(unique(ld[[col]])) == 1L)
      out[[col]] <- as.numeric(ld[[col]][1])
    }
  }
  out$n_rows <- nrow(ld)
  out
}

g3_geom_names <- function(plot) vapply(plot$layers, function(l) class(l$geom)[1], character(1))

# Build `plot`; write every layer plus panel 1's x/y scales (facets here use fixed scales, so
# panel 1 speaks for all). Warnings (NA removal, uneven raster) are expected and silenced.
g3_write <- function(plot, row, case, spec, out_file) {
  b <- suppressWarnings(suppressMessages(ggplot2::ggplot_build(plot)))
  pp <- b$layout$panel_params[[1]]
  geoms <- g3_geom_names(plot)
  fixture <- list(
    row = row,
    case = case,
    spec = spec,
    collation = Sys.getlocale("LC_COLLATE"),
    versions = versions_json(),
    layers = lapply(seq_along(b$data), function(i) g3_layer_json(b$data[[i]], geoms[i])),
    panel = list(x = axis_json(pp$x), y = axis_json(pp$y))
  )
  writeLines(to_json(fixture), out_file, useBytes = TRUE)
  invisible(fixture)
}

g3_here <- function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
}
