# Shared helpers for the filled-ribbon rows (PLOT-GG-001, paiml/aprender#6909): geom_area and
# geom_ribbon. Sourced by each row's generate.R after ../oracle.R. R is needed ONLY to regenerate
# the committed fixtures; no test, build or CI job runs R. Run with LC_ALL=C.

read_case <- function(here, file) {
  utils::read.csv(file.path(here, file), stringsAsFactors = FALSE, na.strings = "NA")
}

# One layer of ggplot_build() output: the positional columns ggplot2 left, the panel and group
# ids, and `align_padding` (1/0, NA where stat_align did not run on the panel) when present.
# Fill, colour and the other style columns are not recorded: the engine does not resolve them.
filled_layer_json <- function(ld, geom) {
  out <- list(geom = geom)
  for (col in c("x", "y", "ymin", "ymax", "xmin", "xmax")) {
    if (col %in% names(ld)) out[[col]] <- I(as.numeric(ld[[col]]))
  }
  if ("align_padding" %in% names(ld)) out$align_padding <- I(as.integer(ld$align_padding))
  out$flipped_aes <- I(as.integer(ld$flipped_aes))
  out$panel <- I(as.integer(ld$PANEL))
  out$group <- I(as.integer(ld$group))
  out
}

write_filled_fixture <- function(plot, row, case, spec, out_file) {
  b <- ggplot2::ggplot_build(plot)
  pp <- b$layout$panel_params[[1]]
  geoms <- vapply(plot$layers, function(l) class(l$geom)[1], character(1))
  fixture <- list(
    row = row,
    case = case,
    spec = spec,
    collation = Sys.getlocale("LC_COLLATE"),
    versions = versions_json(),
    layers = lapply(seq_along(b$data), function(i) filled_layer_json(b$data[[i]], geoms[i])),
    panel = list(x = axis_json(pp$x), y = axis_json(pp$y))
  )
  writeLines(to_json(fixture), out_file, useBytes = TRUE)
  invisible(fixture)
}
