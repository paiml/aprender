# Shared helpers for the path and reference-line rows (PLOT-GG-001, paiml/aprender#6909):
# geom_line, geom_path, geom_step, geom_hline, geom_vline, geom_abline.
#
# Sourced by each of those rows' generate.R after ../oracle.R. R is needed ONLY to regenerate the
# committed fixtures; no test, build or CI job runs R. Run with LC_ALL=C: group ids and facet
# panel order sort text by C collation, and the fixtures record the collation they were made under.

# Read a case CSV. `NA` cells are missing values; text columns stay character.
read_case <- function(here, file) {
  utils::read.csv(file.path(here, file), stringsAsFactors = FALSE, na.strings = "NA")
}

# One layer of ggplot_build() output, with the panel and group ids ggplot2 assigned.
#   panel : the PANEL factor as 1-based integers (facet panel order)
#   group : the group id (-1 when no discrete variable or `group` aesthetic is mapped)
# Every other numeric column (x, y, yintercept, xintercept, slope, intercept, linewidth,
# linetype, alpha) is written as it is; NA becomes null.
lines_layer_json <- function(ld, geom) {
  out <- list(geom = geom)
  skip <- c("PANEL", "group", "colour", "flipped_aes")
  for (col in setdiff(names(ld), skip)) out[[col]] <- I(as.numeric(ld[[col]]))
  out$panel <- I(as.integer(ld$PANEL))
  out$group <- I(as.integer(ld$group))
  rgba <- grDevices::col2rgb(ld$colour, alpha = TRUE)
  out$colour_rgba <- lapply(seq_len(ncol(rgba)), function(i) I(as.integer(rgba[, i])))
  out
}

# The vertices geom_step() draws: GeomStep$draw_panel runs stairstep() on each group of a panel.
stairs_json <- function(ld, direction) {
  panels <- sort(unique(as.integer(ld$PANEL)))
  parts <- lapply(panels, function(p) {
    d <- ld[as.integer(ld$PANEL) == p, , drop = FALSE]
    ggplot2:::dapply(d, "group", ggplot2:::stairstep, direction = direction)
  })
  all <- do.call(rbind, parts)
  list(
    direction = direction,
    x = I(as.numeric(all$x)),
    y = I(as.numeric(all$y)),
    panel = I(as.integer(all$PANEL)),
    group = I(as.integer(all$group))
  )
}

# Build `plot`; write layers + panel 1's x/y scales (facets here use fixed scales, so panel 1
# speaks for all). `step_direction` adds the stair-step vertices of every layer.
write_lines_fixture <- function(plot, row, case, spec, out_file, step_direction = NULL) {
  b <- ggplot2::ggplot_build(plot)
  pp <- b$layout$panel_params[[1]]
  geoms <- vapply(plot$layers, function(l) class(l$geom)[1], character(1))
  layers <- lapply(seq_along(b$data), function(i) {
    j <- lines_layer_json(b$data[[i]], geoms[i])
    if (!is.null(step_direction) && geoms[i] == "GeomStep") {
      j$stairs <- stairs_json(b$data[[i]], step_direction)
    }
    j
  })
  fixture <- list(
    row = row,
    case = case,
    spec = spec,
    collation = Sys.getlocale("LC_COLLATE"),
    versions = versions_json(),
    layers = layers,
    panel = list(x = axis_json(pp$x), y = axis_json(pp$y))
  )
  writeLines(to_json(fixture), out_file, useBytes = TRUE)
  invisible(fixture)
}

# Standard preamble of every generate.R: locate this directory, load the helpers.
lines_here <- function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
}
