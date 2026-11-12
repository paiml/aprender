# Shared helpers for the filled-mark and text rows (PLOT-GG-001, paiml/aprender#6909):
# geom_area, geom_ribbon, geom_tile, geom_rect, geom_text.
#
# Sourced by each of those rows' generate.R after ../oracle.R. R is needed ONLY to regenerate the
# committed fixtures; no test, build or CI job runs R. Run with LC_ALL=C: group ids sort text by C
# collation, and every fixture records the collation it was made under.

# Read a case CSV. `NA` cells are missing values; text columns stay character.
#
# read.csv returns whole-number columns as integer, and ggplot2 treats integer positions
# specially (resolution() returns 1). aprender-viz stores every number as f32, so the oracle
# sees doubles too: the fixture then measures the same arithmetic on both sides.
read_case <- function(here, file) {
  df <- utils::read.csv(file.path(here, file), stringsAsFactors = FALSE, na.strings = "NA")
  for (col in names(df)) if (is.integer(df[[col]])) df[[col]] <- as.numeric(df[[col]])
  df
}

rgba_json <- function(col) {
  rgba <- grDevices::col2rgb(col, alpha = TRUE)
  lapply(seq_len(ncol(rgba)), function(i) {
    if (is.na(col[i])) NULL else I(as.integer(rgba[, i]))
  })
}

# One layer of ggplot_build() output. Every column ggplot2 emitted is written, so the Rust side
# can assert which columns exist (area writes xmin/xmax only after stacking; a lone group has no
# align_padding), not only what they hold.
#   numeric columns -> numbers (NA -> null); logical -> booleans; character -> strings;
#   colour / fill -> RGBA 0..255 (NA -> null); group -> integer ids; PANEL -> integer.
marks_layer_json <- function(ld, geom) {
  out <- list(geom = geom, columns = I(setdiff(names(ld), c("PANEL"))))
  for (col in names(ld)) {
    v <- ld[[col]]
    if (col == "PANEL") next
    if (col %in% c("colour", "fill")) {
      out[[paste0(col, "_rgba")]] <- rgba_json(v)
    } else if (col == "group") {
      out$group <- I(as.integer(v))
    } else if (is.character(v) || is.factor(v)) {
      out[[col]] <- I(as.character(v))
    } else if (is.logical(v)) {
      out[[col]] <- I(v)
    } else {
      out[[col]] <- I(as.numeric(v))
    }
  }
  out$panel <- I(as.integer(ld$PANEL))
  out
}

write_marks_fixture <- function(plot, row, case, spec, out_file) {
  b <- ggplot2::ggplot_build(plot)
  pp <- b$layout$panel_params[[1]]
  geoms <- vapply(plot$layers, function(l) class(l$geom)[1], character(1))
  fixture <- list(
    row = row,
    case = case,
    spec = spec,
    collation = Sys.getlocale("LC_COLLATE"),
    versions = versions_json(),
    layers = lapply(seq_along(b$data), function(i) marks_layer_json(b$data[[i]], geoms[i])),
    panel = list(x = axis_json(pp$x), y = axis_json(pp$y))
  )
  writeLines(to_json(fixture), out_file, useBytes = TRUE)
  invisible(fixture)
}

# Standard preamble of every generate.R: locate this directory.
marks_here <- function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
}

# Run every case of a row: case = list(name, spec, plot = function(df) <ggplot>).
run_cases <- function(here, row, cases) {
  suppressPackageStartupMessages(library(ggplot2))
  for (cs in cases) {
    df <- read_case(here, paste0(cs$name, ".csv"))
    write_marks_fixture(cs$plot(df), row, cs$name, cs$spec,
                        file.path(here, paste0("expected_", cs$name, ".json")))
  }
}
