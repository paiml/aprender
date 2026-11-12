# ggplot2 conformance oracle: shared helpers (PLOT-GG-001, paiml/aprender#6909).
#
# Sourced by every row's generate.R. It turns a ggplot object into the JSON fixture that
# aprender-viz's conformance tests compare against. R is needed ONLY to regenerate fixtures;
# no test, build or CI job runs R.
#
# jsonlite is deliberately not used (it is not installed on the reference host, and the oracle
# should not depend on a serializer whose number formatting we do not control). The writer below
# emits every double with 17 significant digits, which round-trips an IEEE-754 binary64 exactly.

json_str <- function(s) {
  s <- gsub("\\", "\\\\", s, fixed = TRUE)
  s <- gsub("\"", "\\\"", s, fixed = TRUE)
  s <- gsub("\n", "\\n", s, fixed = TRUE)
  paste0("\"", s, "\"")
}

json_scalar <- function(v) {
  if (is.na(v)) return("null")
  if (is.logical(v)) return(if (v) "true" else "false")
  if (is.numeric(v)) {
    if (!is.finite(v)) stop("non-finite number has no JSON form: ", v)
    return(sprintf("%.17g", v))
  }
  json_str(as.character(v))
}

# Unnamed list / atomic vector of length != 1 -> array; named list -> object.
# `I(x)` (class AsIs) forces a length-1 vector to be written as an array.
to_json <- function(x, indent = 0L) {
  pad <- strrep("  ", indent + 1L)
  end <- strrep("  ", indent)
  if (is.null(x)) return("null")
  if (is.factor(x)) x <- as.character(x)
  if (is.list(x) && !is.null(names(x))) {
    if (length(x) == 0L) return("{}")
    items <- vapply(seq_along(x), function(i) {
      paste0(pad, json_str(names(x)[i]), ": ", to_json(x[[i]], indent + 1L))
    }, character(1))
    return(paste0("{\n", paste(items, collapse = ",\n"), "\n", end, "}"))
  }
  if (is.list(x)) {
    if (length(x) == 0L) return("[]")
    items <- vapply(x, function(e) paste0(pad, to_json(e, indent + 1L)), character(1))
    return(paste0("[\n", paste(items, collapse = ",\n"), "\n", end, "]"))
  }
  if (length(x) == 1L && !inherits(x, "AsIs")) return(json_scalar(x))
  paste0("[", paste(vapply(as.list(x), json_scalar, character(1)), collapse = ", "), "]")
}

# One position axis of panel 1, as ggplot2 resolved it.
#   limits           : the scale's trained limits (data range for a continuous scale)
#   continuous_range : the limits after the default expansion -- what the panel shows
#   breaks / labels  : major breaks inside the panel and their labels. ggplot2 keeps a break
#                      that falls outside the panel as NA; those are dropped here and counted in
#                      `n_breaks_censored`, so a renderer compares only what is drawn.
axis_json <- function(view) {
  if (isTRUE(view$is_discrete())) return(discrete_axis_json(view))
  br <- view$get_breaks()
  lab <- view$get_labels()
  keep <- !is.na(br)
  list(
    limits = I(as.numeric(view$limits)),
    continuous_range = I(as.numeric(view$continuous_range)),
    breaks = I(as.numeric(br[keep])),
    labels = I(as.character(lab[keep])),
    n_breaks_censored = sum(!keep)
  )
}

# A discrete position axis: the levels, the continuous range the panel spans (levels sit at 1..n),
# and the position of each break. `labels` are the level names.
discrete_axis_json <- function(view) {
  br <- view$get_breaks()
  list(
    limits = I(as.character(view$limits)),
    continuous_range = I(as.numeric(view$continuous_range)),
    breaks = I(as.numeric(attr(br, "pos"))),
    labels = I(as.character(view$get_labels())),
    n_breaks_censored = sum(is.na(br))
  )
}

# Layer data as ggplot_build() returns it, plus every colour resolved to 0..255 RGBA so the Rust
# side compares numbers, not R colour names.
layer_json <- function(ld, geom, drop = c("PANEL", "group")) {
  out <- list(geom = geom)
  for (col in setdiff(names(ld), drop)) {
    v <- ld[[col]]
    out[[col]] <- I(if (is.factor(v)) as.character(v) else v)
  }
  if ("colour" %in% names(ld)) {
    rgba <- grDevices::col2rgb(ld$colour, alpha = TRUE)
    out$colour_rgba <- lapply(seq_len(ncol(rgba)), function(i) I(as.integer(rgba[, i])))
  }
  out
}

versions_json <- function() {
  list(
    R = R.version.string,
    ggplot2 = as.character(utils::packageVersion("ggplot2")),
    scales = as.character(utils::packageVersion("scales")),
    labeling = as.character(utils::packageVersion("labeling"))
  )
}

# Build `plot`, extract layer data + panel 1's x/y scales, write the fixture.
write_fixture <- function(plot, row, spec, data_file, out_file, drop = c("PANEL", "group")) {
  b <- ggplot2::ggplot_build(plot)
  pp <- b$layout$panel_params[[1]]
  geoms <- vapply(plot$layers, function(l) class(l$geom)[1], character(1))
  fixture <- list(
    row = row,
    spec = spec,
    data_file = data_file,
    versions = versions_json(),
    layers = lapply(seq_along(b$data), function(i) layer_json(b$data[[i]], geoms[i], drop)),
    panel = list(x = axis_json(pp$x), y = axis_json(pp$y))
  )
  writeLines(to_json(fixture), out_file, useBytes = TRUE)
  invisible(fixture)
}

