# Helpers for the position-scale rows (PLOT-GG-001 train: discrete, log10, continuous arguments).
# Sourced after oracle.R. A fixture holds several `cases`, each one ggplot over the row's data.csv,
# so one data file exercises every scale argument.

# One position axis of panel 1, as the view scale resolved it. Numbers are exported in the
# scale's own space (log10 values for scale_*_log10), exactly as ggplot2 stores them.
#   kind              : "discrete" or "continuous"
#   limits            : continuous: trained limits; discrete: the level names
#   continuous_range  : the range the panel spans after expansion
#   breaks            : continuous: break positions inside the panel; discrete: level names
#   break_positions   : discrete only, the integer position of each break
#   labels            : one per kept break, as ggplot2 prints them (a label whose break fell
#                       outside the panel is dropped with it)
#   n_breaks_censored : continuous only; breaks proposed but outside the panel
#   minor_breaks      : continuous only; ggplot2's minor breaks inside the panel
axis_scale_json <- function(view) {
  br <- view$get_breaks()
  lab <- view$get_labels()
  if (is.character(view$limits)) {
    keep <- !is.na(br)
    return(list(
      kind = "discrete",
      limits = I(as.character(view$limits)),
      continuous_range = I(as.numeric(view$continuous_range)),
      breaks = I(as.character(br[keep])),
      break_positions = I(as.numeric(match(br[keep], view$limits))),
      labels = I(as.character(lab[keep]))
    ))
  }
  keep <- !is.na(br)
  list(
    kind = "continuous",
    limits = I(as.numeric(view$limits)),
    continuous_range = I(as.numeric(view$continuous_range)),
    breaks = I(as.numeric(br[keep])),
    labels = I(as.character(lab[keep])),
    n_breaks_censored = sum(!keep),
    minor_breaks = I(as.numeric(view$get_breaks_minor()))
  )
}

# `cases` is a list of list(name=, spec=, plot=, rows=). `rows` (optional, 1-based, repeats allowed)
# names the rows of data.csv the plot was built from; it is recorded so the Rust side builds the same
# frame. Writes one JSON document. `collate` records LC_COLLATE: discrete level order depends on it.
write_cases_fixture <- function(cases, row, data_file, out_file) {
  out <- lapply(cases, function(cs) {
    b <- ggplot2::ggplot_build(cs$plot)
    pp <- b$layout$panel_params[[1]]
    ld <- b$data[[1]]
    list(
      name = cs$name,
      spec = cs$spec,
      rows = if (is.null(cs$rows)) NULL else I(as.numeric(cs$rows)),
      layer = list(x = I(as.numeric(ld$x)), y = I(as.numeric(ld$y))),
      panel = list(x = axis_scale_json(pp$x), y = axis_scale_json(pp$y))
    )
  })
  fixture <- list(
    row = row,
    data_file = data_file,
    versions = c(versions_json(), list(collate = Sys.getlocale("LC_COLLATE"))),
    cases = out
  )
  writeLines(to_json(fixture), out_file, useBytes = TRUE)
  invisible(fixture)
}

fixture_here <- function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
}
