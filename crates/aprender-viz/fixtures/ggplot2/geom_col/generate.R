# Row 9 + stat_identity + position_stack: geom_col() over a discrete x (PLOT-GG-001, #6928).
# Regenerate (needs R + ggplot2 4.0.3; nothing else):
#   Rscript crates/aprender-viz/fixtures/ggplot2/geom_col/generate.R
# Writes expected.json beside this file. Commit both; tests never run R.
here <- (function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
})()
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "oracle_scales.R"))
suppressPackageStartupMessages(library(ggplot2))
stopifnot(as.character(packageVersion("ggplot2")) == "4.0.3")

df <- utils::read.csv(file.path(here, "data.csv"), stringsAsFactors = FALSE)
u <- df[df$g == "u", ]

cases <- list(
  list(name = "col_plain", spec = "ggplot(df[df$g == 'u', ], aes(x, y)) + geom_col()",
       plot = ggplot(u, aes(x, y)) + geom_col(), rows = which(df$g == "u")),
  list(name = "col_stack_fill", spec = "ggplot(df, aes(x, y, fill = g)) + geom_col()",
       plot = ggplot(df, aes(x, y, fill = g)) + geom_col()),
  list(name = "col_identity_fill",
       spec = "ggplot(df, aes(x, y, fill = g)) + geom_col(position = 'identity')",
       plot = ggplot(df, aes(x, y, fill = g)) + geom_col(position = "identity")),
  list(name = "col_width",
       spec = "ggplot(df[df$g == 'u', ], aes(x, y)) + geom_col(width = 0.5)",
       plot = ggplot(u, aes(x, y)) + geom_col(width = 0.5), rows = which(df$g == "u"))
)

out <- lapply(cases, function(cs) {
  b <- ggplot2::ggplot_build(cs$plot)
  pp <- b$layout$panel_params[[1]]
  ld <- b$data[[1]]
  stopifnot(class(cs$plot$layers[[1]]$geom)[1] == "GeomCol")
  list(
    name = cs$name,
    spec = cs$spec,
    rows = if (is.null(cs$rows)) NULL else I(as.numeric(cs$rows)),
    layer = list(
      x = I(as.numeric(ld$x)), y = I(as.numeric(ld$y)),
      ymin = I(as.numeric(ld$ymin)), ymax = I(as.numeric(ld$ymax)),
      xmin = I(as.numeric(ld$xmin)), xmax = I(as.numeric(ld$xmax)),
      group = I(as.integer(ld$group))
    ),
    panel = list(x = axis_scale_json(pp$x), y = axis_scale_json(pp$y))
  )
})
fixture <- list(row = "geom_col+stat_identity", data_file = "data.csv",
                versions = versions_json(), cases = out)
writeLines(to_json(fixture), file.path(here, "expected.json"), useBytes = TRUE)
