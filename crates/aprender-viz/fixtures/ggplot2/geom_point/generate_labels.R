# Row 1, magnitude fixture: default axis labels across magnitudes (fixed and scientific).
# Regenerate (needs R + ggplot2 4.0.3; nothing else):
#   Rscript crates/aprender-viz/fixtures/ggplot2/geom_point/generate_labels.R
# Writes labels_magnitude.tsv beside this file. Commit both; tests never run R.
here <- (function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
})()
suppressPackageStartupMessages(library(ggplot2))
stopifnot(identical(as.character(packageVersion("ggplot2")), "4.0.3"))

# The inputs are R doubles, NOT rounded to binary32: aprender-viz stores f32, and the test must
# see whether that storage changes a label. (An earlier version pre-rounded every input and so
# could not see the error; review of paiml/aprender#6992, B2'.) A set whose id starts with
# "refuse_" is one the verb must refuse by name; R's labels are recorded to show what ggplot2 prints.
num <- function(v) paste(sprintf("%.17g", v), collapse = ",")
labs <- function(v) paste(v[!is.na(v)], collapse = " ")

row_for <- function(id, x, y) {
  pp <- ggplot_build(ggplot(data.frame(x = x, y = y), aes(x, y)) + geom_point())$layout$panel_params[[1]]
  paste(id, num(x), num(y), labs(pp$x$get_labels()), labs(pp$y$get_labels()), sep = "\t")
}
rows <- character()
for (e in c(-4:15)) {
  for (m in c(1, 2.5, 3.7, 7, 9.99)) {
    for (off in c(0, 1)) {
      b <- m * 10^e
      a <- if (off == 1) b * 0.37 else 0
      k <- 0:5
      x <- a + (b - a) * k / 5
      y <- (k * k - 7) * 0.5 * 10^max(e, 0)
      rows <- c(rows, row_for(sprintf("e%d_m%g_o%d", e, m, off), x, y))
    }
  }
}
# Sets where f32 storage changes the labels (review of #6992): the verb refuses these by name.
for (s in list(
  list("refuse_16777217", c(16777217, 16777218, 16777219)),
  list("refuse_16777217_pair", c(16777217, 16777219)),
  list("refuse_99999_99", c(99999.99, 100000.01)),
  list("refuse_0_9999999", c(0.9999999, 1)),
  list("refuse_9_999999e15", c(9.999999e15, 9.9999999e15)),
  list("refuse_3e9_ties", c(3000000000, 3000000500)),
  list("refuse_3e9_ids", c(3000000001, 3000000257, 3000000513)))) {
  rows <- c(rows, row_for(s[[1]], s[[2]], s[[2]]))
}
writeLines(c("# id\tx\ty\tx_labels\ty_labels (ggplot2 4.0.3 get_labels(), NA dropped)", rows),
           file.path(here, "labels_magnitude.tsv"))
