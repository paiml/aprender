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

# Round to binary32 and back: aprender-viz stores f32, so the inputs are exact on both sides.
f32 <- function(v) readBin(writeBin(v, raw(), size = 4), "numeric", size = 4, n = length(v))
num <- function(v) paste(sprintf("%.17g", v), collapse = ",")
labs <- function(v) paste(v[!is.na(v)], collapse = " ")

rows <- character()
for (e in c(-4:15)) {
  for (m in c(1, 2.5, 3.7, 7, 9.99)) {
    for (off in c(0, 1)) {
      b <- m * 10^e
      a <- if (off == 1) b * 0.37 else 0
      k <- 0:5
      x <- f32(a + (b - a) * k / 5)
      y <- f32((k * k - 7) * 0.5 * 10^max(e, 0))
      pp <- ggplot_build(ggplot(data.frame(x = x, y = y), aes(x, y)) + geom_point())$layout$panel_params[[1]]
      rows <- c(rows, paste(sprintf("e%d_m%g_o%d", e, m, off), num(x), num(y),
                            labs(pp$x$get_labels()), labs(pp$y$get_labels()), sep = "\t"))
    }
  }
}
writeLines(c("# id\tx\ty\tx_labels\ty_labels (ggplot2 4.0.3 get_labels(), NA dropped)", rows),
           file.path(here, "labels_magnitude.tsv"))
