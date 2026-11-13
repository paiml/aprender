# Sets whose expanded range edge lands on a candidate break (#6909). R 4.6.2, ggplot2 4.0.3:
# LC_ALL=C Rscript generate.R > expected.tsv. Columns: id, x, y (%.17g), x labels, y labels.
suppressPackageStartupMessages(library(ggplot2))
stopifnot(identical(as.character(packageVersion("ggplot2")), "4.0.3"))
num <- function(v) paste(sprintf("%.17g", v), collapse = ",")
labs <- function(v) paste(v[!is.na(v)], collapse = " ")
emit <- function(id, x, y = x) {
  pp <- ggplot_build(ggplot(data.frame(x = x, y = y), aes(x, y)) + geom_point())$layout$panel_params[[1]]
  cat(id, num(x), num(y), labs(pp$x$get_labels()), paste0(labs(pp$y$get_labels()), "\n"), sep = "\t")
}
# hi: the upper edge hi + 0.05 (hi - lo) is B = m 10^e; lo: the lower edge is -B.
edge <- function(side, e, m, frac) {
  B <- m * 10^e
  if (side == "hi") { lo <- B * frac; hi <- (B + 0.05 * lo) / 1.05 }
  else { hi <- B * (1 + frac); lo <- (-B + 0.05 * hi) / 1.05 }
  emit(sprintf("A_%s_e%d_m%g_f%g", side, e, m, frac), c(lo, (lo + hi) / 2, hi))
}
edge("hi", -4, 1, 0.2); edge("lo", -4, 1, 0.2); edge("hi", -4, 2, 0.2)
edge("hi", -4, 4, 0.9); edge("hi", -4, 8, 0.9); edge("hi", -3, 2.5, 0.9)
edge("hi", -3, 3, 0.9); edge("hi", -2, 6, 0.9); edge("lo", -1, 5, 0.5)
edge("lo", 0, 1, 0); edge("hi", 3, 2.5, 0.5); edge("hi", 9, 7.5, 0.9)
edge("hi", 12, 3, 0.9); edge("hi", 14, 2.5, 0.5); edge("hi", 14, 7.5, 0.9)
x <- c(0.0226, 0.0234, 0.0246); emit("B542", x, rev(x))
x <- c(-2.97e15, -2.91e15, -2.87e15); emit("B1194", x, rev(x))
