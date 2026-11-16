# Row: geom_boxplot() = stat_boxplot + position_dodge2 (PLOT-GG-001, paiml/aprender#6909).
# Regenerate (needs R + ggplot2 4.0.3; nothing else):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_boxplot/generate.R
# Writes data.csv and expected.json beside this file. Commit both; tests never run R.
here <- (function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
})()
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "oracle_scales.R"))
suppressPackageStartupMessages(library(ggplot2))

# Every value is exactly representable in f32 (the Rust frame stores f32).
blk <- function(x, f, n, gr, y) data.frame(x = x, f = f, n = n, gr = gr, y = y,
                                           stringsAsFactors = FALSE)
d <- rbind(
  blk("a", "p", 1, 0, c(30, 1, 2, 3, 4, 5, 6, 7, 8, 9)),             # block A: high outlier
  blk("b", "p", 2, 0, c(-20, 10, 11, 12, 13, 14, 15, 16, 17, 18)),   # block B: low outlier
  blk("c", "q", 3, 0, rev(c(-30, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 60))), # C: both sides, unsorted
  blk("d", "p", 4, 0, 7),                                            # D: n = 1
  blk("e", "q", 5, 0, c(2, 1)),                                      # E: n = 2
  blk("f", "p", 6, 0, c(10, 1, 2)),                                  # F: n = 3
  blk("g", "q", 7, 0, c(100, 1, 2, 3)),                              # G: n = 4
  blk("h", "p", 8, 0, c(3, 1, 5, 2, 4)),                             # H: n = 5
  blk("i", "q", 9, 0, c(2, 3, 2, 3, 9, 2, 3, 2)),                    # I: ties
  blk("j", "p", 10, 0, c(4, 4, 4, 4, 4)),                            # J: all equal
  blk("u", "p", 1, 7, c(1, 2, 3, 4, 5, 6)),                          # K: several boxes per x
  blk("u", "q", 1, 3, c(2, 3, 4, 5, 6, 20)),
  blk("v", "p", 2, 7, c(10, 11, 12, 13, 14, 15)),
  blk("w", "q", 4, 3, c(3, 4, 5, 6, 7, 8)),
  blk("z", "p", 6, 1, c(1, 2, 3, 4, 5)),                             # N: three boxes at one x
  blk("z", "q", 6, 2, c(2, 3, 4, 5, 8)),
  blk("z", "r", 6, 3, c(0, 1, 2, 3, 4)),
  blk("a", "p", 1, 0, NA),                                           # NA y (row 59)
  blk("a", "p", NA, 0, 5)                                            # NA numeric x (row 60)
)
d$n <- as.numeric(d$n)   # a double, as the Rust frame holds it (R's integer x has resolution 1)
d$y <- as.numeric(d$y)
write.csv(d, file.path(here, "data.csv"), row.names = FALSE, na = "NA", quote = FALSE)
d <- utils::read.csv(file.path(here, "data.csv"), stringsAsFactors = FALSE)
d$n <- as.numeric(d$n); d$gr <- as.numeric(d$gr); d$y <- as.numeric(d$y)

# rows are located by value so the indices below cannot drift from the data above
ix <- function(...) which(d$x %in% c(...) & !is.na(d$y) & !is.na(d$n))
rows_of <- function(...) sort(unique(unlist(list(...))))
A <- ix("a"); A <- A[d$n[A] == 1]
B <- ix("b"); C <- ix("c"); Dd <- ix("d"); E <- ix("e"); Ff <- ix("f"); G <- ix("g")
H <- ix("h"); I <- ix("i"); J <- ix("j"); K <- ix("u", "v", "w"); N <- ix("z")
NAY <- which(is.na(d$y)); NAX <- which(is.na(d$n))

box <- function(m, ..., rows) {
  list(rows = rows, plot = ggplot(d[rows, , drop = FALSE], m) + geom_boxplot(...),
       probe = ggplot(d[rows, , drop = FALSE], m) +
         stat_boxplot(geom = "blank", position = "identity", ...))
}
cases <- list(
  c(list(name = "no_x_single", spec = "ggplot(d, aes(y = y)) + geom_boxplot()"),
    box(aes(y = y), rows = A)),
  c(list(name = "text_x_single", spec = "ggplot(d, aes(x, y)) + geom_boxplot()"),
    box(aes(x, y), rows = A)),
  c(list(name = "categories_outliers", spec = "ggplot(d, aes(x, y)) + geom_boxplot()"),
    box(aes(x, y), rows = rows_of(A, B, C))),
  c(list(name = "small_n_1_to_5", spec = "ggplot(d, aes(x, y)) + geom_boxplot()"),
    box(aes(x, y), rows = rows_of(Dd, E, Ff, G, H))),
  c(list(name = "ties_and_all_equal", spec = "ggplot(d, aes(x, y)) + geom_boxplot()"),
    box(aes(x, y), rows = rows_of(I, J))),
  c(list(name = "coef_6", spec = "ggplot(d, aes(x, y)) + geom_boxplot(coef = 6)"),
    box(aes(x, y), coef = 6, rows = rows_of(A, B, C))),
  c(list(name = "coef_zero", spec = "ggplot(d, aes(x, y)) + geom_boxplot(coef = 0)"),
    box(aes(x, y), coef = 0, rows = rows_of(A, B, C))),
  c(list(name = "coef_large", spec = "ggplot(d, aes(x, y)) + geom_boxplot(coef = 100)"),
    box(aes(x, y), coef = 100, rows = rows_of(A, C))),
  c(list(name = "width_half", spec = "ggplot(d, aes(x, y)) + geom_boxplot(width = 0.5)"),
    box(aes(x, y), width = 0.5, rows = rows_of(A, B, C))),
  c(list(name = "width_wide_overlap", spec = "ggplot(d, aes(x, y)) + geom_boxplot(width = 2)"),
    box(aes(x, y), width = 2, rows = rows_of(A, B))),
  c(list(name = "fill_split", spec = "ggplot(d, aes(x, y, fill = f)) + geom_boxplot()"),
    box(aes(x, y, fill = f), rows = K)),
  c(list(name = "colour_split", spec = "ggplot(d, aes(x, y, colour = f)) + geom_boxplot()"),
    box(aes(x, y, colour = f), rows = K)),
  c(list(name = "group_split", spec = "ggplot(d, aes(x, y, group = gr)) + geom_boxplot()"),
    box(aes(x, y, group = gr), rows = K)),
  c(list(name = "group_split_width_half",
         spec = "ggplot(d, aes(x, y, group = gr)) + geom_boxplot(width = 0.5)"),
    box(aes(x, y, group = gr), width = 0.5, rows = K)),
  c(list(name = "continuous_x_one_box_width_half",
         spec = "ggplot(d, aes(n, y)) + geom_boxplot(width = 0.5)"),
    box(aes(n, y), width = 0.5, rows = rows_of(A, B, C))),
  c(list(name = "fill_split_width_half",
         spec = "ggplot(d, aes(x, y, fill = f)) + geom_boxplot(width = 0.5)"),
    box(aes(x, y, fill = f), width = 0.5, rows = K)),
  c(list(name = "three_boxes_one_x", spec = "ggplot(d, aes(x, y, fill = f)) + geom_boxplot()"),
    box(aes(x, y, fill = f), rows = N)),
  c(list(name = "no_x_fill", spec = "ggplot(d, aes(y = y, fill = f)) + geom_boxplot()"),
    box(aes(y = y, fill = f), rows = K)),
  c(list(name = "continuous_x_one_box", spec = "ggplot(d, aes(n, y)) + geom_boxplot()"),
    box(aes(n, y), rows = rows_of(A, B, C))),
  c(list(name = "continuous_x_grouped",
         spec = "ggplot(d, aes(n, y, group = n)) + geom_boxplot()"),
    box(aes(n, y, group = n), rows = rows_of(A, B, C))),
  c(list(name = "continuous_x_fill", spec = "ggplot(d, aes(n, y, fill = f)) + geom_boxplot()"),
    box(aes(n, y, fill = f), rows = K)),
  c(list(name = "resolution_counts_zero",
         spec = "ggplot(d, aes(n, y, group = n)) + geom_boxplot()"),
    box(aes(n, y, group = n), rows = rows_of(B, J))),
  c(list(name = "na_y_dropped", spec = "ggplot(d, aes(x, y)) + geom_boxplot()"),
    box(aes(x, y), rows = rows_of(A, NAY))),
  c(list(name = "group_all_missing_y_vanishes",
         spec = "ggplot(d, aes(x, y)) + geom_boxplot()"),
    box(aes(x, y), rows = rows_of(B, which(d$x == "a" & is.na(d$y))))),
  c(list(name = "na_numeric_x_dropped",
         spec = "ggplot(d, aes(n, y, group = n)) + geom_boxplot()"),
    box(aes(n, y, group = n), rows = rows_of(A, NAX)))
)

num <- function(v) I(as.numeric(v))
out <- lapply(cases, function(cs) {
  warns <- character()
  b <- withCallingHandlers(ggplot_build(cs$plot), warning = function(w) {
    warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning")
  })
  pr <- suppressWarnings(ggplot_build(cs$probe))$data[[1]]
  pp <- b$layout$panel_params[[1]]
  ld <- b$data[[1]]
  stopifnot(inherits(cs$plot$layers[[1]]$geom, "GeomBoxplot"))
  layer <- list(geom = class(cs$plot$layers[[1]]$geom)[1])
  for (nm in c("group", "x", "xmin", "xmax", "ymin", "lower", "middle", "upper", "ymax",
               "notchlower", "notchupper", "ymin_final", "ymax_final")) {
    layer[[nm]] <- num(ld[[nm]])
  }
  # relvarwidth is dropped from layer data by GeomBoxplot$setup_data; read it, per group, from
  # the stat alone (geom = "blank", position = "identity" keeps it).
  layer$relvarwidth <- num(pr$relvarwidth[match(ld$group, pr$group)])
  layer$outliers <- lapply(ld$outliers, function(o) num(o))
  list(name = cs$name, spec = cs$spec, rows = num(cs$rows),
       r_warnings = I(as.character(warns)), layer = layer,
       panel = list(x = axis_scale_json(pp$x), y = axis_scale_json(pp$y)))
})
fixture <- list(row = "geom_boxplot", data_file = "data.csv", versions = versions_json(),
                cases = out)
writeLines(to_json(fixture), file.path(here, "expected.json"), useBytes = TRUE)
