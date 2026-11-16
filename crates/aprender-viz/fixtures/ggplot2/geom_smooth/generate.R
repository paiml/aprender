# Row: geom_smooth(method = "lm") (PLOT-GG-001, paiml/aprender#6983, master #6909): stat_smooth with
# formula y ~ x, the lm fit evaluated at n points per group, the standard error and the band.
# Regenerate (R + ggplot2 4.0.3 only, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_smooth/generate.R
# Writes expected_<case>.json and qt_table.json beside this file. Commit them; tests never run R.
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
source(file.path(here, "..", "oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

row <- "geom_smooth"
rd <- function(file) utils::read.csv(file.path(here, file), stringsAsFactors = FALSE, na.strings = "NA")
out <- function(case) file.path(here, paste0("expected_", case, ".json"))

# One layer of ggplot_build() output: the stat's columns and the group id. Colour, fill, linewidth
# and the other style columns are not recorded: the engine does not resolve them.
smooth_layer_json <- function(ld, geom) {
  res <- list(geom = geom)
  for (col in c("x", "y", "ymin", "ymax", "se")) {
    if (col %in% names(ld)) res[[col]] <- I(as.numeric(ld[[col]]))
  }
  res$flipped_aes <- I(as.integer(ld$flipped_aes))
  res$group <- I(as.integer(ld$group))
  res$n_rows <- nrow(ld)
  res
}

w <- function(case, plot, spec) {
  b <- suppressWarnings(ggplot2::ggplot_build(plot))
  pp <- b$layout$panel_params[[1]]
  geoms <- vapply(plot$layers, function(l) class(l$geom)[1], character(1))
  fixture <- list(
    row = row, case = case, spec = spec,
    collation = Sys.getlocale("LC_COLLATE"),
    versions = versions_json(),
    layers = lapply(seq_along(b$data), function(i) smooth_layer_json(b$data[[i]], geoms[i])),
    panel = list(x = axis_json(pp$x), y = axis_json(pp$y))
  )
  writeLines(to_json(fixture), out(case), useBytes = TRUE)
}

sm <- function(...) geom_smooth(method = "lm", formula = y ~ x, ...)
base <- function(file) ggplot(rd(file), aes(x, y))
S <- "ggplot(d, aes(x, y)) + geom_smooth(method = \"lm\", formula = y ~ x"

w("clean", base("data_clean.csv") + sm(), paste0(S, ")"))
w("noisy", base("data_noisy.csv") + sm(), paste0(S, ")"))
w("n2", base("data_n2.csv") + sm(), paste0(S, ")"))
w("n3", base("data_n3.csv") + sm(), paste0(S, ")"))
w("constant_x", base("data_constant_x.csv") + sm() +
    geom_point(data = rd("data_point.csv")),
  paste0(S, ") + geom_point(data = pts)"))
w("constant_y", base("data_constant_y.csv") + sm(), paste0(S, ")"))
w("negative", base("data_negative.csv") + sm(), paste0(S, ")"))
w("offset", base("data_offset.csv") + sm(), paste0(S, ")"))
w("na", base("data_na.csv") + sm(), paste0(S, ")"))
w("se_false", base("data_noisy.csv") + sm(se = FALSE), paste0(S, ", se = FALSE)"))
w("level90", base("data_noisy.csv") + sm(level = 0.9), paste0(S, ", level = 0.9)"))
w("n5", base("data_noisy.csv") + sm(n = 5), paste0(S, ", n = 5)"))
w("groups", ggplot(rd("data_groups.csv"), aes(x, y, group = g)) + sm(),
  "ggplot(d, aes(x, y, group = g)) + geom_smooth(method = \"lm\", formula = y ~ x)")
w("group_vanishes", ggplot(rd("data_groups_vanish.csv"), aes(x, y, group = g)) + sm(),
  "ggplot(d, aes(x, y, group = g)) + geom_smooth(method = \"lm\", formula = y ~ x)")
w("with_points", base("data_noisy.csv") + geom_point() + sm() +
    geom_point(data = rd("data_point.csv")),
  "ggplot(d, aes(x, y)) + geom_point() + geom_smooth(method = \"lm\", formula = y ~ x) + geom_point(data = pts)")
w("own_data", ggplot(rd("data_point.csv"), aes(x, y)) + geom_point() + sm(data = rd("data_wide.csv")),
  "ggplot(pts, aes(x, y)) + geom_point() + geom_smooth(method = \"lm\", formula = y ~ x, data = wide)")


# Offset ladder: the same y over x = c + k * spacing, from c = 1e3 to 1e10 and down to unit spacing
# near 1. R's lm() drops the slope (rank 1, a flat line at mean(y), df = n - 1) once the slope
# column's norm falls below 1e-7 of its original norm; every case is f32-exact.
for (case in c("ladder_1e3", "ladder_1e5", "ladder_1e7", "ladder_3e7", "ladder_1e8",
               "ladder_1e8_rank1", "ladder_1e10", "ladder_1e10_rank1", "ladder_small",
               "ladder_small_rank1", "ladder_edge_full", "ladder_edge_rank1")) {
  d <- rd(paste0("data_", case, ".csv"))
  w(case, ggplot(d, aes(x, y)) + sm(), paste0(S, ")"))
  fit <- lm(y ~ x, d)
  cat(sprintf("%s rank=%d\n", case, fit$rank))
}

# Student-t upper-tail quantiles, for the in-crate implementation to be tested against directly.
dfs <- c(1, 2, 3, 5, 10, 30, 78, 100, 1000, 1e4, 1e6)
qs <- c(0.5, 0.25, 0.1, 0.05, 0.025, 0.005, 0.0005, 1e-6)
tab <- expand.grid(q = qs, df = dfs)
tab$t <- mapply(function(q, d) qt(q, d, lower.tail = FALSE), tab$q, tab$df)
qt_fixture <- list(
  row = row, case = "qt", spec = "qt(q, df, lower.tail = FALSE)",
  versions = versions_json(),
  q = I(tab$q), df = I(tab$df), t = I(tab$t)
)
writeLines(to_json(qt_fixture), file.path(here, "qt_table.json"), useBytes = TRUE)
