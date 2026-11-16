# Data for the geom_violin row (PLOT-GG-001). Every y is a multiple of 1/64, so it is exact in the
# f32 cells aprender-viz's DataFrame stores. Run once; the CSVs are committed and generate.R reads them.
#   Rscript crates/aprender-viz/fixtures/ggplot2/geom_violin/make_data.R
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
q <- function(v) round(v * 64) / 64
out <- function(df, name) utils::write.csv(df, file.path(here, name), row.names = FALSE, quote = FALSE, na = "NA")
set.seed(20261116)

# Three x levels with unequal n; f is a two-level fill; xn is a numeric x per group.
g <- rep(c("a", "b", "c"), c(10, 6, 8))
three <- data.frame(g = g, y = q(c(rnorm(10), rnorm(6, 2), rnorm(8, 1))),
                    f = rep(c("p", "q"), length.out = 24),
                    xn = c(1.5, 5, 9)[match(g, c("a", "b", "c"))])
out(three, "data_three.csv")

out(data.frame(g = "a", y = q(rnorm(12))), "data_single.csv")

# Missing y in two rows (one in a, one in b).
na <- three[, c("g", "y")]
na$y[c(2, 15)] <- NA
out(na, "data_na.csv")

# d has one point and e two: ggplot2 drops d (one row) and keeps e.
tiny <- rbind(three[, c("g", "y")], data.frame(g = c("d", "e", "e"), y = c(5, 6, 7.5)))
out(tiny, "data_tiny.csv")

# a is constant; b is not.
out(data.frame(g = rep(c("a", "b"), c(4, 3)), y = c(1, 1, 1, 1, 1, 2, 3)), "data_constant.csv")
out(data.frame(g = "a", y = c(1, 1, 1, 1)), "data_constant_only.csv")

# x = a has three fill groups, x = b has two (p and r): the dodge splits by what is present.
dg <- data.frame(g = rep(c("a", "b"), c(15, 10)),
                 f = c(rep(c("p", "q", "r"), 5), rep(c("p", "r"), 5)))
dg$y <- q(rnorm(nrow(dg)) + ifelse(dg$f == "q", 1, 0))
out(dg, "data_dodge3.csv")
