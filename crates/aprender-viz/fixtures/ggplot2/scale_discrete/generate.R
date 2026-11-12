# Rows: scale_x_discrete + scale_y_discrete (levels, limits, default expansion add 0.6, breaks, labels).
# Regenerate (needs R + ggplot2 4.0.3; nothing else):
#   Rscript crates/aprender-viz/fixtures/ggplot2/scale_discrete/generate.R
here <- (function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
})()
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "oracle_scales.R"))
# Level order is the sort order of the locale: pin it so the oracle does not depend on the host.
# The contract states the Rust side matches the C locale (byte order); other orders need limits=.
invisible(Sys.setlocale("LC_COLLATE", "C"))
stopifnot(Sys.getlocale("LC_COLLATE") == "C")
suppressPackageStartupMessages(library(ggplot2))
df <- utils::read.csv(file.path(here, "data.csv"), stringsAsFactors = FALSE)

pt <- function(...) ggplot(df, ...) + geom_point()
cases <- list(
  list(name = "x_default", spec = "ggplot(df, aes(g, y)) + geom_point()",
       plot = pt(aes(g, y))),
  list(name = "y_default", spec = "ggplot(df, aes(x, g)) + geom_point()",
       plot = pt(aes(x, g))),
  list(name = "x_limits_extra_level",
       spec = "ggplot(df, aes(g, y)) + geom_point() + scale_x_discrete(limits = c('mid', 'low', 'high', 'extra'))",
       plot = pt(aes(g, y)) + scale_x_discrete(limits = c("mid", "low", "high", "extra"))),
  list(name = "x_breaks_labels",
       spec = "ggplot(df, aes(g, y)) + geom_point() + scale_x_discrete(breaks = c('high', 'mid'), labels = c('H', 'M'))",
       plot = pt(aes(g, y)) + scale_x_discrete(breaks = c("high", "mid"), labels = c("H", "M"))),
  list(name = "x_expand",
       spec = "ggplot(df, aes(g, y)) + geom_point() + scale_x_discrete(expand = expansion(mult = c(0, 0.1), add = c(0.2, 1)))",
       plot = pt(aes(g, y)) + scale_x_discrete(expand = expansion(mult = c(0, 0.1), add = c(0.2, 1)))),
  list(name = "x_single_level",
       spec = "ggplot(df[df$g == 'low', ], aes(g, y)) + geom_point()",
       plot = ggplot(df[df$g == "low", ], aes(g, y)) + geom_point(), rows = which(df$g == "low")),
  list(name = "x_mixed_case_levels",
       spec = "ggplot(df, aes(m, y)) + geom_point()  # m = b,B,a,A,10,9,Zeta,alpha under LC_COLLATE=C",
       plot = pt(aes(m, y))),
  list(name = "y_limits_labels",
       spec = "ggplot(df, aes(x, g)) + geom_point() + scale_y_discrete(limits = c('mid', 'low', 'high'), labels = c('M', 'L', 'H'))",
       plot = pt(aes(x, g)) + scale_y_discrete(limits = c("mid", "low", "high"), labels = c("M", "L", "H")))
)

write_cases_fixture(cases, row = "scale_x_discrete+scale_y_discrete", data_file = "data.csv",
                    out_file = file.path(here, "expected.json"))
