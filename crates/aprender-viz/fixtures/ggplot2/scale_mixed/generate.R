# Row: mixed position-scale cases (PLOT-GG-001 rows 2-4, fresh comparisons for #6916): discrete on
# either axis with and without a spec, continuous scales on wide/tiny/negative data, log10 on wide
# and sub-decade data, and a log axis beside a discrete or limited one.
# Regenerate: Rscript crates/aprender-viz/fixtures/ggplot2/scale_mixed/generate.R
here <- (function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
})()
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "oracle_scales.R"))
suppressPackageStartupMessages(library(ggplot2))
df <- utils::read.csv(file.path(here, "data.csv"), stringsAsFactors = FALSE)
case <- function(name, plot) list(name = name, spec = deparse1(substitute(plot)), plot = plot)

cases <- list(
  case("disc_x_nospec", ggplot(df, aes(g, y)) + geom_point()),
  case("disc_y_nospec", ggplot(df, aes(x, g)) + geom_point()),
  case("disc_x_default", ggplot(df, aes(g, y)) + geom_point() + scale_x_discrete()),
  case("disc_limits_reversed", ggplot(df, aes(g, y)) + geom_point() + scale_x_discrete(limits = c("d", "c", "b", "a"))),
  case("disc_expand_add1", ggplot(df, aes(g, y)) + geom_point() + scale_x_discrete(expand = expansion(add = 1))),
  case("disc_breaks_labels", ggplot(df, aes(g, y)) + geom_point() + scale_x_discrete(breaks = c("d", "a"), labels = c("D", "A"))),
  case("disc_y_limits_extra", ggplot(df, aes(x, g)) + geom_point() + scale_y_discrete(limits = c("a", "b", "c", "d", "e"))),
  case("disc_limits_subset", ggplot(df, aes(g, y)) + geom_point() + scale_x_discrete(limits = c("b", "a"))),
  case("cont_big_default", ggplot(df, aes(big, y)) + geom_point() + scale_x_continuous()),
  case("cont_small_default", ggplot(df, aes(small, y)) + geom_point() + scale_x_continuous()),
  case("cont_neg_default", ggplot(df, aes(n, y)) + geom_point() + scale_x_continuous()),
  case("cont_breaks_outside", ggplot(df, aes(x, y)) + geom_point() + scale_x_continuous(breaks = c(-100, 0, 5, 100))),
  case("cont_expand_asym", ggplot(df, aes(x, y)) + geom_point() + scale_y_continuous(expand = expansion(mult = c(0, 0.2)))),
  case("cont_y_labels", ggplot(df, aes(x, y)) + geom_point() + scale_y_continuous(breaks = c(0, 2.5, 5), labels = c("lo", "mid", "hi"))),
  case("cont_small_limits_none", ggplot(df, aes(x, small)) + geom_point() + scale_y_continuous(limits = c(2e-05, 4e-05), expand = expansion(0))),
  case("log_wide", ggplot(df, aes(p, y)) + geom_point() + scale_x_log10()),
  case("log_narrow", ggplot(df, aes(q, y)) + geom_point() + scale_x_log10()),
  case("log_y", ggplot(df, aes(x, p)) + geom_point() + scale_y_log10()),
  case("log_limits_breaks", ggplot(df, aes(p, y)) + geom_point() + scale_x_log10(limits = c(1, 10000), breaks = c(1, 100, 10000))),
  case("log_x_cont_y_limits", ggplot(df, aes(q, x)) + geom_point() + scale_x_log10() + scale_y_continuous(limits = c(0, 10))),
  case("log_limits_huge", ggplot(df, aes(p, y)) + geom_point() + scale_x_log10(limits = c(1e300, 1e300))),
  case("cont_limits_huge", ggplot(df, aes(x, y)) + geom_point() + scale_x_continuous(limits = c(-1e300, 1e300))),
  case("disc_x_log_y", ggplot(df, aes(g, big)) + geom_point() + scale_y_log10())
)

write_cases_fixture(cases, row = "scale_mixed", data_file = "data.csv",
                    out_file = file.path(here, "expected.json"))
