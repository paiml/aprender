# Row: scale edge cases found in review of #6916 (NaN limits = R's NA, overflow, duplicate discrete
# breaks, unsorted minor breaks, subnormal labels, zero_range tolerance) plus the refusals: inputs
# where R errors, or where R returns numbers this crate declines to reproduce.
# Regenerate: LC_COLLATE=C Rscript crates/aprender-viz/fixtures/ggplot2/scale_edge/generate.R
here <- (function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
})()
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "oracle_scales.R"))
suppressPackageStartupMessages(library(ggplot2))
df <- utils::read.csv(file.path(here, "data.csv"), stringsAsFactors = FALSE)
M <- .Machine$double.xmax
P <- ggplot(df, aes(x, y)) + geom_point()
D <- ggplot(df, aes(g, y)) + geom_point()

cases <- list(
  # NA limit = use the data range at that end (positional, not sorted).
  list(name = "limits_na_lo", spec = "scale_x_continuous(limits = c(NA, 10))",
       plot = P + scale_x_continuous(limits = c(NA, 10))),
  list(name = "limits_na_hi", spec = "scale_x_continuous(limits = c(0, NA))",
       plot = P + scale_x_continuous(limits = c(0, NA))),
  list(name = "limits_na_both", spec = "scale_x_continuous(limits = c(NA, NA))",
       plot = P + scale_x_continuous(limits = c(NA, NA))),
  list(name = "limits_na_lo_log", spec = "scale_x_log10(limits = c(NA, 100))",
       plot = P + scale_x_log10(limits = c(NA, 100))),
  # Overflow of the span and of the expansion.
  list(name = "limits_pm_max", spec = "scale_x_continuous(limits = c(-M, M))",
       plot = P + scale_x_continuous(limits = c(-M, M))),
  list(name = "limits_zero_max", spec = "scale_x_continuous(limits = c(0, M))",
       plot = P + scale_x_continuous(limits = c(0, M))),
  list(name = "limits_pm_max_no_expand", spec = "scale_x_continuous(limits = c(-M, M), expand = expansion(0))",
       plot = P + scale_x_continuous(limits = c(-M, M), expand = expansion(0))),
  # Duplicate discrete limits / breaks.
  list(name = "disc_dup_limits", spec = "scale_x_discrete(limits = c('a', 'a', 'b'))",
       plot = D + scale_x_discrete(limits = c("a", "a", "b"))),
  list(name = "disc_dup_breaks", spec = "scale_x_discrete(breaks = c('a', 'a'))",
       plot = D + scale_x_discrete(breaks = c("a", "a"))),
  list(name = "disc_dup_breaks_labels", spec = "scale_x_discrete(breaks = c('a', 'a', 'b'), labels = c('A', 'B', 'C'))",
       plot = D + scale_x_discrete(breaks = c("a", "a", "b"), labels = c("A", "B", "C"))),
  # Unsorted major breaks: scales::regular_minor_breaks tests max(b) after inserting the low end.
  list(name = "breaks_unsorted", spec = "scale_x_continuous(breaks = c(5, 1, 3))",
       plot = P + scale_x_continuous(breaks = c(5, 1, 3))),
  # Subnormal limits: labels in scientific form with a three-digit exponent.
  list(name = "limits_subnormal", spec = "scale_x_continuous(limits = c(0, 5e-324))",
       plot = P + scale_x_continuous(limits = c(0, 5e-324))),
  # scales::zero_range() tolerance is 1000 * eps (relative): 1 + 1e-15 is a zero-width range.
  list(name = "limits_zero_range_tol", spec = "scale_x_continuous(limits = c(1, 1 + 1e-15))",
       plot = P + scale_x_continuous(limits = c(1, 1 + 1e-15)))
)
write_cases_fixture(cases, row = "scale_edge", data_file = "data.csv",
                    out_file = file.path(here, "expected.json"))

# Refusals: what R does with an input this crate refuses with a typed error.
outcome <- function(name, spec, plot, axis = "x") {
  r <- tryCatch({
    pp <- ggplot_build(plot)$layout$panel_params[[1]][[axis]]
    list(err = NULL, range = I(as.numeric(pp$continuous_range)))
  }, error = function(e) list(err = conditionMessage(e), range = NULL))
  list(name = name, spec = spec, r_error = r$err, r_continuous_range = r$range)
}
refusals <- list(
  outcome("cont_labels_short", "breaks = c(1, 3, 5), labels = c('a', 'b')",
          P + scale_x_continuous(breaks = c(1, 3, 5), labels = c("a", "b"))),
  outcome("cont_labels_long", "breaks = c(1, 3, 5), labels = c('a', 'b', 'c', 'd')",
          P + scale_x_continuous(breaks = c(1, 3, 5), labels = c("a", "b", "c", "d"))),
  outcome("cont_labels_default_breaks", "labels = c('a', 'b')",
          P + scale_x_continuous(labels = c("a", "b"))),
  outcome("disc_labels_short", "breaks = c('a', 'b', 'c'), labels = c('A', 'B')",
          D + scale_x_discrete(breaks = c("a", "b", "c"), labels = c("A", "B"))),
  outcome("disc_labels_default_breaks_short", "labels = c('A', 'B')",
          D + scale_x_discrete(labels = c("A", "B"))),
  outcome("expand_neg_mult", "expand = expansion(mult = -0.6)",
          P + scale_x_continuous(expand = expansion(mult = -0.6))),
  outcome("expand_neg_add", "expand = expansion(add = -1)",
          P + scale_x_continuous(expand = expansion(add = -1))),
  outcome("expand_nan", "expand = expansion(mult = c(NaN, 0))",
          P + scale_x_continuous(expand = expansion(mult = c(NaN, 0)))),
  outcome("expand_inf_add", "expand = expansion(add = Inf)",
          P + scale_x_continuous(expand = expansion(add = Inf))),
  outcome("disc_expand_neg", "expand = expansion(add = -0.9)",
          D + scale_x_discrete(expand = expansion(add = -0.9))),
  outcome("limits_na_hi_below_data", "limits = c(10, NA)",
          P + scale_x_continuous(limits = c(10, NA)))
)
writeLines(to_json(list(row = "scale_edge", versions = versions_json(), refusals = refusals)),
           file.path(here, "refusals.json"), useBytes = TRUE)
