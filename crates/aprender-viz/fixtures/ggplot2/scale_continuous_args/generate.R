# Row: scale_x_continuous / scale_y_continuous arguments limits, breaks, labels, expand = expansion().
# Regenerate: Rscript crates/aprender-viz/fixtures/ggplot2/scale_continuous_args/generate.R
here <- (function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
})()
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "oracle_scales.R"))
suppressPackageStartupMessages(library(ggplot2))
df <- utils::read.csv(file.path(here, "data.csv"))
pt <- ggplot(df, aes(x, y)) + geom_point()

cases <- list(
  list(name = "limits_no_expand_x",
       spec = "ggplot(df, aes(x, y)) + geom_point() + scale_x_continuous(limits = c(0, 10), expand = expansion(0))",
       plot = pt + scale_x_continuous(limits = c(0, 10), expand = expansion(0))),
  list(name = "limits_no_expand_y",
       spec = "ggplot(df, aes(x, y)) + geom_point() + scale_y_continuous(limits = c(-4, 8), expand = expansion(0))",
       plot = pt + scale_y_continuous(limits = c(-4, 8), expand = expansion(0))),
  list(name = "breaks_labels_expansion",
       spec = "ggplot(df, aes(x, y)) + geom_point() + scale_x_continuous(breaks = c(2, 5, 8, 20), labels = c('two', 'five', 'eight', 'twenty'), expand = expansion(mult = c(0.02, 0.1), add = c(1, 0)))",
       plot = pt + scale_x_continuous(breaks = c(2, 5, 8, 20),
                                      labels = c("two", "five", "eight", "twenty"),
                                      expand = expansion(mult = c(0.02, 0.1), add = c(1, 0)))),
  list(name = "limits_breaks_y",
       spec = "ggplot(df, aes(x, y)) + geom_point() + scale_y_continuous(limits = c(-3, 8), breaks = c(0, 4, 8))",
       plot = pt + scale_y_continuous(limits = c(-3, 8), breaks = c(0, 4, 8))),
  list(name = "add_expansion_y",
       spec = "ggplot(df, aes(x, y)) + geom_point() + scale_y_continuous(expand = expansion(add = 2))",
       plot = pt + scale_y_continuous(expand = expansion(add = 2))),
  # Reversed limits: ggplot2 sorts them (limits = c(10, 0) is limits 0..10).
  list(name = "limits_reversed_x",
       spec = "ggplot(df, aes(x, y)) + geom_point() + scale_x_continuous(limits = c(10, 0))",
       plot = pt + scale_x_continuous(limits = c(10, 0))),
  # Values outside explicit limits become NA but the row stays in the layer data.
  list(name = "limits_oob_na_x",
       spec = "ggplot(df, aes(x, y)) + geom_point() + scale_x_continuous(limits = c(2, 8))",
       plot = pt + scale_x_continuous(limits = c(2, 8))),
  list(name = "limits_oob_na_y",
       spec = "ggplot(df, aes(x, y)) + geom_point() + scale_y_continuous(limits = c(0, 4))",
       plot = pt + scale_y_continuous(limits = c(0, 4))),
  # Zero-width ranges: expand_range() uses a width of 1 for the multiplicative part, so the panel
  # is value +- (1 * mult + add), i.e. 0.05 each side by default (not +-1 and not +-0.5).
  list(name = "single_point",
       spec = "ggplot(df[3, ], aes(x, y)) + geom_point()",
       plot = ggplot(df[3, ], aes(x, y)) + geom_point(), rows = 3),
  list(name = "all_equal",
       spec = "ggplot(df[c(3, 3, 3), ], aes(x, y)) + geom_point()",
       plot = ggplot(df[c(3, 3, 3), ], aes(x, y)) + geom_point(), rows = c(3, 3, 3)),
  list(name = "limits_equal_x",
       spec = "ggplot(df[1:3, ], aes(x, y)) + geom_point() + scale_x_continuous(limits = c(3, 3))",
       plot = ggplot(df[1:3, ], aes(x, y)) + geom_point() + scale_x_continuous(limits = c(3, 3)),
       rows = 1:3),
  list(name = "single_point_add_expand",
       spec = "ggplot(df[3, ], aes(x, y)) + geom_point() + scale_x_continuous(expand = expansion(mult = 0.1, add = 2))",
       plot = ggplot(df[3, ], aes(x, y)) + geom_point() +
         scale_x_continuous(expand = expansion(mult = 0.1, add = 2)), rows = 3),

  # --- Single value 0: zero_range() must treat lo == hi as zero-width even at 0 (relative tolerance is
  # undefined there); width 1 gives 0 +- 0.05.
  list(name = "single_zero",
       spec = "ggplot(df[1, ], aes(z, y)) + geom_point()",
       plot = ggplot(df[1, ], aes(z, y)) + geom_point(), rows = 1),
  list(name = "single_zero_add_expand",
       spec = "ggplot(df[1, ], aes(z, y)) + geom_point() + scale_x_continuous(expand = expansion(mult = 0.1, add = 2))",
       plot = ggplot(df[1, ], aes(z, y)) + geom_point() +
         scale_x_continuous(expand = expansion(mult = 0.1, add = 2)), rows = 1),
  list(name = "limits_zero_zero",
       spec = "ggplot(df[1:3, ], aes(z, y)) + geom_point() + scale_x_continuous(limits = c(0, 0))",
       plot = ggplot(df[1:3, ], aes(z, y)) + geom_point() + scale_x_continuous(limits = c(0, 0)), rows = 1:3),
  # --- Infinite data (review of #6916): training ignores them; censor(only.finite = TRUE) keeps them.
  list(name = "inf_mixed_x",
       spec = "ggplot(df, aes(f, y)) + geom_point()",
       plot = ggplot(df, aes(f, y)) + geom_point()),
  list(name = "inf_limits_x",
       spec = "ggplot(df, aes(f, y)) + geom_point() + scale_x_continuous(limits = c(0, 3))",
       plot = ggplot(df, aes(f, y)) + geom_point() + scale_x_continuous(limits = c(0, 3))),
  list(name = "inf_only_limits_x",
       spec = "ggplot(df[c(1, 3), ], aes(f, y)) + geom_point() + scale_x_continuous(limits = c(0, 3))",
       plot = ggplot(df[c(1, 3), ], aes(f, y)) + geom_point() + scale_x_continuous(limits = c(0, 3)),
       rows = c(1, 3)),
  list(name = "inf_mixed_y_limits",
       spec = "ggplot(df, aes(z, f)) + geom_point() + scale_y_continuous(limits = c(-1, 4))",
       plot = ggplot(df, aes(z, f)) + geom_point() + scale_y_continuous(limits = c(-1, 4)))
)

write_cases_fixture(cases, row = "scale_continuous_args", data_file = "data.csv",
                    out_file = file.path(here, "expected.json"))
