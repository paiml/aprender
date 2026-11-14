# Rows: scale_x_log10 + scale_y_log10 (transform, limits, expansion in log space, log breaks/labels).
# Regenerate: Rscript crates/aprender-viz/fixtures/ggplot2/scale_log10/generate.R
here <- (function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
})()
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "oracle_scales.R"))
suppressPackageStartupMessages(library(ggplot2))
df <- utils::read.csv(file.path(here, "data.csv"))

cases <- list(
  list(name = "wide_default", spec = "ggplot(df, aes(x, y)) + geom_point() + scale_x_log10() + scale_y_log10()",
       plot = ggplot(df, aes(x, y)) + geom_point() + scale_x_log10() + scale_y_log10()),
  list(name = "narrow_x_sci_y", spec = "ggplot(df, aes(u, v)) + geom_point() + scale_x_log10() + scale_y_log10()",
       plot = ggplot(df, aes(u, v)) + geom_point() + scale_x_log10() + scale_y_log10()),
  list(name = "limits_no_expand",
       spec = "ggplot(df, aes(x, y)) + geom_point() + scale_x_log10(limits = c(1, 10000), expand = expansion(0)) + scale_y_log10()",
       plot = ggplot(df, aes(x, y)) + geom_point() +
         scale_x_log10(limits = c(1, 10000), expand = expansion(0)) + scale_y_log10()),
  list(name = "breaks_labels",
       spec = "ggplot(df, aes(x, y)) + geom_point() + scale_x_log10(breaks = c(1, 10, 100, 1000), labels = c('one', 'ten', 'hundred', 'thousand')) + scale_y_log10()",
       plot = ggplot(df, aes(x, y)) + geom_point() +
         scale_x_log10(breaks = c(1, 10, 100, 1000), labels = c("one", "ten", "hundred", "thousand")) +
         scale_y_log10()),
  # Non-positive values on a log axis become NA (with a warning); the row stays in the layer data.
  list(name = "nonpositive_x_na",
       spec = "ggplot(df, aes(w, y)) + geom_point() + scale_x_log10() + scale_y_log10()",
       plot = ggplot(df, aes(w, y)) + geom_point() + scale_x_log10() + scale_y_log10()),
  # Zero-width log range: +-0.05 in log10 space around the single value.
  list(name = "single_point_log",
       spec = "ggplot(df[4, ], aes(x, y)) + geom_point() + scale_x_log10() + scale_y_log10()",
       plot = ggplot(df[4, ], aes(x, y)) + geom_point() + scale_x_log10() + scale_y_log10(), rows = 4),
  list(name = "all_equal_log",
       spec = "ggplot(df[c(4, 4, 4), ], aes(x, y)) + geom_point() + scale_x_log10() + scale_y_log10()",
       plot = ggplot(df[c(4, 4, 4), ], aes(x, y)) + geom_point() + scale_x_log10() + scale_y_log10(),
       rows = c(4, 4, 4)),
  list(name = "limits_equal_log",
       spec = "ggplot(df[1:4, ], aes(x, y)) + geom_point() + scale_x_log10(limits = c(10, 10))",
       plot = ggplot(df[1:4, ], aes(x, y)) + geom_point() + scale_x_log10(limits = c(10, 10)),
       rows = 1:4),

  # --- R's log10(0) is -Inf and censor(only.finite = TRUE) leaves infinite values alone (review of #6916).
  # Training ignores non-finite values, so the range comes from the finite ones only.
  list(name = "zero_x",
       spec = "ggplot(df[1:3, ], aes(z, y)) + geom_point() + scale_x_log10()",
       plot = ggplot(df[1:3, ], aes(z, y)) + geom_point() + scale_x_log10(), rows = 1:3),
  list(name = "zero_x_limits",
       spec = "ggplot(df[1:3, ], aes(z, y)) + geom_point() + scale_x_log10(limits = c(1, 100))",
       plot = ggplot(df[1:3, ], aes(z, y)) + geom_point() + scale_x_log10(limits = c(1, 100)), rows = 1:3),
  # Only a zero: nothing finite to train on, so explicit limits are needed (without them R gives an
  # infinite range and this crate an Err; see ggplot2_row3_log_axis_without_positive_data_is_an_error).
  list(name = "zero_only_limits",
       spec = "ggplot(df[1, ], aes(z, y)) + geom_point() + scale_x_log10(limits = c(1, 100))",
       plot = ggplot(df[1, ], aes(z, y)) + geom_point() + scale_x_log10(limits = c(1, 100)), rows = 1),
  # -Inf, 0, 1, Inf, 10, 100: log10(-Inf) is NA, log10(0) is -Inf, log10(Inf) is Inf.
  list(name = "inf_mixed_x",
       spec = "ggplot(df, aes(i, y)) + geom_point() + scale_x_log10()",
       plot = ggplot(df, aes(i, y)) + geom_point() + scale_x_log10()),
  list(name = "inf_mixed_x_limits",
       spec = "ggplot(df, aes(i, y)) + geom_point() + scale_x_log10(limits = c(1, 100))",
       plot = ggplot(df, aes(i, y)) + geom_point() + scale_x_log10(limits = c(1, 100))),
  # Tiny positive (2^-100, exact in f32), then 0, then a negative, all in one column.
  list(name = "tiny_zero_negative_x",
       spec = "ggplot(df, aes(t, y)) + geom_point() + scale_x_log10()",
       plot = ggplot(df, aes(t, y)) + geom_point() + scale_x_log10()),
  list(name = "tiny_single_x",
       spec = "ggplot(df[1, ], aes(t, y)) + geom_point() + scale_x_log10()",
       plot = ggplot(df[1, ], aes(t, y)) + geom_point() + scale_x_log10(), rows = 1),
  # Zero-width range AT 0 in log space (log10(1) = 0): zero_range() must short-circuit on lo == hi.
  list(name = "single_one_log",
       spec = "ggplot(df[2, ], aes(x, y)) + geom_point() + scale_x_log10()",
       plot = ggplot(df[2, ], aes(x, y)) + geom_point() + scale_x_log10(), rows = 2),
  # All data negative, explicit limits: every row is NA, the panel is the limits (nothing to train on
  # is not an error when limits are given).
  list(name = "all_negative_limits",
       spec = "ggplot(df[c(1, 5), ], aes(w, y)) + geom_point() + scale_x_log10(limits = c(1, 100))",
       plot = ggplot(df[c(1, 5), ], aes(w, y)) + geom_point() + scale_x_log10(limits = c(1, 100)),
       rows = c(1, 5)),
  # An explicit break at 0 maps to -Inf, outside the panel: censored.
  list(name = "break_at_zero",
       spec = "ggplot(df, aes(x, y)) + geom_point() + scale_x_log10(breaks = c(0, 1, 10, 100))",
       plot = ggplot(df, aes(x, y)) + geom_point() + scale_x_log10(breaks = c(0, 1, 10, 100)))
)

write_cases_fixture(cases, row = "scale_x_log10+scale_y_log10", data_file = "data.csv",
                    out_file = file.path(here, "expected.json"))
