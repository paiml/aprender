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
       rows = 1:4)

)

write_cases_fixture(cases, row = "scale_x_log10+scale_y_log10", data_file = "data.csv",
                    out_file = file.path(here, "expected.json"))
