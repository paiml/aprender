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
         scale_y_log10())
)

write_cases_fixture(cases, row = "scale_x_log10+scale_y_log10", data_file = "data.csv",
                    out_file = file.path(here, "expected.json"))
