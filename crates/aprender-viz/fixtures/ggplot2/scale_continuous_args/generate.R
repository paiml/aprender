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
       plot = pt + scale_y_continuous(expand = expansion(add = 2)))
)

write_cases_fixture(cases, row = "scale_continuous_args", data_file = "data.csv",
                    out_file = file.path(here, "expected.json"))
