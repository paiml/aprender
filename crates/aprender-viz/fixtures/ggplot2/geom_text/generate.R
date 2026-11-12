# Row geom_text fixtures: geom_text(), default scales. Layout only: positions, scale ranges and
# the label strings as passthrough; fonts, sizes in pixels and text extents are not modelled.
# Regenerate (needs R + ggplot2 4.0.3; nothing else), from the repository root:
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_text/generate.R
# Writes expected_<case>.json beside each <case>.csv. Commit both; tests never run R.
here <- (function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
})()
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "marks_oracle.R"))

plain <- function(df) ggplot(df, aes(x, y, label = lab)) + geom_text()
s <- "ggplot(df, aes(x, y, label = lab)) + geom_text()"
cases <- list(
  list(name = "basic", spec = s, plot = plain),
  list(name = "unsorted_dup", spec = s, plot = plain),
  list(name = "na_xy", spec = s, plot = plain),
  list(name = "groups_text", spec = "ggplot(df, aes(x, y, label = lab, group = g)) + geom_text()",
       plot = function(df) ggplot(df, aes(x, y, label = lab, group = g)) + geom_text()),
  list(name = "style", spec = "ggplot(df, aes(x, y, label = lab)) + geom_text(colour = 'red', alpha = 0.5, size = 5)",
       plot = function(df) ggplot(df, aes(x, y, label = lab)) + geom_text(colour = "red", alpha = 0.5, size = 5)),
  list(name = "odd_labels", spec = s, plot = plain),
  list(name = "single_row", spec = s, plot = plain),
  list(name = "negative", spec = s, plot = plain),
  list(name = "na_label", spec = s, plot = plain)
)
run_cases(here, "geom_text", cases)
