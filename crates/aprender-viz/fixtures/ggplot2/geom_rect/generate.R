# Row geom_rect fixtures: geom_rect(), default scales. The four bounds pass through unchanged.
# Regenerate (needs R + ggplot2 4.0.3; nothing else), from the repository root:
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_rect/generate.R
# Writes expected_<case>.json beside each <case>.csv. Commit both; tests never run R.
here <- (function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
})()
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "marks_oracle.R"))

plain <- function(df) ggplot(df, aes(xmin = x0, xmax = x1, ymin = y0, ymax = y1)) + geom_rect()
s <- "ggplot(df, aes(xmin = x0, xmax = x1, ymin = y0, ymax = y1)) + geom_rect()"
cases <- list(
  list(name = "basic", spec = s, plot = plain),
  list(name = "reversed", spec = s, plot = plain),
  list(name = "na_edge", spec = s, plot = plain),
  list(name = "duplicate_rows", spec = s, plot = plain),
  list(name = "groups_text", spec = sub("y1)", "y1, group = g)", s, fixed = TRUE),
       plot = function(df) ggplot(df, aes(xmin = x0, xmax = x1, ymin = y0, ymax = y1, group = g)) + geom_rect()),
  list(name = "zero_width", spec = s, plot = plain),
  list(name = "negative", spec = s, plot = plain),
  list(name = "style", spec = sub("geom_rect()", "geom_rect(fill = 'steelblue', colour = 'red', alpha = 0.5)", s, fixed = TRUE),
       plot = function(df) ggplot(df, aes(xmin = x0, xmax = x1, ymin = y0, ymax = y1)) +
         geom_rect(fill = "steelblue", colour = "red", alpha = 0.5)),
  list(name = "single_row", spec = s, plot = plain)
)
run_cases(here, "geom_rect", cases)
