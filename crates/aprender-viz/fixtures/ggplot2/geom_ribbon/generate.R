# Row geom_ribbon fixtures: geom_ribbon(), default scales. Layer data is sorted by group then x.
# Regenerate (needs R + ggplot2 4.0.3; nothing else), from the repository root:
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_ribbon/generate.R
# Writes expected_<case>.json beside each <case>.csv. Commit both; tests never run R.
here <- (function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
})()
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "marks_oracle.R"))

plain <- function(df) ggplot(df, aes(x, ymin = lo, ymax = hi)) + geom_ribbon()
grouped <- function(df) ggplot(df, aes(x, ymin = lo, ymax = hi, group = g)) + geom_ribbon()
cases <- list(
  list(name = "basic_unsorted", spec = "ggplot(df, aes(x, ymin = lo, ymax = hi)) + geom_ribbon()", plot = plain),
  list(name = "groups_text", spec = "ggplot(df, aes(x, ymin = lo, ymax = hi, group = g)) + geom_ribbon()", plot = grouped),
  list(name = "groups_num", spec = "ggplot(df, aes(x, ymin = lo, ymax = hi, group = g)) + geom_ribbon()", plot = grouped),
  list(name = "na_ymin", spec = "ggplot(df, aes(x, ymin = lo, ymax = hi)) + geom_ribbon()", plot = plain),
  list(name = "na_x", spec = "ggplot(df, aes(x, ymin = lo, ymax = hi)) + geom_ribbon()", plot = plain),
  list(name = "dup_x", spec = "ggplot(df, aes(x, ymin = lo, ymax = hi)) + geom_ribbon()", plot = plain),
  list(name = "reversed", spec = "ggplot(df, aes(x, ymin = lo, ymax = hi)) + geom_ribbon()", plot = plain),
  list(name = "two_row_groups", spec = "ggplot(df, aes(x, ymin = lo, ymax = hi, group = g)) + geom_ribbon()", plot = grouped),
  list(name = "style", spec = "ggplot(df, aes(x, ymin = lo, ymax = hi)) + geom_ribbon(fill = 'steelblue', colour = 'red', alpha = 0.5)",
       plot = function(df) ggplot(df, aes(x, ymin = lo, ymax = hi)) + geom_ribbon(fill = "steelblue", colour = "red", alpha = 0.5))
)
run_cases(here, "geom_ribbon", cases)
