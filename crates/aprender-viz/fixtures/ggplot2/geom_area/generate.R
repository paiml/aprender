# Row geom_area fixtures: geom_area() = stat_align + position_stack, default scales.
# Regenerate (needs R + ggplot2 4.0.3; nothing else), from the repository root:
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_area/generate.R
# Writes expected_<case>.json beside each <case>.csv. Commit both; tests never run R.
here <- (function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
})()
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "marks_oracle.R"))

# Every value is exactly representable in binary32 (aprender-viz stores f32).
cases <- list(
  list(name = "groups_text", spec = "ggplot(df, aes(x, y, group = g)) + geom_area()",
       plot = function(df) ggplot(df, aes(x, y, group = g)) + geom_area()),
  list(name = "groups_num", spec = "ggplot(df, aes(x, y, group = g)) + geom_area()",
       plot = function(df) ggplot(df, aes(x, y, group = g)) + geom_area()),
  list(name = "single_unsorted", spec = "ggplot(df, aes(x, y)) + geom_area()",
       plot = function(df) ggplot(df, aes(x, y)) + geom_area()),
  list(name = "single_dup_x", spec = "ggplot(df, aes(x, y)) + geom_area()",
       plot = function(df) ggplot(df, aes(x, y)) + geom_area()),
  list(name = "na_rows", spec = "ggplot(df, aes(x, y, group = g)) + geom_area()",
       plot = function(df) ggplot(df, aes(x, y, group = g)) + geom_area()),
  list(name = "two_row_groups", spec = "ggplot(df, aes(x, y, group = g)) + geom_area()",
       plot = function(df) ggplot(df, aes(x, y, group = g)) + geom_area()),
  list(name = "zero_y", spec = "ggplot(df, aes(x, y, group = g)) + geom_area()",
       plot = function(df) ggplot(df, aes(x, y, group = g)) + geom_area()),
  list(name = "style", spec = "ggplot(df, aes(x, y)) + geom_area(fill = 'steelblue', colour = 'red', alpha = 0.5)",
       plot = function(df) ggplot(df, aes(x, y)) + geom_area(fill = "steelblue", colour = "red", alpha = 0.5)),
  list(name = "tiny_gap", spec = "ggplot(df, aes(x, y, group = g)) + geom_area()",
       plot = function(df) ggplot(df, aes(x, y, group = g)) + geom_area()),
  list(name = "dup_in_group_multi", spec = "ggplot(df, aes(x, y, group = g)) + geom_area()",
       plot = function(df) ggplot(df, aes(x, y, group = g)) + geom_area())
)
run_cases(here, "geom_area", cases)
