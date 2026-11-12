# Row geom_tile fixtures: geom_tile(), default scales. Tile width/height = resolution of x / y.
# Regenerate (needs R + ggplot2 4.0.3; nothing else), from the repository root:
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_tile/generate.R
# Writes expected_<case>.json beside each <case>.csv. Commit both; tests never run R.
here <- (function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
})()
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "marks_oracle.R"))

plain <- function(df) ggplot(df, aes(x, y)) + geom_tile()
cases <- list(
  list(name = "grid_regular", spec = "ggplot(df, aes(x, y)) + geom_tile()", plot = plain),
  list(name = "fractional_res", spec = "ggplot(df, aes(x, y)) + geom_tile()", plot = plain),
  list(name = "unsorted_dup", spec = "ggplot(df, aes(x, y)) + geom_tile()", plot = plain),
  list(name = "na_cells", spec = "ggplot(df, aes(x, y)) + geom_tile()", plot = plain),
  list(name = "single_cell", spec = "ggplot(df, aes(x, y)) + geom_tile()", plot = plain),
  list(name = "same_x", spec = "ggplot(df, aes(x, y)) + geom_tile()", plot = plain),
  list(name = "groups_text", spec = "ggplot(df, aes(x, y, group = g)) + geom_tile()",
       plot = function(df) ggplot(df, aes(x, y, group = g)) + geom_tile()),
  list(name = "style", spec = "ggplot(df, aes(x, y)) + geom_tile(fill = 'steelblue', colour = 'red', alpha = 0.5)",
       plot = function(df) ggplot(df, aes(x, y)) + geom_tile(fill = "steelblue", colour = "red", alpha = 0.5)),
  list(name = "wide_gaps", spec = "ggplot(df, aes(x, y)) + geom_tile()", plot = plain)
)
run_cases(here, "geom_tile", cases)
