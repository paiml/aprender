# Row 1 fixture: geom_point() + default scale_x_continuous()/scale_y_continuous().
# Regenerate (needs R + ggplot2 4.0.3; nothing else):
#   Rscript crates/aprender-viz/fixtures/ggplot2/geom_point/generate.R
# Writes expected.json beside this file. Commit both; tests never run R.
here <- (function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
})()
source(file.path(here, "..", "oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

# Every value is exactly representable in binary32, because aprender-viz's DataFrame stores f32:
# the comparison then measures the scale math, not a data round-trip.
df <- utils::read.csv(file.path(here, "data.csv"))

p <- ggplot(df, aes(x, y)) + geom_point()

write_fixture(
  p,
  row = "geom_point+scale_continuous",
  spec = "ggplot(df, aes(x, y)) + geom_point()",
  data_file = "data.csv",
  out_file = file.path(here, "expected.json")
)
