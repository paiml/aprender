# Shared scales of plots that mix the filled marks and text (geom_area, geom_ribbon, geom_tile,
# geom_rect, geom_text) with each other and with geom_point: the only companions layout() lays out
# beside them (PLOT-GG-001, paiml/aprender#6909). Every other companion, and any stat other than
# identity on a geom_point beside them, is refused with a typed error.
# Regenerate (needs R + ggplot2 4.0.3; nothing else), from the repository root:
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/marks_companions/generate.R
# Writes expected_<case>.json from wide.csv. Commit both; tests never run R.
here <- (function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", arg[1])))
})()
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "marks_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

# The five layers, each with the mapping the Rust test gives it.
marks <- list(
  area = list(spec = "geom_area(aes(x, y, group = g))", layer = function() geom_area(aes(x, y, group = g))),
  ribbon = list(spec = "geom_ribbon(aes(x = x, ymin = lo, ymax = hi, group = g))",
                layer = function() geom_ribbon(aes(x = x, ymin = lo, ymax = hi, group = g))),
  tile = list(spec = "geom_tile(aes(x, y, group = g))", layer = function() geom_tile(aes(x, y, group = g))),
  rect = list(spec = "geom_rect(aes(xmin = x0, xmax = x1, ymin = y0, ymax = y1, group = g))",
              layer = function() geom_rect(aes(xmin = x0, xmax = x1, ymin = y0, ymax = y1, group = g))),
  text = list(spec = "geom_text(aes(x, y, label = lab, group = g))",
              layer = function() geom_text(aes(x, y, label = lab, group = g)))
)
# The point reaches past every marks layer on both axes, so it must widen both scales.
point <- list(spec = "geom_point(aes(px, py))", layer = function() geom_point(aes(px, py)))
# NA cells: a continuous scale trains on the non-missing values of each axis on its own, so a row
# with an NA y still trains x. (An infinite point position is refused beside these layers.)
point_na <- list(spec = "geom_point(aes(qx, qy))", layer = function() geom_point(aes(qx, qy)))

df <- read_case(here, "wide.csv")
write_case <- function(name, parts) {
  spec <- paste(c("ggplot(df)", vapply(parts, function(p) p$spec, "")), collapse = " + ")
  p <- Reduce(`+`, lapply(parts, function(p) p$layer()), ggplot(df))
  write_marks_fixture(p, "marks_companions", name, spec,
                      file.path(here, paste0("expected_", name, ".json")))
}
names5 <- names(marks)
for (n in names5) write_case(paste0("point_", n), list(point, marks[[n]]))
for (n in names5) write_case(paste0(n, "_point"), list(marks[[n]], point))
for (i in seq_along(names5)) for (j in seq_along(names5)) if (i < j) {
  write_case(paste0(names5[i], "_", names5[j]), list(marks[[names5[i]]], marks[[names5[j]]]))
}
write_case("point_na_area", list(point_na, marks$area))
write_case("point_na_text", list(point_na, marks$text))
write_case("area_text_point", list(marks$area, marks$text, point))
write_case("rect_tile_ribbon", list(marks$rect, marks$tile, marks$ribbon))
