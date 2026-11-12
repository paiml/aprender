# Row: geom_step (PLOT-GG-001, paiml/aprender#6909). direction hv, vh, mid is applied at draw
# time, so the fixtures record the stair vertices GeomStep$draw_panel feeds the path renderer.
# Regenerate (R + ggplot2 4.0.3 only, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_step/generate.R
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[1])))
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "lines_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))

df <- read_case(here, "data.csv")
dsg <- read_case(here, "data_single_group.csv")
row <- "geom_step"
out <- function(case) file.path(here, paste0("expected_", case, ".json"))
w <- function(case, plot, spec, dir) write_lines_fixture(plot, row, case, spec, out(case), step_direction = dir)

w("hv", ggplot(df, aes(x, y)) + geom_step(), "ggplot(df, aes(x, y)) + geom_step()", "hv")
w("vh", ggplot(df, aes(x, y)) + geom_step(direction = "vh"), "ggplot(df, aes(x, y)) + geom_step(direction = 'vh')", "vh")
w("mid", ggplot(df, aes(x, y)) + geom_step(direction = "mid"), "ggplot(df, aes(x, y)) + geom_step(direction = 'mid')", "mid")
w("group_hv", ggplot(df, aes(x, y, group = gn)) + geom_step(), "ggplot(df, aes(x, y, group = gn)) + geom_step()", "hv")
w("group_mid", ggplot(df, aes(x, y, group = g)) + geom_step(direction = "mid"), "ggplot(df, aes(x, y, group = g)) + geom_step(direction = 'mid')", "mid")
w("facet_vh", ggplot(df, aes(x, y)) + geom_step(direction = "vh") + facet_wrap(~f), "ggplot(df, aes(x, y)) + geom_step(direction = 'vh') + facet_wrap(~f)", "vh")
w("single_row_group", ggplot(dsg, aes(x, y, group = gn)) + geom_step(), "ggplot(dsg, aes(x, y, group = gn)) + geom_step()", "hv")
w("colour_alpha", ggplot(df, aes(x, y)) + geom_step(colour = "blue", alpha = 0.25), "ggplot(df, aes(x, y)) + geom_step(colour = 'blue', alpha = 0.25)", "hv")
