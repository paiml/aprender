# Row: geom_smooth (PLOT-GG-001, paiml/aprender#6909).
# Regenerate (R + ggplot2 4.0.3 only, C collation):
#   LC_ALL=C Rscript crates/aprender-viz/fixtures/ggplot2/geom_smooth/generate.R
# Writes expected_<case>.json beside this file. Commit them; tests never run R.
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
source(file.path(here, "..", "oracle.R"))
source(file.path(here, "..", "g3_oracle.R"))
suppressPackageStartupMessages(library(ggplot2))
row <- "geom_smooth"
out <- function(case) file.path(here, paste0("expected_", case, ".json"))
w <- function(case, plot, spec) g3_write(plot, row, case, spec, out(case))
df <- g3_read(here, "data.csv"); dfa <- g3_read(here, "data_facet.csv"); dna <- g3_read(here, "data_na.csv")
dtwo <- g3_read(here, "data_two.csv"); dthree <- g3_read(here, "data_three.csv"); dl <- g3_read(here, "data_layer.csv")
dpt <- g3_read(here, "data_pt.csv"); dg2 <- g3_read(here, "data_g2.csv")
lm_ <- function(se) geom_smooth(method = "lm", formula = y ~ x, se = se)
d1 <- g3_read(here, "data_g1.csv")
w("se_false", ggplot(d1, aes(x, y)) + lm_(FALSE), "ggplot(d1, aes(x, y)) + geom_smooth(method = 'lm', se = FALSE)")
w("se_true", ggplot(d1, aes(x, y)) + lm_(TRUE), "ggplot(d1, aes(x, y)) + geom_smooth(method = 'lm', se = TRUE)")
w("group_se", ggplot(df, aes(x, y, group = gn)) + lm_(TRUE), "ggplot(df, aes(x, y, group = gn)) + geom_smooth(method = 'lm')")
w("group_text_nose", ggplot(df, aes(x, y, group = g)) + lm_(FALSE), "ggplot(df, aes(x, y, group = g)) + geom_smooth(method = 'lm', se = FALSE)")
w("group_two_row_nose", ggplot(dg2, aes(x, y, group = gn)) + lm_(FALSE), "ggplot(dg2, aes(x, y, group = gn)) + geom_smooth(method = 'lm', se = FALSE)")
w("facet_se", ggplot(dfa, aes(x, y)) + lm_(TRUE) + facet_wrap(~f), "ggplot(dfa, aes(x, y)) + geom_smooth(method = 'lm') + facet_wrap(~f)")
w("na_se", ggplot(dna, aes(x, y)) + lm_(TRUE), "ggplot(dna, aes(x, y)) + geom_smooth(method = 'lm')  # NA rows removed")
w("colour_alpha", ggplot(d1, aes(x, y)) + geom_smooth(method = "lm", formula = y ~ x, colour = "red", alpha = 0.5), "geom_smooth(method = 'lm', colour = 'red', alpha = 0.5)")
w("layer_data", ggplot(dpt, aes(x, y)) + geom_point() + geom_smooth(data = dl, method = "lm", formula = y ~ x), "geom_point() + geom_smooth(data = dl, method = 'lm')")
w("three_rows_se", ggplot(dthree, aes(x, y)) + lm_(TRUE), "ggplot(dthree, aes(x, y)) + geom_smooth(method = 'lm')")
w("two_rows_nose", ggplot(dtwo, aes(x, y)) + lm_(FALSE), "ggplot(dtwo, aes(x, y)) + geom_smooth(method = 'lm', se = FALSE)")

# qt(0.975, df) for the Student-t quantile of the band: a table for df = 1..30 and large df.
dfs <- c(1:30, 40, 50, 60, 80, 100, 120, 200, 500, 1000, 5000, 100000)
qt_fixture <- list(
  row = row, case = "qt_table", spec = "qt(0.975, df)", collation = Sys.getlocale("LC_COLLATE"),
  versions = versions_json(),
  df = I(as.numeric(dfs)), qt = I(qt(0.975, dfs))
)
writeLines(to_json(qt_fixture), out("qt_table"), useBytes = TRUE)
