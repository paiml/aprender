# PLOT-GG-001 conformance matrix data

Spec: `docs/specifications/PLOT-GG-001.md`. Reference: ggplot2 4.0.3 on R 4.6.2. These files are the matrix as read at origin/main `53fa483e0` on 2026-11-12. They are copied unchanged.

## Files

- `<family>.tsv`: one row per exported ggplot2 name in the family.
- `<family>.exports.txt`: the raw export list the rows were derived from. Row count equals export count.

Families: coords, core, facets, geoms, guides, positions, scales, stats, theme-elements, themes.

## Columns (tab-separated, header on line 1)

| Column | Meaning |
|---|---|
| `family` | The family above. |
| `ggplot2_name` | The exported ggplot2 name (a theme element name for theme-elements). |
| `status` | `implemented`, `partial` or `absent` (definitions in the spec, section 6). |
| `aprender_symbol` | `crates/aprender-viz/src/...:LINE Symbol` or `-`. |
| `proving_test` | `path::test_fn` that checks the computed values, or `-`. |
| `notes` | The main gap, at most 15 words. |

## Deriving each export list (R 4.6.2, ggplot2 4.0.3)

    Rscript -e 'library(ggplot2); cat(sort(grep("^geom_", getNamespaceExports("ggplot2"), value=TRUE)), sep="\n")'

Change the pattern per family. Prefix-only families: `^geom_` (geoms, 53), `^stat_` (stats, 36), `^position_` (positions, 8), `^coord_` (coords, 12). Families that add named helpers to a prefix:

| Family | Exports selected |
|---|---|
| scales | `^scale_` plus `derive`, `dup_axis`, `expand_limits`, `expand_scale`, `expansion`, `lims`, `sec_axis`, `xlim`, `ylim` |
| facets | `^facet_` plus `as_labeller`, `label_*`, `labeller`, `vars` |
| themes | `^theme_` and `^element_` plus `%+replace%`, `margin`, `margin_auto`, `margin_part`, `rel`, `theme` |
| guides | `^guide_` and `^draw_key_` plus `guides` |
| core | the remaining top-level functions: `aes*`, `after_scale`, `after_stat`, `annotate`, `annotation_*`, `autolayer`, `autoplot`, `borders`, `fortify`, `ggplot*`, `ggsave`, `ggtitle`, `labs`, `last_plot`, `layer`, `qplot`, `quickplot`, `stage`, `xlab`, `ylab` |

The theme-elements family is the element tree:

    Rscript -e 'library(ggplot2); cat(sort(names(get_element_tree())), sep="\n")'

Check: for each family, `wc -l <family>.exports.txt` equals the number of data rows in `<family>.tsv` (the file's line count minus the header).
