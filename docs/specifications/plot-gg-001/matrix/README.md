# PLOT-GG-001 conformance matrix data

Spec: `docs/specifications/PLOT-GG-001.md`. Reference: ggplot2 4.0.3 on R 4.6.2. These files are the matrix as read at origin/main `53fa483e0` on 2026-11-12; 21 rows were re-read at `54555d69b` after PR #6910 and PR #6913 (see the spec, section 6). `LINE` numbers in `aprender_symbol` are valid at the commit the row was read; the symbol name is the key.

## Files

- `<family>.tsv`: one row per exported ggplot2 name in the family.
- `<family>.exports.txt`: the raw export list the rows were derived from. Row count equals export count.

Families: coords, core, facets, geoms, guides, positions, scales, stats, theme-elements, themes.

The `scales` file also holds the `scales-helper` family: its 9 rows have `family` = `scales-helper` (`derive`, `dup_axis`, `expand_limits`, `expand_scale`, `expansion`, `lims`, `sec_axis`, `xlim`, `ylim`). They are the non-`scale_` helpers that the scales file selects, so that file has 131 `scales` rows and 9 `scales-helper` rows (140 in all).

Not rows: `ggproto-classes.txt` (the 128 exported ggproto classes, left out on purpose), `other-exports.txt` (the 150 exported functions not yet in the matrix) and `ggplot2-4.0.3-exports.txt` (all 643 exports). Their counts add up with the rows, see "Recount".

## Columns (tab-separated, header on line 1)

| Column | Meaning |
|---|---|
| `family` | The family above; `scales-helper` for the 9 helper rows in `scales.tsv`. |
| `ggplot2_name` | The exported ggplot2 name (a theme element name for theme-elements). |
| `status` | `implemented`, `partial` or `absent` (definitions in the spec, section 6; `implemented` needs fixture + test + contract + falsifier + the public `layout()` path). |
| `aprender_symbol` | `crates/aprender-viz/src/...:LINE Symbol` or `-`. |
| `proving_test` | `path::test_fn` that checks the computed values, or `-`. |
| `notes` | The main gap, at most 15 words. The optional `[drawn: framebuffer=... svg=... wasm=...]` tag is not counted in the 15 words: the separate drawn state per target (`smoke` = render test checking success only, `-` = not drawn). A row without the tag claims nothing about drawing. Only the 3 implemented rows carry it. |

## Deriving each export list (R 4.6.2, ggplot2 4.0.3)

    Rscript -e 'library(ggplot2); cat(sort(grep("^geom_", getNamespaceExports("ggplot2"), value=TRUE)), sep="\n")'

Change the pattern per family. Prefix-only families: `^geom_` (geoms, 53), `^stat_` (stats, 36), `^position_` (positions, 8), `^coord_` (coords, 12). Families that add named helpers to a prefix:

| Family | Exports selected |
|---|---|
| scales | `^scale_` plus `derive`, `dup_axis`, `expand_limits`, `expand_scale`, `expansion`, `lims`, `sec_axis`, `xlim`, `ylim` |
| facets | `^facet_` plus `as_labeller`, `label_*`, `labeller`, `vars` |
| themes | `^theme_` and `^element_` plus `%+replace%`, `margin`, `margin_auto`, `margin_part`, `rel`, `theme` |
| guides | `^guide_` and `^draw_key_` plus `guides` |
| core | the named top-level functions (the other top-level functions are in other-exports.txt): `aes*`, `after_scale`, `after_stat`, `annotate`, `annotation_*`, `autolayer`, `autoplot`, `borders`, `fortify`, `ggplot*`, `ggsave`, `ggtitle`, `labs`, `last_plot`, `layer`, `qplot`, `quickplot`, `stage`, `xlab`, `ylab` |

The theme-elements family is the element tree:

    Rscript -e 'library(ggplot2); cat(sort(names(get_element_tree())), sep="\n")'

Check: for each family, `wc -l <family>.exports.txt` equals the number of data rows in `<family>.tsv` (the file's line count minus the header).

## Recount

Run from this directory. `LC_ALL=C` fixes the sort order. The first command needs R 4.6.2 with ggplot2 4.0.3.

    export LC_ALL=C
    Rscript -e 'suppressMessages(library(ggplot2)); ex <- sort(getNamespaceExports("ggplot2"), method="radix"); ns <- asNamespace("ggplot2"); writeLines(ex, "ggplot2-4.0.3-exports.txt"); writeLines(ex[vapply(ex, function(n) inherits(get(n, ns), "ggproto"), logical(1))], "ggproto-classes.txt")'
    for f in coords core facets geoms guides positions scales stats themes; do cat $f.exports.txt; done | sort > /tmp/in-rows.txt
    comm -23 ggplot2-4.0.3-exports.txt /tmp/in-rows.txt | comm -23 - ggproto-classes.txt | diff - other-exports.txt && echo other-exports ok
    wc -l ggplot2-4.0.3-exports.txt /tmp/in-rows.txt ggproto-classes.txt other-exports.txt

Expected: 643 exports = 365 in rows + 128 ggproto classes + 150 other exports; 525 rows = 365 + 160 theme elements. Status totals:

    for f in *.tsv; do awk -F'\t' -v f="${f%.tsv}" 'FNR>1{n[$3]++;t++} END{print f,t,n["implemented"]+0,n["partial"]+0,n["absent"]+0}' "$f"; done

Per-file checks: every line has 6 fields (`awk -F'\t' 'NF!=6' *.tsv` prints nothing), `ggplot2_name` is unique across all files (`cut -f2 *.tsv | grep -v '^ggplot2_name$' | sort | uniq -d` prints nothing), and each family's `wc -l <family>.exports.txt` equals its data rows.
