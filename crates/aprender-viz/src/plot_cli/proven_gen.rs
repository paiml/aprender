// The proven-row generator, `include!`d by build.rs (which does the file I/O and writes
// `proven_rows.rs`) and by the plot_cli unit tests (which feed it contracts and oracles the
// build never sees). Plain `//` comments: an inner doc comment cannot be `include!`d.

#[derive(serde::Deserialize)]
struct Contract {
    #[serde(default)]
    ggplot2_row: Option<Row>,
    #[serde(default)]
    falsification_tests: Vec<serde::de::IgnoredAny>,
}

/// The `ggplot2_row:` block. Flat scalars only, so that it can become the `properties` of a
/// pv ontology entity once the `ggplot2-row` entity type is declared.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Row {
    #[serde(rename = "ref")]
    name: String,
    geom: String,
    stat: String,
    position: String,
    scale_x: String,
    scale_y: String,
    coord: String,
    facet: String,
    theme: String,
    aes: String,
    oracle: String,
}

/// The `ggplot2_row:` block of contract file `name`, or `None` when it has none.
fn parse_row(name: &str, yaml: &str) -> Result<Option<Row>, String> {
    let c: Contract = serde_saphyr::from_str(yaml).map_err(|e| format!("{name}: {e}"))?;
    match c.ggplot2_row {
        Some(_) if c.falsification_tests.is_empty() => {
            Err(format!("{name}: a ggplot2_row needs at least one falsification test"))
        }
        row => Ok(row),
    }
}

/// The class R's `ggplot_build()` gives a layer of `geom`: `point` is `GeomPoint`.
fn geom_class(geom: &str) -> String {
    let camel: String = geom
        .split('_')
        .flat_map(|w| w.chars().take(1).flat_map(char::to_uppercase).chain(w.chars().skip(1)))
        .collect();
    format!("Geom{camel}")
}

/// What the R call `spec` names after `key` (`flip` in `coord_flip()` after `coord_`), or
/// ggplot2's default `default` when it names none.
fn named<'a>(spec: &'a str, key: &str, default: &'a str) -> &'a str {
    spec.find(key).map_or(default, |i| {
        let rest = &spec[i + key.len()..];
        &rest[..rest.find(|c: char| !c.is_ascii_alphanumeric() && c != '_').unwrap_or(rest.len())]
    })
}

/// The block's oracle backs it: at least one `expected*.json` (`(file, text)`), and each draws
/// one layer of the block's geom, as a row proves one layer, from an R call (`spec`) whose stat,
/// position, scales, coord, facet and theme are the block's. The defaults are geom_point's: a
/// row whose geom has another default stat is refused until this table knows it.
fn check_oracle(name: &str, row: &Row, oracles: &[(String, String)]) -> Result<(), String> {
    if oracles.is_empty() {
        return Err(format!("{name}: oracle {} has no expected*.json", row.oracle));
    }
    let want = geom_class(&row.geom);
    for (file, text) in oracles {
        let v: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("{name}: {file}: {e}"))?;
        let layers = v["layers"].as_array().map_or(&[][..], Vec::as_slice);
        let got: Vec<&str> = layers.iter().map(|l| l["geom"].as_str().unwrap_or("?")).collect();
        if got != [want.as_str()] {
            return Err(format!(
                "{name}: oracle {file} draws layers {got:?}; the block declares one {want} layer"
            ));
        }
        let spec = v["spec"].as_str().ok_or_else(|| format!("{name}: oracle {file} has no spec"))?;
        for (dim, key, default, declared) in [
            ("stat", "stat = '", "identity", &row.stat),
            ("position", "position = '", "identity", &row.position),
            ("scale_x", "scale_x_", "continuous", &row.scale_x),
            ("scale_y", "scale_y_", "continuous", &row.scale_y),
            ("coord", "coord_", "cartesian", &row.coord),
            ("facet", "facet_", "null", &row.facet),
            ("theme", "theme_", "grey", &row.theme),
        ] {
            let got = named(spec, key, default);
            if got != declared {
                return Err(format!("{name}: oracle {file} has {dim} {got}; the block declares {declared}"));
            }
        }
    }
    Ok(())
}

/// `(file name, text)` of each `expected*.json` in an oracle directory, by name.
fn expected_jsons(dir: &Path) -> Result<Vec<(String, String)>, String> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Ok(Vec::new()) };
    let mut out = Vec::new();
    for path in entries.flatten().map(|e| e.path()) {
        let n = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if n.starts_with("expected") && n.ends_with(".json") {
            println!("cargo:rerun-if-changed={}", path.display());
            out.push((n.clone(), std::fs::read_to_string(&path).map_err(|e| format!("{n}: {e}"))?));
        }
    }
    out.sort();
    Ok(out)
}

/// Every `ggplot2_row:` block of the `ggplot2-*.yaml` contracts in `dir`, keyed by contract id,
/// each checked against its oracle (a directory relative to `workspace`).
fn read_rows(dir: &Path, workspace: &Path) -> Result<BTreeMap<String, Row>, String> {
    let mut rows = BTreeMap::new();
    for path in std::fs::read_dir(dir).map_err(|e| format!("read {dir:?}: {e}"))?.flatten() {
        let path = path.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        let Some(id) = name.strip_suffix(".yaml").filter(|n| n.starts_with("ggplot2-")) else {
            continue;
        };
        println!("cargo:rerun-if-changed={}", path.display());
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{name}: {e}"))?;
        let Some(row) = parse_row(name, &text)? else { continue };
        check_oracle(name, &row, &expected_jsons(&workspace.join(&row.oracle))?)?;
        rows.insert(id.to_string(), row);
    }
    Ok(rows)
}

/// The table at `table` is the contracts in `dir` rendered: a table that differs is an error,
/// or with `regen` is rewritten. Without `dir` (a packaged crate) the committed table is kept.
fn sync(dir: &Path, workspace: &Path, table: &Path, regen: bool) -> Result<(), String> {
    if !dir.is_dir() {
        return Ok(());
    }
    println!("cargo:rerun-if-changed={}", dir.display());
    let text = render(&read_rows(dir, workspace)?);
    if std::fs::read_to_string(table).unwrap_or_default() == text {
        return Ok(());
    }
    if regen {
        return std::fs::write(table, text).map_err(|e| format!("write {table:?}: {e}"));
    }
    Err(format!(
        "{table:?} differs from the ggplot2_row blocks in {dir:?}; regenerate it with \
         APRENDER_VIZ_REGEN_PROVEN=1 cargo build -p aprender-viz and commit it"
    ))
}

fn words(s: &str) -> String {
    let quoted: Vec<String> = s.split_whitespace().map(|w| format!("{w:?}")).collect();
    format!("&[{}]", quoted.join(", "))
}

fn render(rows: &BTreeMap<String, Row>) -> String {
    let mut out = String::from(
        "// @generated by crates/aprender-viz/build.rs from the `ggplot2_row:` blocks of\n\
         // contracts/trueno-viz/ggplot2-*.yaml. Do not edit: change the contract, then run\n\
         // APRENDER_VIZ_REGEN_PROVEN=1 cargo build -p aprender-viz\n\
         &[\n",
    );
    for (id, r) in rows {
        out.push_str(&format!(
            "    ProvenRow {{\n        contract: {id:?},\n        name: {:?},\n        \
             geom: {:?},\n        stat: {:?},\n        position: {:?},\n        \
             scales: &[(\"x\", {:?}), (\"y\", {:?})],\n        coord: {:?},\n        \
             facet: {:?},\n        theme: {:?},\n        aes: {},\n    }},\n",
            r.name,
            r.geom,
            r.stat,
            r.position,
            r.scale_x,
            r.scale_y,
            r.coord,
            r.facet,
            r.theme,
            words(&r.aes),
        ));
    }
    out.push_str("]\n");
    out
}
