//! ONT-001 §5 ONT-4c, B.8 — `extract:csv`: a CSV dataset becomes a `csv:Dataset` node.
//!
//! In-house, quote-aware reader of exactly what the vocabulary needs (R-13): the header row (`csv:header`, raw;
//! one `csv:column` per name), the data-row count (`csv:rows`, an integer), one inferred `csv:dtype` per column
//! (`int` when every value parses as an integer, else `float` when every value parses as a number, else
//! `string`), and `csv:sha256` over the file's bytes. `csv:producer` is the contract's `entity.properties.producer`
//! (`resolves: path` — it must exist under the repo root, v4.16). A data row whose column count differs from the
//! header's is refused naming the file and the row, never a node (the `pc_extract.csv` positive control, R-3).

use std::path::Path;

use sha2::{Digest, Sha256};

use crate::ontology::rdf::{iri, Graph, Term, ONT_BASE, RDF_TYPE};

use super::claims::{self, DocStats};
use super::gguf::{hex, ExtractError};
use super::pv_contract::scalar;

/// The `csv:` vocabulary: `https://ont.paiml.dev/v1alpha1/csv/<name>`.
#[must_use]
pub fn csv(name: &str) -> String {
    format!("{ONT_BASE}csv/{name}")
}

/// Split one CSV record, honouring double quotes (`""` is an escaped quote inside a quoted field).
#[must_use]
pub fn split_record(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if quoted && chars.peek() == Some(&'"') => {
                cur.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    out.push(cur);
    out
}

/// What the file holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Table {
    pub header: String,
    pub columns: Vec<String>,
    pub rows: u64,
    pub dtypes: Vec<&'static str>,
}

fn dtype_of(values: &[&str]) -> &'static str {
    if values.is_empty() {
        return "string";
    }
    if values.iter().all(|v| v.trim().parse::<i64>().is_ok()) {
        "int"
    } else if values.iter().all(|v| v.trim().parse::<f64>().is_ok()) {
        "float"
    } else {
        "string"
    }
}

/// Parse the header and every data row; a row with the wrong column count is refused naming it.
pub fn parse(file: &str, text: &str) -> Result<Table, ExtractError> {
    let err = |what: String| ExtractError {
        file: file.to_string(),
        what,
    };
    let mut lines = text.lines().map(|l| l.strip_suffix('\r').unwrap_or(l));
    let header = lines
        .next()
        .ok_or_else(|| err("empty: no header row".into()))?;
    let columns = split_record(header);
    let mut cells: Vec<Vec<String>> = vec![Vec::new(); columns.len()];
    let mut rows = 0u64;
    for (i, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        let rec = split_record(line);
        if rec.len() != columns.len() {
            return Err(err(format!(
                "row {} has {} column(s), the header has {}",
                i + 2,
                rec.len(),
                columns.len()
            )));
        }
        for (c, v) in cells.iter_mut().zip(rec) {
            c.push(v);
        }
        rows += 1;
    }
    let dtypes = cells
        .iter()
        .map(|c| dtype_of(&c.iter().map(String::as_str).collect::<Vec<_>>()))
        .collect();
    Ok(Table {
        header: header.to_string(),
        columns,
        rows,
        dtypes,
    })
}

/// One CSV into `g`.
pub fn emit_file(
    g: &mut Graph,
    stem: &str,
    rel: &str,
    bytes: &[u8],
    producer: Option<&str>,
    exists: &dyn Fn(&str) -> bool,
) -> Result<(), ExtractError> {
    let text = std::str::from_utf8(bytes).map_err(|e| ExtractError {
        file: rel.to_string(),
        what: format!("not UTF-8: {e}"),
    })?;
    let t = parse(rel, text)?;
    let s = iri("csv", stem);
    g.insert(s.clone(), RDF_TYPE, Term::iri(csv("Dataset")));
    g.insert(s.clone(), csv("header"), Term::string(&t.header));
    for c in &t.columns {
        g.insert(s.clone(), csv("column"), Term::string(c));
    }
    g.insert(s.clone(), csv("rows"), Term::integer(t.rows));
    for d in &t.dtypes {
        g.insert(s.clone(), csv("dtype"), Term::string(*d));
    }
    g.insert(
        s.clone(),
        csv("sha256"),
        Term::string(hex(&Sha256::digest(bytes))),
    );
    if let Some(p) = producer {
        if !exists(p) {
            return Err(ExtractError {
                file: rel.to_string(),
                what: format!("csv:producer unresolved: {p}"),
            });
        }
        g.insert(s, csv("producer"), Term::string(p));
    }
    Ok(())
}

/// Every `entity: {type: csv}` contract under `contract_dir`.
pub fn extract(contract_dir: &Path, g: &mut Graph) -> DocStats {
    let root = super::repo_root(contract_dir);
    let exists = |p: &str| root.join(p).exists();
    let mut stats = DocStats::default();
    for (stem, r, doc) in claims::entity_docs(contract_dir, "csv") {
        let producer = scalar(
            doc.get("entity")
                .and_then(|e| e.get("properties"))
                .and_then(|p| p.get("producer")),
        );
        match std::fs::read(root.join(&r)) {
            Ok(bytes) => match emit_file(g, &stem, &r, &bytes, producer.as_deref(), &exists) {
                Ok(()) => stats.files_read += 1,
                Err(e) => stats.errors.push(e),
            },
            Err(e) => stats.refuse(&r, format!("unreadable: {e}")),
        }
    }
    stats
}

/// The positive control (R-3): a data row with one column too many is refused, and the same table without it is
/// not — every run, in memory.
#[must_use]
pub fn positive_control() -> bool {
    let good = b"a,b\n1,2\n";
    let bad = b"a,b\n1,2\n3,4,5\n";
    let yes = |_: &str| true;
    let refused = emit_file(&mut Graph::new(), "__pc__", "pc.csv", bad, None, &yes)
        .is_err_and(|e| e.what.contains("row 3 has 3 column(s), the header has 2"));
    refused && emit_file(&mut Graph::new(), "__pc__", "pc.csv", good, None, &yes).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_table_parses_with_its_dtypes_and_quoted_fields() {
        let t = parse("t.csv", "x,y,\"z,w\"\n1,1.5,\"a,\"\"b\"\n2,3,c\r\n").expect("parses");
        assert_eq!(t.columns, vec!["x", "y", "z,w"]);
        assert_eq!(t.rows, 2);
        assert_eq!(t.dtypes, vec!["int", "float", "string"]);
        assert_eq!(split_record("a,\"b,\"\"c\""), vec!["a", "b,\"c"]);
    }

    #[test]
    fn a_column_mismatch_and_an_unresolved_producer_are_refused_naming_the_file() {
        let no = |_: &str| false;
        let e = emit_file(&mut Graph::new(), "t", "d/t.csv", b"a,b\n1\n", None, &no)
            .expect_err("refused");
        assert_eq!(e.file, "d/t.csv");
        assert!(e.what.contains("row 2 has 1 column(s)"), "{e}");
        let e = emit_file(
            &mut Graph::new(),
            "t",
            "d/t.csv",
            b"a\n1\n",
            Some("gen.sh"),
            &no,
        )
        .expect_err("refused");
        assert_eq!(e.what, "csv:producer unresolved: gen.sh");
    }

    #[test]
    fn the_node_carries_the_b8_properties() {
        let yes = |_: &str| true;
        let mut g = Graph::new();
        emit_file(
            &mut g,
            "t",
            "t.csv",
            b"feature1,feature2,label\n0.1,2,1\n",
            Some("p.sh"),
            &yes,
        )
        .expect("emitted");
        let s = iri("csv", "t");
        for (p, n) in [
            ("column", 3),
            ("header", 1),
            ("rows", 1),
            ("sha256", 1),
            ("producer", 1),
        ] {
            assert_eq!(g.objects(&s, &csv(p)).len(), n, "{p}");
        }
    }

    #[test]
    fn the_positive_control_fires() {
        assert!(positive_control());
    }
}
