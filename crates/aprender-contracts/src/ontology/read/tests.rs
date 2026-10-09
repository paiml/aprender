//! The reader's tests: Turtle and N-Triples forms, the §14.5 controls (blank relabel, `"01"` vs `"1"`, escapes,
//! `é`, `@fr`/`@en-GB`), refusals at a line, and the R-15 falsifiers — nothing converts a read graph into pv's
//! own graph, nothing outside this module names it, and pv reads its own `contracts.nt` back with no blank node.

use std::path::{Path, PathBuf};

use super::{read, InputGraph, ReadError, Syntax};

const BASE: &str = "http://example.org/doc";
const XSD: &str = "http://www.w3.org/2001/XMLSchema#";
const RDF: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";

fn graph(text: &str, syntax: Syntax) -> InputGraph {
    read(text, syntax, Some(BASE)).expect("the document reads")
}

fn ttl(text: &str) -> String {
    graph(text, Syntax::Turtle).to_ntriples()
}

fn nt(text: &str) -> String {
    graph(text, Syntax::NTriples).to_ntriples()
}

fn refused(text: &str, syntax: Syntax) -> ReadError {
    read(text, syntax, None).expect_err("the document is refused")
}

/// Lines, each ended by LF, in the order given (the caller gives them in byte order).
fn lines(ls: &[&str]) -> String {
    ls.iter().map(|l| format!("{l}\n")).collect()
}

#[test]
fn prefixes_a_semicolons_and_commas() {
    let doc = "@prefix ex: <http://example.org/> .\n\
               PREFIX foaf: <http://xmlns.com/foaf/0.1/>\n\
               ex:alice a foaf:Person ; foaf:knows ex:bob , ex:carol ; .\n";
    let want = lines(&[
        &format!("<http://example.org/alice> <{RDF}type> <http://xmlns.com/foaf/0.1/Person> ."),
        "<http://example.org/alice> <http://xmlns.com/foaf/0.1/knows> <http://example.org/bob> .",
        "<http://example.org/alice> <http://xmlns.com/foaf/0.1/knows> <http://example.org/carol> .",
    ]);
    assert_eq!(ttl(doc), want);
}

#[test]
fn relative_iris_resolve_against_the_base_and_base_moves() {
    let doc = "<> <p> <#frag> .\n@base <http://other.example/dir/> .\n<a> <../q> \"x\" .\n";
    let want = lines(&[
        "<http://example.org/doc> <http://example.org/p> <http://example.org/doc#frag> .",
        "<http://other.example/dir/a> <http://other.example/q> \"x\" .",
    ]);
    assert_eq!(ttl(doc), want);
}

#[test]
fn a_relative_iri_with_no_base_is_refused() {
    let e = refused("<a> <http://x/p> <http://x/o> .\n", Syntax::Turtle);
    assert_eq!((e.line, e.col), (1, 1));
    assert!(e.message.contains("no base"), "{e}");
}

/// §14.5 blank relabel: the file's labels do not reach the output, so renaming them changes nothing.
#[test]
fn blank_labels_are_relabelled_in_order_of_meeting() {
    let a = ttl("_:zed <http://x/p> _:alpha .\n_:alpha <http://x/p> _:zed .\n");
    let b = ttl("_:q <http://x/p> _:r .\n_:r <http://x/p> _:q .\n");
    assert_eq!(
        a,
        lines(&["_:b0 <http://x/p> _:b1 .", "_:b1 <http://x/p> _:b0 ."])
    );
    assert_eq!(a, b);
}

#[test]
fn anon_property_lists_and_collections() {
    let doc = "@prefix : <http://x/> .\n\
               [] :p [ :q 1 ] .\n\
               :s :list ( :a \"b\" ) ; :empty () .\n\
               [ :r \"solo\" ] .\n";
    let int = format!("<{XSD}integer>");
    let want = lines(&[
        &format!("<http://x/s> <http://x/empty> <{RDF}nil> ."),
        "<http://x/s> <http://x/list> _:b2 .",
        "_:b0 <http://x/p> _:b1 .",
        &format!("_:b1 <http://x/q> \"1\"^^{int} ."),
        &format!("_:b2 <{RDF}first> <http://x/a> ."),
        &format!("_:b2 <{RDF}rest> _:b3 ."),
        &format!("_:b3 <{RDF}first> \"b\" ."),
        &format!("_:b3 <{RDF}rest> <{RDF}nil> ."),
        "_:b4 <http://x/r> \"solo\" .",
    ]);
    assert_eq!(ttl(doc), want);
}

/// §14.5 `"01"` vs `"1"`: a number keeps its lexical form, and the type follows its shape.
#[test]
fn numbers_keep_their_lexical_form() {
    let doc = "@prefix : <http://x/> .\n\
               :s :p 01 , 1 , \"01\"^^<http://www.w3.org/2001/XMLSchema#integer> , -1.50 , .5 , 1e3 , 2.E-1 , \
               true , false .\n:t :p 7.";
    let g = graph(doc, Syntax::Turtle);
    let out = g.to_ntriples();
    for (lexical, dt) in [
        ("01", "integer"),
        ("1", "integer"),
        ("-1.50", "decimal"),
        (".5", "decimal"),
        ("1e3", "double"),
        ("2.E-1", "double"),
        ("true", "boolean"),
        ("false", "boolean"),
        ("7", "integer"),
    ] {
        let lit = format!("\"{lexical}\"^^<{XSD}{dt}> .\n");
        assert!(out.contains(&lit), "missing {lit}in\n{out}");
    }
    assert_eq!(g.len(), 9, "\"01\" written twice is one triple:\n{out}");
}

/// §14.5 escapes, `é`, `@fr` and `@en-GB`, and the long and single-quoted forms.
#[test]
fn escapes_unicode_and_language_tags() {
    let doc = r##"@prefix : <http://x/> .
:s :p "tab\there \"q\" \\ \u00E9 é" , "chat"@fr , "colour"@en-GB , 'single' , """long
"quoted" line""" , '''x''' .
<http://x/caf\u00E9> :p :o .
"##;
    let out = ttl(doc);
    for want in [
        r#"<http://x/s> <http://x/p> "tab\there \"q\" \\ é é" ."#,
        r#"<http://x/s> <http://x/p> "chat"@fr ."#,
        r#"<http://x/s> <http://x/p> "colour"@en-gb ."#,
        r#"<http://x/s> <http://x/p> "single" ."#,
        r#"<http://x/s> <http://x/p> "long\n\"quoted\" line" ."#,
        r#"<http://x/s> <http://x/p> "x" ."#,
        "<http://x/café> <http://x/p> <http://x/o> .",
    ] {
        assert!(
            out.contains(&format!("{want}\n")),
            "missing {want}\nin\n{out}"
        );
    }
}

#[test]
fn the_writer_escapes_control_characters_canonically() {
    let out = nt("<http://x/s> <http://x/p> \"a\\u0007b\\fc\\bd\" .\n");
    assert_eq!(out, "<http://x/s> <http://x/p> \"a\\u0007b\\fc\\bd\" .\n");
}

#[test]
fn prefixed_names_take_escapes_percent_and_inner_dots() {
    let doc = "@prefix : <http://x/> .\n:a.b :p\\~q :c%20d .\n:e :p :f.\n";
    let want = lines(&[
        "<http://x/a.b> <http://x/p~q> <http://x/c%20d> .",
        "<http://x/e> <http://x/p> <http://x/f> .",
    ]);
    assert_eq!(ttl(doc), want);
}

#[test]
fn two_reads_are_byte_identical_and_a_repeated_triple_is_one() {
    let doc = "@prefix : <http://x/> .\n:s :p [ :q ( 1 2 ) ] , _:k .\n:s :p _:k .\n";
    assert_eq!(ttl(doc), ttl(doc));
    assert_eq!(graph(doc, Syntax::Turtle).len(), 7);
}

/// §14.2 R-INPUT: a malformed file is refused at a line.
#[test]
fn malformed_turtle_is_refused_at_its_line() {
    let cases = [
        ("<http://x/s> <http://x/p> <http://x/o>", 1, "expected '.'"),
        (
            "@prefix : <http://x/> .\n:s :p :o .\n:s :p \"open .\n",
            3,
            "line break",
        ),
        (
            "@prefix : <http://x/> .\nx:s :p :o .\n",
            2,
            "undefined prefix",
        ),
        ("@foo <http://x/> .\n", 1, "unknown directive"),
        (
            "<http://x/s> <http://x/p> \"a\"@ .\n",
            1,
            "empty language tag",
        ),
        (
            "\n\n<http://x/s> <http://x/p> <http://x/a b> .\n",
            3,
            "not allowed in an IRI",
        ),
        ("<http://x/s> <http://x/p> \"\\q\" .\n", 1, "bad escape"),
        ("<http://x/s> <http://x/p> 1e .\n", 1, "expected '.'"),
    ];
    for (doc, line, fragment) in cases {
        let e = refused(doc, Syntax::Turtle);
        assert_eq!(e.line, line, "{doc:?}: {e}");
        assert!(e.message.contains(fragment), "{doc:?}: {e}");
    }
}

#[test]
fn ntriples_reads_comments_blank_lines_and_every_term() {
    let doc = "# head\n<http://x/s> <http://x/p> _:a . # tail\n\n_:a <http://x/q> \"v\"@EN .\n\
               _:a <http://x/r> \"1\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n";
    let want = lines(&[
        "<http://x/s> <http://x/p> _:b0 .",
        "_:b0 <http://x/q> \"v\"@en .",
        &format!("_:b0 <http://x/r> \"1\"^^<{XSD}integer> ."),
    ]);
    assert_eq!(nt(doc), want);
}

#[test]
fn ntriples_refuses_what_only_turtle_allows() {
    let cases = [
        ("<a> <http://x/p> <http://x/o> .\n", "absolute IRIs only"),
        ("<http://x/s> <http://x/p> 'o' .\n", "expected an object"),
        ("@prefix : <http://x/> .\n", "expected a subject"),
        (
            "<http://x/s> <http://x/p> \"\"\"o\"\"\" .\n",
            "expected '.'",
        ),
        (
            "<http://x/s> <http://x/p> <http://x/o>\n.\n",
            "expected '.'",
        ),
        (
            "<http://x/s> <http://x/p> <http://x/o> . <http://x/s> <http://x/p> <http://x/o> .\n",
            "end of the line",
        ),
    ];
    for (doc, fragment) in cases {
        let e = refused(doc, Syntax::NTriples);
        assert_eq!(e.line, 1, "{doc:?}: {e}");
        assert!(e.message.contains(fragment), "{doc:?}: {e}");
    }
}

#[test]
fn syntax_by_extension_and_by_name() {
    assert_eq!(Syntax::for_path(Path::new("a/b.ttl")), Some(Syntax::Turtle));
    assert_eq!(
        Syntax::for_path(Path::new("a/b.nt")),
        Some(Syntax::NTriples)
    );
    assert_eq!(Syntax::for_path(Path::new("a/b.rdf")), None);
    assert_eq!(Syntax::from_name("ntriples"), Some(Syntax::NTriples));
    assert_eq!(Syntax::from_name("ttl"), Some(Syntax::Turtle));
    assert_eq!(Syntax::from_name("xml"), None);
}

// ---- R-15 falsifiers (quorum q-s3reader Q1) ----

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every `.rs` file under `dir`, recursively, except under `skip`.
fn rs_files(dir: &Path, skip: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).expect("the source directory lists");
    for entry in entries {
        let path = entry.expect("a directory entry").path();
        if path.starts_with(skip) {
            continue;
        }
        if path.is_dir() {
            rs_files(&path, skip, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The code lines of `source` (not `//` comments) that contain one of `needles`.
fn hits(source: &str, needles: &[&str]) -> Vec<String> {
    source
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .filter(|l| needles.iter().any(|n| l.contains(n)))
        .map(str::to_string)
        .collect()
}

fn hits_in(files: &[PathBuf], needles: &[&str]) -> Vec<String> {
    files
        .iter()
        .flat_map(|f| {
            let text = std::fs::read_to_string(f).expect("a source file reads");
            hits(&text, needles)
                .into_iter()
                .map(move |l| format!("{}: {l}", f.display()))
        })
        .collect()
}

/// What would turn a read graph into pv's graph: naming its types, or a conversion.
const INTO_PV_GRAPH: [&str; 5] = [
    "rdf::Graph",
    "rdf::Term",
    "rdf::Triple",
    "impl From<",
    "impl Into<",
];

/// What would bring the reader into another module: its types or its path.
const NAMES_THE_READER: [&str; 5] = [
    "InputTerm",
    "InputTriple",
    "InputGraph",
    "ontology::read",
    "read::read",
];

#[test]
fn the_scans_catch_what_they_look_for() {
    assert_eq!(
        hits("use crate::ontology::rdf::Graph;\n", &INTO_PV_GRAPH).len(),
        1
    );
    assert_eq!(
        hits("impl From<InputGraph> for X {}\n", &INTO_PV_GRAPH).len(),
        1
    );
    assert!(hits("//! unlike rdf::Term\n", &INTO_PV_GRAPH).is_empty());
    assert_eq!(hits("let g: InputGraph = x;\n", &NAMES_THE_READER).len(), 1);
}

/// Q1 falsifier: no conversion from a read graph into [`super::super::rdf`]'s graph.
#[test]
fn the_reader_never_converts_into_pv_graph() {
    let dir = crate_dir().join("src/ontology/read");
    let mut files = Vec::new();
    rs_files(&dir, &dir.join("tests.rs"), &mut files);
    assert!(files.len() >= 5, "scanned only {files:?}");
    let found = hits_in(&files, &INTO_PV_GRAPH);
    assert!(
        found.is_empty(),
        "the reader reaches pv's graph:\n{}",
        found.join("\n")
    );
}

/// Q1 falsifier: no path from the reader into `contracts.nt` or its hash. Nothing in this crate outside the
/// reader names it, nor anything in `pv` but the `pv ontology` command that runs it.
#[test]
fn nothing_outside_the_reader_names_it() {
    let lib = crate_dir().join("src");
    let cli = crate_dir().join("../aprender-contracts-cli/src");
    let mut files = Vec::new();
    rs_files(&lib, &lib.join("ontology/read"), &mut files);
    rs_files(&cli, &cli.join("commands/ontology.rs"), &mut files);
    assert!(files.len() > 100, "scanned only {} files", files.len());
    let found = hits_in(&files, &NAMES_THE_READER);
    assert!(found.is_empty(), "the reader leaks:\n{}", found.join("\n"));
}

/// Q1 falsifier: pv's own `contracts.nt` reads back with no blank node, triple for triple. pv writes
/// `^^xsd:string` and the reader writes the simple literal (one term, RDF 1.1), so that suffix is dropped
/// before comparing. A planted `_:x` line is counted, so the zero is not vacuous.
#[test]
fn contracts_nt_reads_back_with_no_blank_node() {
    let path = crate_dir().join("../../contracts/contracts.nt");
    let text = std::fs::read_to_string(&path).expect("contracts/contracts.nt reads");
    let g = read(&text, Syntax::NTriples, None).expect("pv's own N-Triples reads");
    assert_eq!(g.blank_nodes(), 0);
    let mut want: Vec<String> = text
        .lines()
        .map(|l| format!("{}\n", l.replace(&format!("^^<{XSD}string>"), "")))
        .collect();
    want.sort();
    assert!(
        want.len() > 1000,
        "contracts.nt has only {} lines",
        want.len()
    );
    assert_eq!(g.len(), want.len());
    assert!(
        g.to_ntriples() == want.concat(),
        "the reader and pv's writer disagree on contracts.nt"
    );
    let planted = format!("{text}_:x <http://x/p> <http://x/o> .\n");
    let p = read(&planted, Syntax::NTriples, None).expect("the planted file reads");
    assert_eq!(p.blank_nodes(), 1);
}
