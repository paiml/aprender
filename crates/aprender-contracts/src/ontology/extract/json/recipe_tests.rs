use std::fs;
use std::path::Path;

use super::*;
use crate::ontology::rdf::{iri, Term};
use crate::ontology::shapes::expand;

/// The D1 example recipe (aprender#3769), with its sha256 pinned.
const RECIPE: &str = "schema: cookbook-recipe/v1\nid: run-greedy\nargv: [apr, run, \"{model:main}\", --prompt, \"2+2?\"]\nmodels:\n  main:\n    role: model\n    ref: hf://Qwen/Q/q.gguf\n    sha256: aaaa\nexpect:\n  exit: 0\n  stdout_contains: [\"4\"]\nmin_apr: 0.69.1\nhosts: [lambda, gx10]\nbackends: [cuda]\n";

fn vocab() -> Vocabulary {
    Vocabulary {
        prefix: "recipe".into(),
        root_class: "recipe:Recipe".into(),
        nested: Vec::new(),
    }
}

fn none() -> Receipts {
    Receipts {
        dir: PathBuf::new(),
        current: None,
    }
}

fn sha(text: &str) -> String {
    hex(&Sha256::digest(text.as_bytes()))
}

/// `<root>/receipts/CURRENT = 0.69.1-abc123`, and one receipt per host with `verdict` and `recipe_sha256`.
fn receipts(root: &Path, hosts: &[&str], verdict: &str, apr_version: &str, recipe_sha: &str) {
    let cur = "0.69.1-abc123";
    fs::create_dir_all(root.join("receipts")).unwrap();
    fs::write(root.join("receipts/CURRENT"), format!("{cur}\n")).unwrap();
    for h in hosts {
        let d = root.join("receipts").join(cur).join(h);
        fs::create_dir_all(&d).unwrap();
        let body = serde_json::json!({"verdict": verdict, "apr_version": apr_version, "recipe_sha256": recipe_sha});
        fs::write(d.join("run-greedy.json"), body.to_string()).unwrap();
    }
}

#[test]
fn the_row_is_the_cookbook_derivations_row_key_for_key() {
    let got = row(RECIPE.as_bytes(), &none()).unwrap();
    let want = serde_json::json!({
        "id": "run-greedy", "schema": "cookbook-recipe/v1", "verb": "run", "recipe_sha256": sha(RECIPE),
        "argv0_is_apr": true, "placeholders_resolve": true,
        "model_sha256": ["aaaa"], "model_ref": ["hf://Qwen/Q/q.gguf"],
        "expect_exit": 0, "has_output_check": true, "min_apr": "0.69.1",
        "host": ["lambda", "gx10"], "backend": ["cuda"], "receipt_current_pass": false,
    });
    assert_eq!(got, want);
}

#[test]
fn the_extracted_graph_equals_the_json_extraction_of_the_derived_jsonl_line() {
    // What the cookbook's shape judged before R3: the script's JSONL line through extract:json. The recipe
    // extractor must hand the shape the same triples, or the closed shape's verdicts change under the move.
    let line = row(RECIPE.as_bytes(), &none()).unwrap().to_string();
    let mut via_jsonl = Graph::new();
    super::super::extract_text(
        &mut via_jsonl,
        "cookbook-recipe-v1",
        "r.jsonl",
        &line,
        &vocab(),
    )
    .unwrap();
    let mut via_recipe = Graph::new();
    let n = extract_recipes(
        &mut via_recipe,
        "cookbook-recipe-v1",
        "recipes/apr",
        &[(PathBuf::from("r.yaml"), RECIPE.as_bytes().to_vec())],
        &none(),
        &vocab(),
    )
    .unwrap();
    assert_eq!(n, 1);
    assert_eq!(via_recipe.to_ntriples(), via_jsonl.to_ntriples());
}

/// Case table: one planted defect per derived fact, each turning exactly its field.
#[test]
fn each_planted_defect_turns_its_own_field() {
    let cases: &[(&str, &str, &str, serde_json::Value)] = &[
        (
            "argv[0] is not apr",
            "argv: [apr,",
            "argv: [llama-cli,",
            serde_json::json!(false),
        ),
        (
            "a placeholder names an undeclared slot",
            "{model:main}",
            "{model:draft}",
            serde_json::json!(false),
        ),
        (
            "a placeholder half-matched is no placeholder",
            "\"{model:main}\"",
            "\"x{model:main}\"",
            serde_json::json!(false),
        ),
        (
            "expect judges only the exit code",
            "  stdout_contains: [\"4\"]\n",
            "",
            serde_json::json!(false),
        ),
    ];
    let field = [
        "argv0_is_apr",
        "placeholders_resolve",
        "placeholders_resolve",
        "has_output_check",
    ];
    for ((what, from, to, want), key) in cases.iter().zip(field) {
        let planted = RECIPE.replacen(from, to, 1);
        assert_ne!(planted, RECIPE, "{what}: the plant did not apply");
        let got = row(planted.as_bytes(), &none()).unwrap();
        assert_eq!(&got[key], want, "{what}: {key}");
        let clean = row(RECIPE.as_bytes(), &none()).unwrap();
        for k in clean
            .as_object()
            .unwrap()
            .keys()
            .filter(|k| *k != key && *k != "recipe_sha256")
        {
            assert_eq!(got[k], clean[k], "{what}: only {key} may move, but {k} did");
        }
    }
}

#[test]
fn an_empty_or_absent_field_emits_no_triple_so_min_count_catches_it() {
    let planted = RECIPE
        .replace("hosts: [lambda, gx10]\n", "hosts: []\n")
        .replace("min_apr: 0.69.1\n", "");
    let got = row(planted.as_bytes(), &none()).unwrap();
    assert!(got["host"].is_null() && got["min_apr"].is_null());
    let mut g = Graph::new();
    extract_recipes(
        &mut g,
        "c",
        "d",
        &[(PathBuf::from("r.yaml"), planted.into_bytes())],
        &none(),
        &vocab(),
    )
    .unwrap();
    let s = iri("recipe", "c.1");
    assert!(g.objects(&s, &expand("recipe:host")).is_empty());
    assert!(g.objects(&s, &expand("recipe:min_apr")).is_empty());
}

#[test]
fn a_receipt_passes_only_when_every_host_passed_the_current_binary_on_these_bytes() {
    let s = sha(RECIPE);
    let v = "apr 0.69.1 (abc123)";
    // (case, hosts with a receipt, verdict, version, recipe sha, want)
    let table: &[(&str, &[&str], &str, &str, &str, bool)] = &[
        (
            "both hosts PASS the current binary",
            &["lambda", "gx10"],
            "PASS",
            v,
            &s,
            true,
        ),
        (
            "one declared host has no receipt",
            &["lambda"],
            "PASS",
            v,
            &s,
            false,
        ),
        ("a host FAILed", &["lambda", "gx10"], "FAIL", v, &s, false),
        (
            "a receipt from another binary",
            &["lambda", "gx10"],
            "PASS",
            "apr 0.69.0 (fff000)",
            &s,
            false,
        ),
        (
            "the recipe was edited after it ran",
            &["lambda", "gx10"],
            "PASS",
            v,
            &sha("edited"),
            false,
        ),
    ];
    for (what, hosts, verdict, version, recipe_sha, want) in table {
        let d = tempfile::tempdir().unwrap();
        receipts(d.path(), hosts, verdict, version, recipe_sha);
        let got = row(RECIPE.as_bytes(), &Receipts::at(&d.path().join("receipts"))).unwrap();
        assert_eq!(
            got["receipt_current_pass"],
            serde_json::json!(*want),
            "{what}"
        );
    }
    let d = tempfile::tempdir().unwrap();
    receipts(d.path(), &["lambda", "gx10"], "PASS", v, &s);
    fs::write(d.path().join("receipts/CURRENT"), "nosha\n").unwrap();
    let got = row(RECIPE.as_bytes(), &Receipts::at(&d.path().join("receipts"))).unwrap();
    assert_eq!(
        got["receipt_current_pass"],
        serde_json::json!(false),
        "a CURRENT with no sha judges nothing"
    );
}

fn contract(r#ref: &str) -> serde_yaml::Value {
    serde_yaml::from_str(&format!(
        "entity: {{type: recipe, ref: {ref}}}\nvocabulary: {{prefix: recipe, root_class: \"recipe:Recipe\"}}\n",
        r#ref = r#ref
    ))
    .unwrap()
}

#[test]
fn the_walk_reads_every_yaml_below_ref_in_component_order_and_nothing_else() {
    let d = tempfile::tempdir().unwrap();
    let r = d.path().join("recipes/apr");
    for (p, id) in [("run/a.yaml", "a"), ("run-x/b.yaml", "b"), ("c.yaml", "c")] {
        fs::create_dir_all(r.join(p).parent().unwrap()).unwrap();
        fs::write(
            r.join(p),
            RECIPE.replace("id: run-greedy", &format!("id: {id}")),
        )
        .unwrap();
    }
    fs::write(r.join("README.md"), "not a recipe").unwrap();
    let mut g = Graph::new();
    let n = extract_into(&mut g, "c", &contract("recipes/apr"), d.path()).unwrap();
    assert_eq!(n, 3);
    // c.yaml < run/a.yaml < run-x/b.yaml by component; a string sort would put run-x before run/.
    let ids: Vec<String> = (1..=3)
        .map(|i| {
            let t = g.objects(&iri("recipe", &format!("c.{i}")), &expand("recipe:id"))[0];
            t.as_literal().unwrap().0.to_string()
        })
        .collect();
    assert_eq!(ids, ["c", "a", "b"]);
}

#[test]
fn an_empty_corpus_or_a_non_recipe_is_refused_never_graded() {
    let d = tempfile::tempdir().unwrap();
    fs::create_dir_all(d.path().join("recipes/apr")).unwrap();
    let mut g = Graph::new();
    let empty = extract_into(&mut g, "c", &contract("recipes/apr"), d.path());
    assert!(
        matches!(empty, Err(ExtractError::Recipe { .. })),
        "{empty:?}"
    );
    let absent = extract_into(&mut g, "c", &contract("recipes/none"), d.path());
    assert!(
        matches!(absent, Err(ExtractError::RefUnreadable { .. })),
        "{absent:?}"
    );
    fs::write(d.path().join("recipes/apr/ok.yaml"), RECIPE).unwrap();
    fs::write(d.path().join("recipes/apr/torn.yaml"), "- just\n- a list\n").unwrap();
    let torn = extract_into(&mut g, "c", &contract("recipes/apr"), d.path());
    assert!(
        matches!(&torn, Err(ExtractError::Recipe { path, .. }) if path.ends_with("torn.yaml")),
        "{torn:?}"
    );
    assert!(
        g.is_empty(),
        "a refused corpus adds no node of its good recipes"
    );
}

#[test]
fn only_a_recipe_contract_applies() {
    assert!(applies(&contract("recipes/apr")));
    let json: serde_yaml::Value =
        serde_yaml::from_str("entity: {type: json, ref: x.jsonl}").unwrap();
    assert!(!applies(&json));
}

#[test]
fn the_positive_control_fires() {
    assert!(positive_control());
    // …and it is not vacuous: the resolving recipe really does extract true.
    let mut g = Graph::new();
    extract_recipes(
        &mut g,
        "c",
        "d",
        &[(PathBuf::from("r.yaml"), RECIPE.as_bytes().to_vec())],
        &none(),
        &vocab(),
    )
    .unwrap();
    let t = g.objects(
        &iri("recipe", "c.1"),
        &expand("recipe:placeholders_resolve"),
    );
    assert_eq!(t, [&Term::boolean(true)]);
}
