use super::*;

fn esc(file: &str, decl: &str, kind: &str) -> Escape {
    Escape {
        file: file.into(),
        line: 1,
        kind: kind.into(),
        decl: decl.into(),
    }
}

fn allowed(file: &str, decl: &str, kind: &str, confirmed_by: &str) -> Allowed {
    Allowed {
        file: file.into(),
        decl: decl.into(),
        kind: kind.into(),
        reason: "r".into(),
        ticket: "#1".into(),
        confirmed_by: confirmed_by.into(),
    }
}

fn fails(r: &Report) -> Vec<String> {
    r.lines
        .iter()
        .filter(|l| l.starts_with("FAIL"))
        .cloned()
        .collect()
}

#[test]
fn an_unlisted_escape_is_red_by_name() {
    let mut r = Report::default();
    judge_escapes(&[esc("P/A.lean", "P.x", "axiom")], &[], false, &mut r);
    assert!(r.reject);
    assert!(fails(&r)[0].contains("ESCAPE P/A.lean:1 `axiom` in P.x"));
}

#[test]
fn a_listed_confirmed_escape_is_green() {
    let mut r = Report::default();
    judge_escapes(
        &[esc("P/A.lean", "P.x", "axiom")],
        &[allowed("P/A.lean", "P.x", "axiom", "noah")],
        true,
        &mut r,
    );
    assert!(!r.reject, "{:?}", r.lines);
}

#[test]
fn pending_is_red_only_under_strict() {
    let found = [esc("P/A.lean", "P.x", "axiom")];
    let allow = [allowed("P/A.lean", "P.x", "axiom", "pending")];
    let mut lax = Report::default();
    judge_escapes(&found, &allow, false, &mut lax);
    assert!(!lax.reject && lax.lines.contains(&"PENDING (1)".to_string()));
    let mut strict = Report::default();
    judge_escapes(&found, &allow, true, &mut strict);
    assert!(strict.reject);
}

#[test]
fn an_entry_without_reason_or_ticket_is_red() {
    for blank in ["reason", "ticket"] {
        let mut a = allowed("P/A.lean", "P.x", "axiom", "pending");
        if blank == "reason" {
            a.reason.clear();
        } else {
            a.ticket.clear();
        }
        let mut r = Report::default();
        judge_escapes(&[esc("P/A.lean", "P.x", "axiom")], &[a], false, &mut r);
        assert!(
            fails(&r)
                .iter()
                .any(|l| l.contains(&format!("has no {blank}"))),
            "{blank}: {:?}",
            r.lines
        );
    }
}

#[test]
fn a_stale_entry_is_red() {
    let mut r = Report::default();
    judge_escapes(
        &[],
        &[allowed("P/A.lean", "P.x", "axiom", "pending")],
        false,
        &mut r,
    );
    assert!(fails(&r).iter().any(|l| l.contains("STALE")));
}

#[test]
fn only_a_label_not_in_the_set_fails_and_by_name() {
    let pair = |s: &str, l: &str| (s.to_string(), l.to_string());
    let base: BTreeSet<_> = [pair("c", "Theorems.Old"), pair("c", "Theorems.Fixed")].into();
    let now: BTreeSet<_> = [pair("c", "Theorems.Old"), pair("d", "Theorems.Bogus")].into();
    let mut r = Report::default();
    judge_labels(&now, Some(&base), &mut r);
    let f = fails(&r);
    assert!(
        f.iter()
            .any(|l| l.contains("NEW-UNRESOLVED-LABEL d: Theorems.Bogus")),
        "{f:?}"
    );
    assert_eq!(
        f.len(),
        1,
        "a listed label that resolves now is NOT a failure (infra#992): {f:?}"
    );
    assert!(r.lines.contains(&"RESOLVED-LABEL (1) still listed in unresolved-labels.json -- `make label-ratchet` removes them".to_string()), "{:?}", r.lines);
    let mut ok = Report::default();
    judge_labels(&now, Some(&now), &mut ok);
    assert!(!ok.reject);
}

#[test]
fn exact_names_are_the_fully_qualified_form_only() {
    assert!(is_exact_name(
        "ProvableContracts.CooperativeMatrix.matmul_block_sum"
    ));
    assert!(!is_exact_name("Theorems.Gelu"));
    assert!(!is_exact_name("ProvableContracts.X — prose"));
}

#[test]
fn the_pinned_set_is_status_axioms_plus_allowlisted_axioms() {
    let form = Formalization {
        axioms: vec!["propext".into()],
        capstones: vec![],
    };
    let allow = [
        allowed("P/A.lean", "P.ax", "axiom", "pending"),
        allowed("P/B.lean", "P.f", "unsafe", "pending"),
    ];
    assert_eq!(pinned_axioms(&form, &allow), vec!["propext", "P.ax"]);
}

#[test]
fn render_pins_only_the_cone_and_quotes_odd_components() {
    let roots: BTreeSet<Root> = [
        Root {
            fqn: "P.D.t".into(),
            module: "M.in".into(),
        },
        Root {
            fqn: "P.D.u".into(),
            module: "M.out".into(),
        },
    ]
    .into();
    let cone: BTreeSet<String> = ["M.in".to_string()].into();
    let form = Formalization {
        axioms: vec!["propext".into()],
        capstones: vec!["P.D.t".into()],
    };
    let s = render_axioms(&roots, &cone, &["propext".into(), "P.1x".into()], &form);
    assert!(s.contains("run_cmd pvlAxiomsSubset `P.D.t pvlPinned"));
    assert!(!s.contains("`P.D.u"));
    assert!(s.contains("1 pinned; 1 bound outside"));
    assert!(s.contains("[`propext, `P.«1x»]"));
    assert!(s.contains(
        "/-- info: 'P.D.t' depends on axioms: [propext] -/\n#guard_msgs in #print axioms P.D.t"
    ));
}

#[test]
fn escape_owner_rules() {
    let src = "namespace N\n@[implemented_by g] def f := 1\ntheorem t : True := by sorry\naxiom a : False\nend N\n";
    let toks = lex::tokens(&lex::blank(src));
    let tree = Tree {
        dir: PathBuf::new(),
        files: vec![LeanFile {
            rel: "P/A.lean".into(),
            module: "P.A".into(),
            decls: lex::decls(&toks),
            imports: vec![],
            toks,
        }],
    };
    let got: Vec<(String, String)> = escapes(&tree)
        .into_iter()
        .map(|e| (e.kind, e.decl))
        .collect();
    assert_eq!(
        got,
        vec![
            ("implemented_by".into(), "N.f".into()),
            ("sorry".into(), "N.t".into()),
            ("axiom".into(), "N.a".into())
        ]
    );
}

// ---- on-disk scenarios: CI mutates with `-- --lib`, so these must be LIB tests ----

const BOUND: &str = "ProvableContracts/Theorems/Gelu/Bound.lean";

/// `<tmp>/lean` + `<tmp>/contracts`: the root imports Gelu/Bound (which imports a nested Util module), and one
/// module is an orphan; the contract binds `gelu_bound` by the label `Theorems.Gelu`.
struct Fx {
    d: tempfile::TempDir,
}

impl Fx {
    fn new() -> Self {
        let fx = Self {
            d: tempfile::tempdir().expect("tempdir"),
        };
        fx.put(
            "lean/ProvableContracts.lean",
            "/- header -/\nimport ProvableContracts.Theorems.Gelu.Bound\n",
        );
        fx.put(
            &format!("lean/{BOUND}"),
            "import ProvableContracts.Defs.Deep.Util\nnamespace ProvableContracts.Gelu\n\
             theorem gelu_bound : True := trivial\nend ProvableContracts.Gelu\n",
        );
        fx.put(
            "lean/ProvableContracts/Defs/Deep/Util.lean",
            "def util := 1\n",
        );
        fx.put(
            "lean/ProvableContracts/Theorems/Orphan/Lone.lean",
            "namespace ProvableContracts.Orphan\ntheorem lone : True := trivial\nend ProvableContracts.Orphan\n",
        );
        fx.put("lean/notes.txt", "not lean\n");
        fx.put(
            "contracts/gelu-v1.yaml",
            "equations:\n  e:\n    lean_theorem: Theorems.Gelu\n",
        );
        fx
    }

    fn put(&self, rel: &str, text: &str) {
        let p = self.d.path().join(rel);
        std::fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
        std::fs::write(p, text).expect("write");
    }

    fn lean(&self) -> PathBuf {
        self.d.path().join("lean")
    }

    fn contracts(&self) -> PathBuf {
        self.d.path().join("contracts")
    }

    fn generate_to_disk(&self) {
        let g = generate(&self.lean(), &self.contracts()).expect("generate");
        std::fs::write(self.lean().join(AXIOMS_FILE), g.text).expect("write Axioms.lean");
    }

    fn check(&self, strict: bool) -> Report {
        check(&self.lean(), &self.contracts(), CheckOpts { strict })
    }
}

fn has(r: &Report, needle: &str) -> bool {
    r.lines.iter().any(|l| l.contains(needle))
}

#[test]
fn load_walks_nested_dirs_names_modules_and_skips_non_lean_files() {
    let fx = Fx::new();
    let t = Tree::load(&fx.lean()).expect("load");
    let mods: Vec<&str> = t.files.iter().map(|f| f.module.as_str()).collect();
    assert_eq!(
        mods,
        vec![
            "ProvableContracts.Defs.Deep.Util",
            "ProvableContracts.Theorems.Gelu.Bound",
            "ProvableContracts.Theorems.Orphan.Lone",
            "ProvableContracts"
        ]
    );
    assert_eq!(t.files[1].rel, BOUND);
    assert_eq!(t.files[1].imports, vec!["ProvableContracts.Defs.Deep.Util"]);
    assert_eq!(t.dir, fx.lean());
}

#[test]
fn load_declines_without_the_root_file() {
    let fx = Fx::new();
    std::fs::remove_file(fx.lean().join("ProvableContracts.lean")).expect("rm");
    assert!(Tree::load(&fx.lean())
        .unwrap_err()
        .contains("no ProvableContracts.lean"));
}

#[test]
fn the_cone_is_transitive_and_leaves_the_orphan_out() {
    let fx = Fx::new();
    let cone = Tree::load(&fx.lean()).expect("load").cone();
    let want: BTreeSet<String> = [
        "ProvableContracts",
        "ProvableContracts.Defs.Deep.Util",
        "ProvableContracts.Theorems.Gelu.Bound",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    assert_eq!(cone, want);
}

#[test]
fn bind_resolves_labels_and_exact_names_and_records_the_rest() {
    let fx = Fx::new();
    fx.put(
        "contracts/gelu-v1.yaml",
        "equations:\n  a:\n    lean_theorem: Theorems.Gelu\n  b:\n    lean_theorem: ProvableContracts.Orphan.lone\n\
         \x20 c:\n    lean_theorem: ProvableContracts.Gelu.no_such\n  d:\n    lean_theorem: Theorems.NoSuchThing\n",
    );
    let b = bind(&Tree::load(&fx.lean()).expect("load"), &fx.contracts());
    let roots: Vec<&str> = b.roots.iter().map(|r| r.fqn.as_str()).collect();
    assert_eq!(
        roots,
        vec![
            "ProvableContracts.Gelu.gelu_bound",
            "ProvableContracts.Orphan.lone"
        ]
    );
    let pair = |c: &str, l: &str| (c.to_string(), l.to_string());
    assert_eq!(
        b.missing,
        [pair("gelu-v1", "ProvableContracts.Gelu.no_such")].into()
    );
    assert_eq!(
        b.unresolved_labels,
        [pair("gelu-v1", "Theorems.NoSuchThing")].into()
    );
}

#[test]
fn generate_pins_the_cone_and_counts_the_orphaned_root() {
    let fx = Fx::new();
    fx.put(
        "contracts/gelu-v1.yaml",
        "equations:\n  a:\n    lean_theorem: Theorems.Gelu\n  b:\n    lean_theorem: Theorems.Orphan\n",
    );
    let g = generate(&fx.lean(), &fx.contracts()).expect("generate");
    let (text, b) = (g.text, g.binding);
    assert_eq!(b.roots.len(), 2);
    assert!(text.contains("run_cmd pvlAxiomsSubset `ProvableContracts.Gelu.gelu_bound pvlPinned"));
    assert!(!text.contains("`ProvableContracts.Orphan.lone"), "{text}");
    assert!(text.contains("-- 1 pinned; 1 bound outside"), "{text}");
    let r = {
        fx.generate_to_disk();
        fx.check(false)
    };
    assert!(has(&r, "ROOTS 1 pinned, 1 ORPHANED-ROOT"), "{:?}", r.lines);
}

#[test]
fn a_clean_tree_checks_green_with_no_decline() {
    let fx = Fx::new();
    fx.generate_to_disk();
    let r = fx.check(true);
    assert!(!r.reject && r.decline.is_none(), "{:?}", r.lines);
    assert!(has(&r, "UNRESOLVED-LABEL (0) (listed 0)"), "{:?}", r.lines);
}

#[test]
fn check_rejects_an_unlisted_escape_found_on_disk() {
    let fx = Fx::new();
    fx.put(
        &format!("lean/{BOUND}"),
        "namespace ProvableContracts.Gelu\ntheorem gelu_bound : True := trivial\naxiom m : False\nend ProvableContracts.Gelu\n",
    );
    fx.generate_to_disk();
    let r = fx.check(false);
    assert!(
        r.reject
            && has(
                &r,
                &format!("ESCAPE {BOUND}:3 `axiom` in ProvableContracts.Gelu.m")
            ),
        "{:?}",
        r.lines
    );
}

#[test]
fn check_names_what_a_missing_exact_root_actually_is() {
    let fx = Fx::new();
    fx.put(
        &format!("lean/{BOUND}"),
        "namespace ProvableContracts.Gelu\ntheorem gelu_bound : True := trivial\naxiom m : False\nend ProvableContracts.Gelu\n",
    );
    fx.put(
        "lean/escape-allowlist.yaml",
        format!("- file: {BOUND}\n  decl: ProvableContracts.Gelu.m\n  kind: axiom\n  reason: r\n  ticket: t\n  confirmed_by: pending\n").as_str(),
    );
    fx.put(
        "contracts/gelu-v1.yaml",
        "equations:\n  a:\n    lean_theorem: Theorems.Gelu\n  b:\n    lean_theorem: ProvableContracts.Gelu.m\n\
         \x20 c:\n    lean_theorem: ProvableContracts.Gelu.gone\n",
    );
    fx.generate_to_disk();
    let r = fx.check(false);
    assert!(has(&r, "MISSING-ROOT contract gelu-v1: ProvableContracts.Gelu.m -- it names an `axiom`, not a proved theorem"), "{:?}", r.lines);
    assert!(has(&r, "MISSING-ROOT contract gelu-v1: ProvableContracts.Gelu.gone -- no such theorem in the tree"), "{:?}", r.lines);
    assert!(r.reject);
}

#[test]
fn capstones_must_name_a_theorem_and_get_an_exact_pin() {
    let fx = Fx::new();
    fx.put(
        "lean/formalization.yaml",
        "status:\n  axioms: [propext]\ncapstones:\n  - ProvableContracts.Gelu.gelu_bound\n",
    );
    fx.generate_to_disk();
    let text = std::fs::read_to_string(fx.lean().join(AXIOMS_FILE)).expect("read");
    assert!(
        text.contains("def pvlPinned : List Name := [`propext]"),
        "{text}"
    );
    assert!(
        text.contains(
            "/-- info: 'ProvableContracts.Gelu.gelu_bound' depends on axioms: [propext] -/"
        ),
        "{text}"
    );
    assert!(!fx.check(false).reject);
    fx.put(
        "lean/formalization.yaml",
        "capstones:\n  - ProvableContracts.Gelu.nope\n",
    );
    fx.generate_to_disk();
    let r = fx.check(false);
    assert!(
        r.reject && has(&r, "MISSING-ROOT capstone ProvableContracts.Gelu.nope"),
        "{:?}",
        r.lines
    );
    fx.put("lean/formalization.yaml", "capstones: [unclosed\n");
    assert!(
        generate(&fx.lean(), &fx.contracts()).is_err(),
        "an unreadable formalization.yaml is not the default"
    );
}

#[test]
fn zero_roots_declines_only_when_nothing_failed() {
    let fx = Fx::new();
    fx.put(
        "contracts/gelu-v1.yaml",
        "equations:\n  a:\n    lean_theorem: none\n",
    );
    fx.generate_to_disk();
    let r = fx.check(false);
    assert!(
        !r.reject
            && r.decline
                .as_deref()
                .is_some_and(|d| d.contains("0 contract-bound theorems")),
        "{r:?}"
    );
    std::fs::remove_file(fx.lean().join(AXIOMS_FILE)).expect("rm");
    let r = fx.check(false);
    assert!(
        r.reject && r.decline.is_none() && has(&r, "no Axioms.lean"),
        "{r:?}"
    );
}

#[test]
fn a_stale_axioms_file_is_red() {
    let fx = Fx::new();
    fx.generate_to_disk();
    let p = fx.lean().join(AXIOMS_FILE);
    let text = std::fs::read_to_string(&p).expect("read");
    std::fs::write(&p, format!("{text}-- edit\n")).expect("write");
    let r = fx.check(false);
    assert!(r.reject && has(&r, "STALE Axioms.lean"), "{:?}", r.lines);
}

#[test]
fn check_declines_on_unreadable_inputs_and_rejects_an_unreadable_label_set() {
    let fx = Fx::new();
    fx.generate_to_disk();
    for (file, text) in [
        ("escape-allowlist.yaml", "file: not-a-list\n"),
        ("formalization.yaml", "capstones: [x\n"),
    ] {
        fx.put(&format!("lean/{file}"), text);
        let r = fx.check(false);
        assert!(
            r.decline.as_deref().is_some_and(|d| d.contains(file)) && !r.reject,
            "{file}: {r:?}"
        );
        std::fs::remove_file(fx.lean().join(file)).expect("rm");
    }
    fx.put("lean/unresolved-labels.json", "{not json");
    let r = fx.check(false);
    assert!(
        r.reject && has(&r, "unresolved-labels.json unreadable"),
        "{:?}",
        r.lines
    );
    std::fs::remove_file(fx.lean().join("ProvableContracts.lean")).expect("rm");
    let r = fx.check(false);
    assert!(
        r.decline
            .as_deref()
            .is_some_and(|d| d.contains("no ProvableContracts.lean")),
        "{r:?}"
    );
}

#[test]
fn the_label_set_is_read_and_its_malformations_are_errors() {
    let fx = Fx::new();
    assert_eq!(load_labels(&fx.lean()), Ok(None));
    let pair = |c: &str, l: &str| (c.to_string(), l.to_string());
    let set: BTreeSet<_> = [pair("a-v1", "Theorems.X"), pair("b-v1", "Theorems.Y")].into();
    fx.put("lean/unresolved-labels.json", &render_labels(&set));
    assert_eq!(load_labels(&fx.lean()), Ok(Some(set)));
    for bad in [
        "{}",
        "{\"labels\": [{\"contract\": \"a\"}]}",
        "{\"labels\": \"x\"}",
        "nope",
    ] {
        fx.put("lean/unresolved-labels.json", bad);
        assert!(load_labels(&fx.lean()).is_err(), "{bad}");
    }
}

#[test]
fn render_labels_names_its_only_writer() {
    let s = render_labels(&BTreeSet::new());
    assert!(
        s.contains("\"command\": \"make label-ratchet\"") && s.ends_with("}\n"),
        "{s}"
    );
}

#[test]
fn label_ratchet_seeds_shrinks_and_never_adds() {
    let fx = Fx::new();
    let set = fx.lean().join(LABELS);
    fx.put(
        "contracts/gelu-v1.yaml",
        "equations:\n  a:\n    lean_theorem: Theorems.Gelu\n  b:\n    lean_theorem: Theorems.X\n",
    );
    let r = ratchet_labels(&fx.lean(), &fx.contracts());
    assert!(!r.reject && has(&r, "(1 label(s))"), "{:?}", r.lines);
    assert!(std::fs::read_to_string(&set)
        .expect("seeded")
        .contains("Theorems.X"));
    fx.put(
        "contracts/gelu-v1.yaml",
        "equations:\n  a:\n    lean_theorem: Theorems.Gelu\n  b:\n    lean_theorem: Theorems.Y\n",
    );
    let r = ratchet_labels(&fx.lean(), &fx.contracts());
    assert!(
        r.reject && has(&r, "NEW-UNRESOLVED-LABEL gelu-v1: Theorems.Y"),
        "{:?}",
        r.lines
    );
    let after = std::fs::read_to_string(&set).expect("set");
    assert!(
        !after.contains("Theorems.Y") && !after.contains("Theorems.X"),
        "{after}"
    );
    fx.put("lean/unresolved-labels.json", "{bad");
    assert!(ratchet_labels(&fx.lean(), &fx.contracts()).reject);
    std::fs::remove_file(fx.lean().join("ProvableContracts.lean")).expect("rm");
    assert!(ratchet_labels(&fx.lean(), &fx.contracts())
        .decline
        .is_some());
}

#[test]
fn the_allowlist_reader_handles_absent_empty_and_malformed() {
    let fx = Fx::new();
    assert_eq!(load_allowlist(&fx.lean()), Ok(vec![]));
    fx.put("lean/escape-allowlist.yaml", "");
    assert_eq!(load_allowlist(&fx.lean()), Ok(vec![]));
    fx.put("lean/escape-allowlist.yaml", "- file: f\n  decl: d\n  kind: axiom\n  reason: r\n  ticket: 4139\n  confirmed_by: pending\n");
    let got = load_allowlist(&fx.lean()).expect("read");
    assert_eq!(
        got,
        vec![Allowed {
            file: "f".into(),
            decl: "d".into(),
            kind: "axiom".into(),
            reason: "r".into(),
            ticket: "4139".into(),
            confirmed_by: "pending".into()
        }]
    );
    fx.put("lean/escape-allowlist.yaml", "key: value\n");
    assert!(load_allowlist(&fx.lean()).is_err());
    fx.put("lean/escape-allowlist.yaml", "- [unclosed\n");
    assert!(load_allowlist(&fx.lean()).is_err());
}

#[test]
fn formalization_defaults_and_overrides() {
    let fx = Fx::new();
    let f = load_formalization(&fx.lean()).expect("absent");
    assert_eq!(
        f.axioms,
        DEFAULT_AXIOMS
            .iter()
            .map(|s| (*s).to_string())
            .collect::<Vec<_>>()
    );
    assert!(f.capstones.is_empty());
    fx.put(
        "lean/formalization.yaml",
        "status:\n  axioms: [propext]\ncapstones: [A.b]\n",
    );
    assert_eq!(
        load_formalization(&fx.lean()),
        Ok(Formalization {
            axioms: vec!["propext".into()],
            capstones: vec!["A.b".into()]
        })
    );
}

#[test]
fn an_entry_missing_a_non_reason_field_is_red_too() {
    for field in ["file", "decl", "kind", "confirmed_by"] {
        let mut a = allowed("P/A.lean", "P.x", "axiom", "pending");
        match field {
            "file" => a.file.clear(),
            "decl" => a.decl.clear(),
            "kind" => a.kind.clear(),
            _ => a.confirmed_by.clear(),
        }
        let mut r = Report::default();
        judge_escapes(&[], &[a], false, &mut r);
        assert!(
            fails(&r)
                .iter()
                .any(|l| l.contains(&format!("has no {field}"))),
            "{field}: {:?}",
            r.lines
        );
    }
}

#[test]
fn pending_lists_each_entry_and_confirmed_is_not_pending() {
    let found = [
        esc("P/A.lean", "P.x", "axiom"),
        esc("P/B.lean", "P.y", "axiom"),
    ];
    let allow = [
        allowed("P/A.lean", "P.x", "axiom", "pending"),
        allowed("P/B.lean", "P.y", "axiom", "noah"),
    ];
    let mut r = Report::default();
    judge_escapes(&found, &allow, false, &mut r);
    assert_eq!(
        r.lines,
        vec![
            "PENDING (1)".to_string(),
            "  axiom P.x in P/A.lean -- #1".to_string()
        ]
    );
}

#[test]
fn the_pinned_set_does_not_duplicate_an_axiom_already_pinned() {
    let form = Formalization {
        axioms: vec!["propext".into()],
        capstones: vec![],
    };
    let allow = [
        allowed("P/A.lean", "propext", "axiom", "pending"),
        allowed("P/B.lean", "", "axiom", "pending"),
    ];
    assert_eq!(pinned_axioms(&form, &allow), vec!["propext"]);
}

#[test]
fn owner_of_an_escape_with_no_declaration_is_none() {
    let toks = lex::tokens(&lex::blank("sorry\n"));
    assert_eq!(owner(&lex::decls(&toks), 0, "sorry"), "<none>");
}

#[test]
fn private_theorems_are_not_roots_but_their_escapes_still_scan() {
    let fx = Fx::new();
    fx.put(
        &format!("lean/{BOUND}"),
        "namespace ProvableContracts.Gelu\nprivate theorem gelu_bound : True := by sorry\nend ProvableContracts.Gelu\n",
    );
    let t = Tree::load(&fx.lean()).expect("load");
    assert!(bind(&t, &fx.contracts()).roots.is_empty());
    assert_eq!(escapes(&t).len(), 1);
}
