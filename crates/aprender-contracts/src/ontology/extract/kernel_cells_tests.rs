use super::*;

const F: &str = "kernel-registry.json";

fn row(id: &str, backend: &str, t: u32, layout: &str, arch: &str) -> String {
    format!(
        r#"{{"kernel_id":"{id}","backend":"{backend}","ggml_type":{t},"layout":"{layout}","arch":"{arch}"}}"#
    )
}

fn registry(rows: &[String]) -> Vec<RegistryRow> {
    parse_registry(
        F,
        format!(r#"{{"kernels":[{}]}}"#, rows.join(",")).as_bytes(),
    )
    .expect("fixture registry parses")
}

/// A registry excerpt shaped like KREG-001's: q4_k/q6_k on cpu and cuda, q3_k on cpu only.
fn excerpt() -> Vec<RegistryRow> {
    registry(&[
        row("cpu.matvec.q4_k", "cpu", 12, ROW_MAJOR, ANY_ARCH),
        row("cpu.matvec.q6_k", "cpu", 14, ROW_MAJOR, ANY_ARCH),
        row("cpu.matvec.q3_k", "cpu", 11, ROW_MAJOR, ANY_ARCH),
        row("cuda.gemv.q4_k", "cuda", 12, ROW_MAJOR, ANY_ARCH),
        row("cuda.gemv.q6_k", "cuda", 14, ROW_MAJOR, ANY_ARCH),
    ])
}

fn types(ts: &[u32]) -> BTreeSet<u32> {
    ts.iter().copied().collect()
}

fn ids(ks: &[&str]) -> BTreeSet<String> {
    ks.iter().map(|k| (*k).to_string()).collect()
}

/// (backend, model types, kernels used, unregistered types).
type MapCase = (
    &'static str,
    &'static [u32],
    &'static [&'static str],
    &'static [u32],
);

/// Case table: (backend, host arch, model types) → (kernels used, unregistered types).
#[test]
fn the_static_map_case_table() {
    let rows = excerpt();
    let cases: &[MapCase] = &[
        // Q4_K_M: q4_k + q6_k tensors, both served on cpu and on cuda.
        (
            "cpu",
            &[12, 14],
            &["cpu.matvec.q4_k", "cpu.matvec.q6_k"],
            &[],
        ),
        (
            "cuda",
            &[12, 14],
            &["cuda.gemv.q4_k", "cuda.gemv.q6_k"],
            &[],
        ),
        // q3_k exists on cpu only: the same model is RED on a cuda host and only there.
        (
            "cpu",
            &[11, 12],
            &["cpu.matvec.q3_k", "cpu.matvec.q4_k"],
            &[],
        ),
        ("cuda", &[11, 12], &["cuda.gemv.q4_k"], &[11]),
        // A type no row names (f32 norms here, as the excerpt has no f32 row) is unregistered.
        ("cpu", &[0, 12], &["cpu.matvec.q4_k"], &[0]),
        // A backend with no rows at all: every type is unregistered, none is dropped.
        ("wgpu", &[12, 14], &[], &[12, 14]),
        // A model with no tensor types uses nothing; the shape's minCount 1 is what refuses it.
        ("cpu", &[], &[], &[]),
    ];
    for (backend, ts, uses, unreg) in cases {
        let m = static_kernel_map(&rows, backend, "x86_64", &types(ts));
        assert_eq!(m.uses, ids(uses), "{backend} {ts:?}");
        assert_eq!(m.unregistered, types(unreg), "{backend} {ts:?}");
    }
}

/// RR2-F2 precursor: removing the only row for a type turns that type unregistered for every model that
/// has it, and leaves a model without it untouched.
#[test]
fn removing_a_row_unregisters_its_type_and_nothing_else() {
    let full = excerpt();
    let without: Vec<RegistryRow> = full
        .iter()
        .filter(|r| r.kernel_id != "cpu.matvec.q6_k")
        .cloned()
        .collect();
    let with_q6 = types(&[12, 14]);
    let without_q6 = types(&[11, 12]);
    assert!(static_kernel_map(&full, "cpu", "x86_64", &with_q6)
        .unregistered
        .is_empty());
    assert_eq!(
        static_kernel_map(&without, "cpu", "x86_64", &with_q6).unregistered,
        types(&[14])
    );
    assert_eq!(
        static_kernel_map(&full, "cpu", "x86_64", &without_q6),
        static_kernel_map(&without, "cpu", "x86_64", &without_q6)
    );
}

/// An arch-specific row serves only its arch; the map keeps every row a host of that arch could be
/// served (the superset of `admit`), so both need receipts on x86_64.
#[test]
fn arch_rows_serve_their_arch_and_the_map_is_the_superset() {
    let rows = registry(&[
        row("cpu.matvec.q4_k", "cpu", 12, ROW_MAJOR, ANY_ARCH),
        row("cpu.matvec.q4_k.avx512", "cpu", 12, ROW_MAJOR, "x86_64"),
    ]);
    let t = types(&[12]);
    assert_eq!(
        static_kernel_map(&rows, "cpu", "x86_64", &t).uses,
        ids(&["cpu.matvec.q4_k", "cpu.matvec.q4_k.avx512"])
    );
    assert_eq!(
        static_kernel_map(&rows, "cpu", "aarch64", &t).uses,
        ids(&["cpu.matvec.q4_k"])
    );
    // An arch row alone does not serve another arch: the type is unregistered there.
    let only_x86 = registry(&[row(
        "cpu.matvec.q4_k.avx512",
        "cpu",
        12,
        ROW_MAJOR,
        "x86_64",
    )]);
    assert_eq!(
        static_kernel_map(&only_x86, "cpu", "aarch64", &t).unregistered,
        types(&[12])
    );
}

/// A non-row-major row never serves a GGUF/APR tensor (LAYOUT-001/002).
#[test]
fn a_col_major_row_serves_nothing() {
    let rows = registry(&[row("cpu.matvec.q4_k.col", "cpu", 12, "col_major", ANY_ARCH)]);
    let m = static_kernel_map(&rows, "cpu", "x86_64", &types(&[12]));
    assert!(m.uses.is_empty());
    assert_eq!(m.unregistered, types(&[12]));
}

/// A registry read in part would hide rows, so every defect refuses the whole file.
#[test]
fn a_malformed_registry_is_refused_whole() {
    let good = row("a", "cpu", 12, ROW_MAJOR, ANY_ARCH);
    let cases: &[(&str, String)] = &[
        ("not JSON", "{".to_string()),
        ("no non-empty `kernels`", r#"{"schema":"x"}"#.to_string()),
        ("no non-empty `kernels`", r#"{"kernels":[]}"#.to_string()),
        ("duplicate kernel_id `a`", format!(r#"{{"kernels":[{good},{good}]}}"#)),
        (
            "`ggml_type` missing",
            r#"{"kernels":[{"kernel_id":"a","backend":"cpu","layout":"row_major","arch":"any"}]}"#
                .to_string(),
        ),
        (
            "`ggml_type` missing",
            r#"{"kernels":[{"kernel_id":"a","backend":"cpu","ggml_type":4294967296,"layout":"row_major","arch":"any"}]}"#
                .to_string(),
        ),
        (
            "`backend` missing",
            r#"{"kernels":[{"kernel_id":"a","backend":"","ggml_type":12,"layout":"row_major","arch":"any"}]}"#
                .to_string(),
        ),
        (
            "`arch` missing",
            r#"{"kernels":[{"kernel_id":"a","backend":"cpu","ggml_type":12,"layout":"row_major"}]}"#
                .to_string(),
        ),
    ];
    for (want, doc) in cases {
        let err = parse_registry(F, doc.as_bytes()).expect_err(want);
        assert!(err.what.contains(want), "{want}: got {err}");
        assert_eq!(err.file, F);
    }
}

/// Edges and literals only; no verdict predicate is written on the model cell.
#[test]
fn emit_writes_edges_and_unregistered_literals_only() {
    let mut g = Graph::new();
    let m = static_kernel_map(&excerpt(), "cuda", "x86_64", &types(&[11, 12, 14]));
    emit_model_cell(&mut g, "lambda", "aa", &m);
    let cell = model_cell("lambda", "aa");
    let used: BTreeSet<&str> = g
        .objects(&cell, &rel("usesKernel"))
        .into_iter()
        .filter_map(Term::as_iri)
        .collect();
    let want = [
        kernel_cell("lambda", "cuda.gemv.q4_k"),
        kernel_cell("lambda", "cuda.gemv.q6_k"),
    ];
    assert_eq!(used, want.iter().map(String::as_str).collect());
    assert_eq!(
        g.objects(&cell, &rel("unregisteredQtype")),
        vec![&Term::integer(11)]
    );
    let preds: BTreeSet<String> = g
        .predicates_of(&cell)
        .into_iter()
        .map(str::to_string)
        .collect();
    let allowed: BTreeSet<String> = [
        RDF_TYPE.to_string(),
        rel("usesKernel"),
        rel("unregisteredQtype"),
    ]
    .into_iter()
    .collect();
    assert_eq!(preds, allowed);
}

/// A kernel cell is per host: the same kernel on two hosts is two cells (RR2-F6).
#[test]
fn a_kernel_cell_is_per_host() {
    assert_ne!(
        kernel_cell("lambda", "cuda.gemv.q4_k"),
        kernel_cell("gx10", "cuda.gemv.q4_k")
    );
    // A `/` inside a segment cannot fake the host boundary.
    assert_ne!(kernel_cell("a/b", "c"), kernel_cell("a", "b/c"));
}

// ── RR2 case table: the committed release-readiness-v2 shapes over graphs built by these emitters ──

mod rr2 {
    use super::*;
    use crate::ontology::shapes::{parse_shapes, validate, Severity};

    const HOST: &str = "lambda";
    const PASS: KernelEvidence = KernelEvidence {
        pass: true,
        within_bound: true,
        fresh: true,
        arch_match: true,
    };

    fn shapes() -> Vec<crate::ontology::shapes::NodeShape> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../contracts/release-readiness-v2.yaml");
        let text = std::fs::read_to_string(&path).expect("release-readiness-v2.yaml is committed");
        let doc: serde_yaml::Value = serde_yaml::from_str(&text).expect("the contract is YAML");
        let s =
            parse_shapes("release-readiness-v2", &doc).expect("shapes inside the engine's subset");
        assert_eq!(s.len(), 2, "the kernel and model shapes");
        s
    }

    fn registry() -> Vec<RegistryRow> {
        super::registry(&[
            row("cuda.gemv.q4_k", "cuda", 12, ROW_MAJOR, ANY_ARCH),
            row("cuda.gemv.q6_k", "cuda", 14, ROW_MAJOR, ANY_ARCH),
            row("cuda.gemv.q8_0", "cuda", 8, ROW_MAJOR, ANY_ARCH),
        ])
    }

    /// Three models on one host: A = q4_k + q6_k (a Q4_K_M), B = q4_k only, C = q8_0 only.
    const MODELS: [(&str, &[u32]); 3] = [("A", &[12, 14]), ("B", &[12]), ("C", &[8])];

    /// One plant per case; `None` is the all-green graph.
    #[derive(Clone, Copy)]
    enum Plant {
        /// RR2-F1: the kernel cell exists (some model uses it) but has no receipt.
        NoReceipt(&'static str),
        /// RR2-F2: model C gains a tensor type (q3_k, 11) no row serves.
        Unregistered,
        /// RR2-F3: model B's smoke dispatched a kernel the static map did not predict.
        Unpredicted,
        /// RR2-F4: the kernel's receipt input key is stale.
        Stale(&'static str),
        /// RR2-F5: model A has no smoke receipt.
        NoSmoke,
        /// RR2-F6: the kernel's receipt was measured on another arch.
        ForeignArch(&'static str),
    }

    fn graph(plant: Option<Plant>) -> Graph {
        let rows = registry();
        let mut g = Graph::new();
        let mut used = BTreeSet::new();
        for (m, ts) in MODELS {
            let mut ts = types(ts);
            if m == "C" && matches!(plant, Some(Plant::Unregistered)) {
                ts.insert(11);
            }
            let map = static_kernel_map(&rows, "cuda", "x86_64", &ts);
            used.extend(map.uses.iter().cloned());
            emit_model_cell(&mut g, HOST, m, &map);
            let smoke = SmokeEvidence {
                pass: true,
                fresh: true,
                unpredicted: if m == "B" && matches!(plant, Some(Plant::Unpredicted)) {
                    ids(&["cuda.gemv.q6_k"])
                } else {
                    BTreeSet::new()
                },
            };
            let no_smoke = m == "A" && matches!(plant, Some(Plant::NoSmoke));
            emit_smoke(&mut g, HOST, m, (!no_smoke).then_some(&smoke));
        }
        for k in &used {
            let e = match plant {
                Some(Plant::NoReceipt(x)) if x == k => None,
                Some(Plant::Stale(x)) if x == k => Some(KernelEvidence {
                    fresh: false,
                    ..PASS
                }),
                Some(Plant::ForeignArch(x)) if x == k => Some(KernelEvidence {
                    arch_match: false,
                    ..PASS
                }),
                _ => Some(PASS),
            };
            emit_kernel_cell(&mut g, HOST, k, e);
        }
        g
    }

    /// The focus nodes with a violation.
    fn red(plant: Option<Plant>) -> BTreeSet<String> {
        let report = validate(&graph(plant), &shapes());
        assert!(
            report.focus_nodes_n > 0,
            "the shapes found no focus node: vacuous"
        );
        report
            .results
            .iter()
            .filter(|r| r.severity == Severity::Violation)
            .map(|r| r.focus.clone())
            .collect()
    }

    fn mc(m: &str) -> String {
        model_cell(HOST, m)
    }

    fn kc(k: &str) -> String {
        kernel_cell(HOST, k)
    }

    fn set(xs: &[String]) -> BTreeSet<String> {
        xs.iter().cloned().collect()
    }

    #[test]
    fn rr2_all_green_conforms() {
        let report = validate(&graph(None), &shapes());
        assert!(report.conforms(), "{:#?}", report.results);
        // 3 model cells + 3 kernel cells were graded.
        assert!(report.focus_nodes_n >= 6, "{}", report.focus_nodes_n);
    }

    /// RR2-F1, for each kernel: exactly the model cells whose map uses it, plus the kernel cell itself.
    #[test]
    fn rr2_f1_removing_one_kernel_receipt_reds_exactly_its_models() {
        let cases: &[(&'static str, &[&str])] = &[
            ("cuda.gemv.q4_k", &["A", "B"]),
            ("cuda.gemv.q6_k", &["A"]),
            ("cuda.gemv.q8_0", &["C"]),
        ];
        for (k, models) in cases {
            let mut want: Vec<String> = models.iter().map(|m| mc(m)).collect();
            want.push(kc(k));
            assert_eq!(red(Some(Plant::NoReceipt(k))), set(&want), "{k}");
        }
    }

    #[test]
    fn rr2_f2_to_f6_each_red_names_its_cells() {
        let cases: Vec<(&str, Plant, Vec<String>)> = vec![
            ("F2 unregistered qtype", Plant::Unregistered, vec![mc("C")]),
            ("F3 unpredicted kernel", Plant::Unpredicted, vec![mc("B")]),
            (
                "F4 stale receipt",
                Plant::Stale("cuda.gemv.q6_k"),
                vec![mc("A"), kc("cuda.gemv.q6_k")],
            ),
            ("F5 no smoke", Plant::NoSmoke, vec![mc("A")]),
            (
                "F6 foreign arch",
                Plant::ForeignArch("cuda.gemv.q8_0"),
                vec![mc("C"), kc("cuda.gemv.q8_0")],
            ),
        ];
        for (name, plant, want) in cases {
            assert_eq!(red(Some(plant)), set(&want), "{name}");
        }
    }

    /// A model cell with no kernel edge at all (no tensor types read) is RED, never vacuously green.
    #[test]
    fn rr2_a_model_cell_with_no_kernels_is_red() {
        let mut g = graph(None);
        emit_model_cell(&mut g, HOST, "D", &KernelMap::default());
        let smoke = SmokeEvidence {
            pass: true,
            fresh: true,
            unpredicted: BTreeSet::new(),
        };
        emit_smoke(&mut g, HOST, "D", Some(&smoke));
        let report = validate(&g, &shapes());
        let red: BTreeSet<&str> = report.results.iter().map(|r| r.focus.as_str()).collect();
        assert_eq!(red, [mc("D")].iter().map(String::as_str).collect());
    }

    /// The RR2 inputs as one `CellHost`, for [`build_cells`].
    fn cell_host(types_of_c: Option<&[u32]>, drop_kernel: Option<&str>) -> CellHost {
        let rows = registry();
        let mut models: BTreeMap<String, Option<BTreeSet<u32>>> = BTreeMap::new();
        for (m, ts) in MODELS {
            let ts = if m == "C" { types_of_c } else { Some(ts) };
            models.insert(m.to_string(), ts.map(types));
        }
        CellHost {
            id: HOST.to_string(),
            backend: "cuda".to_string(),
            arch: "sm_89".to_string(),
            models,
            kernels: rows
                .iter()
                .filter(|r| Some(r.kernel_id.as_str()) != drop_kernel)
                .map(|r| (r.kernel_id.clone(), PASS))
                .collect(),
            smokes: MODELS
                .iter()
                .map(|(m, _)| {
                    let s = SmokeEvidence {
                        pass: true,
                        fresh: true,
                        unpredicted: BTreeSet::new(),
                    };
                    ((*m).to_string(), s)
                })
                .collect(),
        }
    }

    fn built_red(h: &CellHost) -> BTreeSet<String> {
        let mut g = Graph::new();
        build_cells(&mut g, &registry(), std::slice::from_ref(h));
        let report = validate(&g, &shapes());
        assert!(report.focus_nodes_n > 0, "vacuous");
        report.results.iter().map(|r| r.focus.clone()).collect()
    }

    /// `build_cells` over the RR2 inputs: all green conforms, and RR2-F1 holds through the builder too.
    #[test]
    fn build_cells_green_and_f1() {
        assert_eq!(built_red(&cell_host(Some(&[8]), None)), BTreeSet::new());
        let red = built_red(&cell_host(Some(&[8]), Some("cuda.gemv.q6_k")));
        let want: BTreeSet<String> = [mc("A"), kc("cuda.gemv.q6_k")].into_iter().collect();
        assert_eq!(red, want);
    }

    /// A model whose tensor types no receipt recorded is RED, never skipped; the others stay green.
    #[test]
    fn build_cells_unknown_types_is_red() {
        let red = built_red(&cell_host(None, None));
        assert_eq!(red, [mc("C")].into_iter().collect());
    }
}

/// A real AC-3 receipt (evidence/kreg/parity/cpu.matvec.q2_k.json on la-71/4539-parity-receipts), trimmed.
const Q2K: &str = r#"{"schema":"kernel-parity-receipt/v1","kernel_id":"cpu.matvec.q2_k","host_arch":"x86_64",
  "oracle_independent":true,"served":{"max_abs_err":2.7e-6,"max_rel_err":3.247e-7},"tolerance_rel":7e-7}"#;

fn judge(json: &str, backend: &str, arch: &str, key: &str) -> KernelEvidence {
    judge_parity_receipt("r.json", json.as_bytes(), backend, arch, key)
        .expect("a parity receipt")
        .1
}

/// Case table for `judge_parity_receipt`: each field judges to the failing value when absent or wrong.
#[test]
fn the_parity_judge_case_table() {
    let keyed = Q2K.replace(
        r#""tolerance_rel""#,
        r#""input_set_hash":"k1","tolerance_rel""#,
    );
    let all = KernelEvidence {
        pass: true,
        within_bound: true,
        fresh: true,
        arch_match: true,
    };
    let cases: Vec<(String, &str, &str, &str, KernelEvidence)> = vec![
        (keyed.clone(), "cpu", "x86_64", "k1", all),
        // A receipt with no input_set_hash: stale.
        (
            Q2K.to_string(),
            "cpu",
            "x86_64",
            "k1",
            KernelEvidence {
                fresh: false,
                ..all
            },
        ),
        // An empty expected key never matches.
        (
            keyed.replace("k1", ""),
            "cpu",
            "x86_64",
            "",
            KernelEvidence {
                fresh: false,
                ..all
            },
        ),
        (
            keyed.clone(),
            "cpu",
            "x86_64",
            "k2",
            KernelEvidence {
                fresh: false,
                ..all
            },
        ),
        (
            keyed.clone(),
            "cpu",
            "aarch64",
            "k1",
            KernelEvidence {
                arch_match: false,
                ..all
            },
        ),
        // cuda matches on `sm`, and this cpu receipt has none.
        (
            keyed.clone(),
            "cuda",
            "x86_64",
            "k1",
            KernelEvidence {
                arch_match: false,
                ..all
            },
        ),
        (
            keyed.replace("7e-7", "1e-7"),
            "cpu",
            "x86_64",
            "k1",
            KernelEvidence {
                within_bound: false,
                ..all
            },
        ),
        (
            keyed.replace(r#","tolerance_rel":7e-7"#, ""),
            "cpu",
            "x86_64",
            "k1",
            KernelEvidence {
                within_bound: false,
                ..all
            },
        ),
        (
            keyed.replace(
                "\"oracle_independent\":true",
                "\"oracle_independent\":false",
            ),
            "cpu",
            "x86_64",
            "k1",
            KernelEvidence { pass: false, ..all },
        ),
        (
            keyed.replace("\"max_rel_err\":3.247e-7", "\"max_rel_err\":null"),
            "cpu",
            "x86_64",
            "k1",
            KernelEvidence {
                pass: false,
                within_bound: false,
                ..all
            },
        ),
    ];
    for (i, (json, backend, arch, key, want)) in cases.into_iter().enumerate() {
        assert_eq!(judge(&json, backend, arch, key), want, "case {i}");
    }
}

#[test]
fn a_non_parity_file_is_refused() {
    for bad in [
        "not json",
        r#"{"schema":"apr-model-ladder-receipt/v2","kernel_id":"x"}"#,
        r#"{"schema":"kernel-parity-receipt/v1"}"#,
        r#"{"schema":"kernel-parity-receipt/v1","kernel_id":""}"#,
    ] {
        assert!(
            judge_parity_receipt("r.json", bad.as_bytes(), "cpu", "x86_64", "k").is_err(),
            "{bad}"
        );
    }
    let (id, _) = judge_parity_receipt("r.json", Q2K.as_bytes(), "cpu", "x86_64", "k")
        .expect("a parity receipt");
    assert_eq!(id, "cpu.matvec.q2_k");
}

/// Inventory rows → models: a hashed row keeps its types, a partial or absent set is `None` (RED), and an
/// unhashed row is left to release-evidence.
#[test]
fn models_from_inventory_reads_types_whole_or_not_at_all() {
    let r = crate::ontology::receipts::parse(
        "gx10.json",
        r#"{"schema":"apr-model-ladder-receipt/v2","host":"gx10","inventory":[
          {"file":"a.gguf","sha256":"AA","tensor_types":[12,14,0]},
          {"file":"b.gguf","sha256":"bb","tensor_types":[12,"q6_k"]},
          {"file":"c.gguf","sha256":"cc"},
          {"file":"d.gguf","tensor_types":[8]}]}"#,
    )
    .expect("a v2 receipt");
    let m = models_from_inventory(&r.inventory);
    let want: BTreeMap<String, Option<BTreeSet<u32>>> = [
        ("aa".to_string(), Some(types(&[0, 12, 14]))),
        ("bb".to_string(), None),
        ("cc".to_string(), None),
    ]
    .into_iter()
    .collect();
    assert_eq!(m, want);
}

/// `kreg-input-sets/v1` → kernel id → hash: a file for this commit is read whole; another schema, another
/// commit, a missing `input_sets` or one bad hash refuses it all. Fed to the judge, a listed kernel whose
/// receipt carries that hash is fresh and an unlisted one is stale (RR2-F4 via the real input).
#[test]
fn input_sets_are_read_whole_for_the_release_commit_only() {
    let h = "a".repeat(64);
    let doc = |schema: &str, at: &str, hash: &str| {
        format!(
            r#"{{"schema":"{schema}","build_identity":"{at}","reuse":{{"fresh":1,"total":1}},
               "input_sets":{{"cpu.matvec.q2_k":{{"receipt":"r.json","input_set_hash":"{hash}","stale":[]}}}}}}"#
        )
    };
    let sets = parse_input_sets(
        "s.json",
        doc(INPUT_SETS_SCHEMA, "abc", &h).as_bytes(),
        "abc",
    )
    .expect("a sets file for this commit");
    assert_eq!(sets.get("cpu.matvec.q2_k"), Some(&h));
    for (bad, why) in [
        (doc("kreg-input-sets/v0", "abc", &h), "schema"),
        (doc(INPUT_SETS_SCHEMA, "def", &h), "release commit"),
        (doc(INPUT_SETS_SCHEMA, "abc", "abc"), "64 hex"),
        (
            r#"{"schema":"kreg-input-sets/v1","build_identity":"abc"}"#.to_string(),
            "input_sets",
        ),
        ("not json".to_string(), "JSON"),
    ] {
        let err = parse_input_sets("s.json", bad.as_bytes(), "abc").expect_err(why);
        assert!(err.to_string().contains(why), "{why}: {err}");
    }
    assert!(parse_input_sets("s.json", doc(INPUT_SETS_SCHEMA, "", &h).as_bytes(), "").is_err());

    let receipt = Q2K.replace(
        r#""schema""#,
        &format!(r#""input_set_hash":"{h}","schema""#),
    );
    let now = |id: &str| sets.get(id).map_or("", String::as_str);
    let (_, fresh) = judge_parity_receipt(
        "r.json",
        receipt.as_bytes(),
        "cpu",
        "x86_64",
        now("cpu.matvec.q2_k"),
    )
    .expect("judged");
    assert!(fresh.fresh);
    let (_, stale) = judge_parity_receipt(
        "r.json",
        receipt.as_bytes(),
        "cpu",
        "x86_64",
        now("cpu.matvec.q4_k"),
    )
    .expect("judged");
    assert!(!stale.fresh);
}
