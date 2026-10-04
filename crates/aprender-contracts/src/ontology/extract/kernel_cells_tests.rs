use super::*;

const F: &str = "kernel-registry.json";

fn row(id: &str, backend: &str, t: u32, layout: &str, arch: &str) -> String {
    format!(
        r#"{{"kernel_id":"{id}","backend":"{backend}","ggml_type":{t},"layout":"{layout}","arch":"{arch}"}}"#
    )
}

/// A registry document: typed rows go to `kernels[]`, op rows (no `ggml_type`) to `ops[]`.
fn doc(rows: &[String]) -> String {
    let (kernels, ops): (Vec<&String>, Vec<&String>) =
        rows.iter().partition(|r| r.contains("\"ggml_type\""));
    let join = |v: Vec<&String>| v.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(",");
    format!(r#"{{"kernels":[{}],"ops":[{}]}}"#, join(kernels), join(ops))
}

fn registry(rows: &[String]) -> Vec<RegistryRow> {
    parse_registry(F, doc(rows).as_bytes()).expect("fixture registry parses")
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

fn op_row(id: &str, backend: &str, arch: &str, archs: Option<&str>) -> String {
    let archs = archs.map_or_else(String::new, |a| format!(r#","archs":{a}"#));
    format!(r#"{{"kernel_id":"{id}","backend":"{backend}","arch":"{arch}"{archs}}}"#)
}

fn shape(ts: &[u32], arch: Option<&str>) -> ModelShape {
    ModelShape {
        types: types(ts),
        arch: arch.map(str::to_string),
    }
}

/// #3715 P1: per-forward ops (`ops[]` rows) are on every model's path of their backend and host arch,
/// narrowed by `archs` when the model's architecture is known; unknown takes them all. They never serve a
/// tensor type, and a model with no types gets none, so it stays RED.
#[test]
fn op_rows_join_every_model_map_of_their_backend() {
    let mut raw = vec![
        row("cpu.matvec.q4_k", "cpu", 12, ROW_MAJOR, ANY_ARCH),
        op_row("cpu.rmsnorm.f32", "cpu", ANY_ARCH, None),
        op_row("cpu.layernorm.f32", "cpu", ANY_ARCH, Some(r#"["phi2"]"#)),
        op_row("cpu.rope.f32.avx512", "cpu", "x86_64", None),
        op_row("cuda.rmsnorm.f32", "cuda", ANY_ARCH, None),
    ];
    let rows = registry(&raw);
    let map = |host_arch, m: &ModelShape| model_kernel_map(&rows, "cpu", host_arch, m);
    let q = shape(&[12], Some("qwen2"));
    assert_eq!(
        map("x86_64", &q).uses,
        ids(&["cpu.matvec.q4_k", "cpu.rmsnorm.f32", "cpu.rope.f32.avx512"])
    );
    assert_eq!(
        map("aarch64", &q).uses,
        ids(&["cpu.matvec.q4_k", "cpu.rmsnorm.f32"])
    );
    assert_eq!(
        map("aarch64", &shape(&[12], Some("phi2"))).uses,
        ids(&["cpu.layernorm.f32", "cpu.matvec.q4_k", "cpu.rmsnorm.f32"])
    );
    // Unknown architecture: the superset.
    assert_eq!(
        map("aarch64", &shape(&[12], None)).uses,
        ids(&["cpu.layernorm.f32", "cpu.matvec.q4_k", "cpu.rmsnorm.f32"])
    );
    // No tensor types: no op rows, nothing to satisfy minCount 1.
    assert_eq!(
        map("x86_64", &shape(&[], Some("qwen2"))),
        KernelMap::default()
    );
    // An op row serves no tensor type: an f32 tensor with no typed row is still unregistered.
    assert_eq!(
        map("x86_64", &shape(&[0, 12], None)).unregistered,
        types(&[0])
    );
    // Removing an op row drops exactly that kernel from every map.
    raw.retain(|r| !r.contains("cpu.rmsnorm.f32"));
    let without = registry(&raw);
    assert_eq!(
        model_kernel_map(&without, "cpu", "aarch64", &q).uses,
        ids(&["cpu.matvec.q4_k"])
    );
}

/// The op-row fields are read whole or the registry is refused.
#[test]
fn op_row_fields_are_refused_when_malformed() {
    let refused = |json: String, want: &str| {
        let e = parse_registry(F, json.as_bytes()).expect_err(want);
        assert!(e.what.contains(want), "{} !~ {want}", e.what);
    };
    let typed = row("t", "cpu", 12, ROW_MAJOR, ANY_ARCH);
    let with_ops = |op: String| format!(r#"{{"kernels":[{typed}],"ops":[{op}]}}"#);
    // A registry from before ops[] is refused, not read as having no ops.
    refused(format!(r#"{{"kernels":[{typed}]}}"#), "no `ops` array");
    refused(
        format!(r#"{{"kernels":[{typed}],"ops":{{}}}}"#),
        "no `ops` array",
    );
    // A type key on an op row is refused, not ignored.
    for (k, v) in [
        ("ggml_type", "null"),
        ("ggml_type", "0"),
        ("qtype", r#""F32""#),
        ("layout", r#""row_major""#),
    ] {
        let op = op_row("k", "cpu", ANY_ARCH, None).replace('}', &format!(r#","{k}":{v}}}"#));
        refused(with_ops(op), &format!("`{k}` on an op row"));
    }
    refused(
        with_ops(r#"{"kernel_id":"k","arch":"any"}"#.to_string()),
        "ops[0]: `backend` missing",
    );
    // A typed row with no, or a non-u32, type is refused, never read as an op.
    refused(
        format!(
            r#"{{"kernels":[{{"kernel_id":"k","backend":"cpu","layout":"{ROW_MAJOR}","arch":"any"}}],"ops":[]}}"#
        ),
        "`ggml_type` missing",
    );
    refused(
        format!(
            r#"{{"kernels":[{{"kernel_id":"k","backend":"cpu","ggml_type":null,"layout":"{ROW_MAJOR}","arch":"any"}}],"ops":[]}}"#
        ),
        "`ggml_type` missing",
    );
    let typed_with_archs = typed.replace('}', r#","archs":["qwen2"]}"#);
    refused(
        format!(r#"{{"kernels":[{typed_with_archs}],"ops":[]}}"#),
        "`archs` on a typed row",
    );
    for archs in [r#"[]"#, r#""qwen2""#, r#"[""]"#, r#"[1]"#, r#"["a","a"]"#] {
        refused(
            with_ops(op_row("k", "cpu", ANY_ARCH, Some(archs))),
            "`archs` not a non-empty list",
        );
    }
    // An id is one row across kernels[] and ops[].
    refused(
        with_ops(op_row("t", "cpu", ANY_ARCH, None)),
        "duplicate kernel_id `t`",
    );
    let rows = registry(&[typed, op_row("k", "cpu", ANY_ARCH, Some(r#"["a","b"]"#))]);
    let k = rows
        .iter()
        .find(|r| r.kernel_id == "k")
        .expect("the op row is read");
    assert_eq!(k.ggml_type, None);
    assert_eq!(k.archs, Some(ids(&["a", "b"])));
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
        ("duplicate kernel_id `a`", format!(r#"{{"kernels":[{good},{good}],"ops":[]}}"#)),
        (
            "`ggml_type` missing",
            r#"{"kernels":[{"kernel_id":"a","backend":"cpu","layout":"row_major","arch":"any"}],"ops":[]}"#
                .to_string(),
        ),
        (
            "`ggml_type` missing",
            r#"{"kernels":[{"kernel_id":"a","backend":"cpu","ggml_type":4294967296,"layout":"row_major","arch":"any"}],"ops":[]}"#
                .to_string(),
        ),
        (
            "`backend` missing",
            r#"{"kernels":[{"kernel_id":"a","backend":"","ggml_type":12,"layout":"row_major","arch":"any"}],"ops":[]}"#
                .to_string(),
        ),
        (
            "`arch` missing",
            r#"{"kernels":[{"kernel_id":"a","backend":"cpu","ggml_type":12,"layout":"row_major"}],"ops":[]}"#
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
        assert_eq!(s.len(), 3, "the kernel, model and sanitizer shapes");
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
        /// RR2-F3's precondition: model B's smoke names no registry kernel path, so the map was never checked.
        NoKernelPath,
        /// RR2-F6: the kernel's receipt was measured on another arch.
        ForeignArch(&'static str),
        /// S-SAN: no sanitizer run covers the kernel.
        NoSan(&'static str),
        /// S-SAN: the covering sanitizer run found something.
        DirtySan(&'static str),
        /// S-SAN: the covering sanitizer run is older than 7 days.
        StaleSan(&'static str),
    }

    const SAN: SanitizerEvidence = SanitizerEvidence {
        clean: true,
        fresh: true,
    };

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
                kernel_path_known: !(m == "B" && matches!(plant, Some(Plant::NoKernelPath))),
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
            let san = match plant {
                Some(Plant::NoSan(x)) if x == k => None,
                Some(Plant::DirtySan(x)) if x == k => Some(SanitizerEvidence {
                    clean: false,
                    ..SAN
                }),
                Some(Plant::StaleSan(x)) if x == k => Some(SanitizerEvidence {
                    fresh: false,
                    ..SAN
                }),
                _ => Some(SAN),
            };
            emit_sanitizer(&mut g, HOST, k, san);
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
            ("F3 no kernel path", Plant::NoKernelPath, vec![mc("B")]),
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

    /// S-SAN: a kernel with no, a dirty, or a stale sanitizer run is RED on its own kernel cell. Model
    /// cells are not derived through it: the v2.model node reads the parity fields only, so a CPU kernel
    /// (no sanitizer) never needs one.
    #[test]
    fn rr2_s_san_each_red_names_only_its_kernel_cell() {
        let k = "cuda.gemv.q6_k";
        for (name, plant) in [
            ("no run", Plant::NoSan(k)),
            ("dirty", Plant::DirtySan(k)),
            ("stale", Plant::StaleSan(k)),
        ] {
            assert_eq!(red(Some(plant)), set(&[kc(k)]), "{name}");
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
            kernel_path_known: true,
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
        let mut models: BTreeMap<String, Option<ModelShape>> = BTreeMap::new();
        for (m, ts) in MODELS {
            let ts = if m == "C" { types_of_c } else { Some(ts) };
            let shape = ts.map(|ts| ModelShape {
                types: types(ts),
                arch: None,
            });
            models.insert(m.to_string(), shape);
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
                .map(|(m, ts)| {
                    let s = SmokeReceipt {
                        pass: true,
                        fresh: true,
                        dispatched: Some(
                            static_kernel_map(&rows, "cuda", "sm_89", &types(ts)).uses,
                        ),
                    };
                    ((*m).to_string(), s)
                })
                .collect(),
            sanitized: rows.iter().map(|r| (r.kernel_id.clone(), SAN)).collect(),
        }
    }

    fn built_red(h: &CellHost) -> BTreeSet<String> {
        built_red_with(&registry(), h)
    }

    fn built_red_with(rows: &[RegistryRow], h: &CellHost) -> BTreeSet<String> {
        let mut g = Graph::new();
        build_cells(&mut g, rows, std::slice::from_ref(h));
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

    /// #3715 P1, RR2-F1 for a per-forward op: every model with tensor types uses the op row, so removing its
    /// receipt fails every one of them and that kernel cell; a model with no types was RED already.
    #[test]
    fn build_cells_f1_for_an_op_row_reds_every_typed_model() {
        const OP: &str = "cuda.rmsnorm.f32";
        let mut rows = registry();
        // A registry needs a typed row; only the op row joins the fixture.
        let one = super::registry(&[
            row("x.typed", "cpu", 12, ROW_MAJOR, ANY_ARCH),
            op_row(OP, "cuda", ANY_ARCH, None),
        ]);
        rows.extend(one.into_iter().filter(|r| r.kernel_id == OP));
        let mut h = cell_host(Some(&[8]), None);
        h.kernels.insert(OP.to_string(), PASS);
        h.sanitized.insert(OP.to_string(), SAN);
        assert_eq!(built_red_with(&rows, &h), BTreeSet::new());
        h.kernels.remove(OP);
        let want: BTreeSet<String> = [mc("A"), mc("B"), mc("C"), kc(OP)].into_iter().collect();
        assert_eq!(built_red_with(&rows, &h), want);
        let mut h = cell_host(None, None);
        h.kernels.insert(OP.to_string(), PASS);
        h.sanitized.insert(OP.to_string(), SAN);
        assert_eq!(built_red_with(&rows, &h), [mc("C")].into_iter().collect());
    }

    /// Through the builder, a smoke receipt is judged against its own model's static map: B (q4_k only)
    /// dispatching q6_k is unpredicted and RED, the same id is predicted for A; a `trace` or null path is
    /// RED on B alone.
    #[test]
    fn build_cells_judges_each_smoke_against_its_own_model() {
        let mut h = cell_host(Some(&[8]), None);
        let b = h.smokes.get_mut("B").expect("B");
        b.dispatched = Some(ids(&["cuda.gemv.q4_k", "cuda.gemv.q6_k"]));
        assert_eq!(built_red(&h), [mc("B")].into_iter().collect());
        let mut h = cell_host(Some(&[8]), None);
        h.smokes.get_mut("B").expect("B").dispatched = None;
        assert_eq!(built_red(&h), [mc("B")].into_iter().collect());
    }

    /// `build_cells` asks for a sanitizer run on cuda hosts only: a kernel with none is RED on cuda and
    /// green on a cpu host with the same receipts.
    #[test]
    fn build_cells_sanitizer_is_cuda_only() {
        // (the smoke of C below dispatches the cpu kernel: a cuda id would be unpredicted there)
        let mut h = cell_host(Some(&[8]), None);
        h.sanitized.remove("cuda.gemv.q8_0");
        assert_eq!(built_red(&h), [kc("cuda.gemv.q8_0")].into_iter().collect());
        // The same host on cpu, with a cpu row serving q8_0, a parity receipt, and no sanitizer run.
        let rows = super::registry(&[row("cpu.matvec.q8_0", "cpu", 8, ROW_MAJOR, ANY_ARCH)]);
        h.backend = "cpu".to_string();
        h.models.retain(|m, _| m == "C");
        h.kernels = [("cpu.matvec.q8_0".to_string(), PASS)]
            .into_iter()
            .collect();
        h.sanitized.clear();
        h.smokes.get_mut("C").expect("C's smoke").dispatched = Some(ids(&["cpu.matvec.q8_0"]));
        let mut g = Graph::new();
        build_cells(&mut g, &rows, std::slice::from_ref(&h));
        let report = validate(&g, &shapes());
        assert!(
            g.iter().any(|t| t.subject == kc("cpu.matvec.q8_0")),
            "the cpu kernel cell was built: not vacuous"
        );
        assert!(report.conforms(), "{:#?}", report.results);
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
          {"file":"a.gguf","sha256":"AA","arch":"qwen2","tensor_types":[12,14,0]},
          {"file":"b.gguf","sha256":"bb","tensor_types":[12,"q6_k"]},
          {"file":"c.gguf","sha256":"cc"},
          {"file":"d.gguf","tensor_types":[8]}]}"#,
    )
    .expect("a v2 receipt");
    let m = models_from_inventory(&r.inventory);
    let want: BTreeMap<String, Option<ModelShape>> = [
        (
            "aa".to_string(),
            Some(ModelShape {
                types: types(&[0, 12, 14]),
                arch: Some("qwen2".to_string()),
            }),
        ),
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

/// A real KTEST-05 receipt (evidence/ktest-05/2026-09-28/gx10/san/receipt.json on
/// la-71/ktest-05-cuda-arch-sanitizer), trimmed to the judged fields.
const SAN_RC: &str = r#"{"schema":"ktest-05-sanitizer-receipt-v1","utc":"2026-09-28T12:53:20Z","tools":[
  {"tool":"memcheck","verdict":"CLEAN"},{"tool":"racecheck","verdict":"CLEAN"},
  {"tool":"initcheck","verdict":"CLEAN"},{"tool":"synccheck","verdict":"CLEAN"}],"clean":true}"#;

/// Case table for `judge_sanitizer_receipt`: a missing, duplicated or RED tool is not clean whatever the
/// receipt's own `clean` says; the window is 7 days, closed, and a run dated after the gate is stale.
#[test]
fn the_sanitizer_judge_case_table() {
    let now = "2026-09-28T16:00:00Z";
    let ok = SanitizerEvidence {
        clean: true,
        fresh: true,
    };
    let dirty = SanitizerEvidence { clean: false, ..ok };
    let stale = SanitizerEvidence { fresh: false, ..ok };
    let no_sync = SAN_RC.replace(r#",{"tool":"synccheck","verdict":"CLEAN"}"#, "");
    let cases: Vec<(&str, String, &str, SanitizerEvidence)> = vec![
        ("the gx10 receipt", SAN_RC.to_string(), now, ok),
        ("a tool missing, clean:true kept", no_sync, now, dirty),
        (
            "all four, plus a second memcheck that is RED",
            SAN_RC.replace(
                r#"{"tool":"memcheck","verdict":"CLEAN"}"#,
                r#"{"tool":"memcheck","verdict":"CLEAN"},{"tool":"memcheck","verdict":"RED"}"#,
            ),
            now,
            dirty,
        ),
        (
            "memcheck RED",
            SAN_RC.replacen(r#""CLEAN""#, r#""RED""#, 1),
            now,
            dirty,
        ),
        (
            "no summary",
            SAN_RC.replacen(r#""CLEAN""#, r#""RED_NO_SUMMARY""#, 1),
            now,
            dirty,
        ),
        (
            "initcheck advisory",
            SAN_RC.replace(
                r#""initcheck","verdict":"CLEAN""#,
                r#""initcheck","verdict":"ADVISORY_RED""#,
            ),
            now,
            ok,
        ),
        (
            "exactly 7 days old",
            SAN_RC.to_string(),
            "2026-10-05T12:53:20Z",
            ok,
        ),
        (
            "7 days + 1 s",
            SAN_RC.to_string(),
            "2026-10-05T12:53:21Z",
            stale,
        ),
        (
            "dated after the gate",
            SAN_RC.to_string(),
            "2026-09-28T12:53:19Z",
            stale,
        ),
        (
            "no utc",
            SAN_RC.replace(r#""utc""#, r#""when""#),
            now,
            stale,
        ),
        (
            "utc not a stamp",
            SAN_RC.replace("2026-09-28T12:53:20Z", "2026-09-28 12:53"),
            now,
            stale,
        ),
    ];
    for (name, json, at, want) in cases {
        let got = judge_sanitizer_receipt("receipt.json", json.as_bytes(), at).expect(name);
        assert_eq!(got, want, "{name}");
    }
    for (name, json, at) in [
        ("not JSON", "{".to_string(), now),
        (
            "another schema",
            SAN_RC.replace("ktest-05-sanitizer-receipt-v1", "v0"),
            now,
        ),
        ("no tools", SAN_RC.replace(r#""tools""#, r#""t""#), now),
        ("gate time unreadable", SAN_RC.to_string(), "today"),
    ] {
        assert!(
            judge_sanitizer_receipt("receipt.json", json.as_bytes(), at).is_err(),
            "{name}"
        );
    }
    // Month and leap-year arithmetic: 2028-02-28 → 2028-03-01 is 2 days.
    let leap = SAN_RC.replace("2026-09-28T12:53:20Z", "2028-02-28T00:00:00Z");
    let e = judge_sanitizer_receipt("r", leap.as_bytes(), "2028-03-01T00:00:00Z").expect("leap");
    assert!(e.fresh);
    assert_eq!(utc_seconds("1970-01-01T00:00:00Z"), Some(0));
    assert_eq!(utc_seconds("2000-03-01T00:00:00Z"), Some(951_868_800));
}

/// `read_host_kernels`: each receipt judged against its own kernel's input set; unlisted → stale; a second
/// receipt for one kernel, or a non-receipt, refuses the directory; an absent directory is empty.
#[test]
fn read_host_kernels_judges_each_receipt_against_its_own_key() {
    let dir = tempfile::tempdir().expect("tempdir");
    let q2k = Q2K.replace(
        r#""tolerance_rel""#,
        r#""input_set_hash":"h2","tolerance_rel""#,
    );
    let q3k = q2k
        .replace("cpu.matvec.q2_k", "cpu.matvec.q3_k")
        .replace("h2", "h3");
    std::fs::write(dir.path().join("a.json"), &q2k).expect("write");
    std::fs::write(dir.path().join("b.json"), &q3k).expect("write");
    std::fs::write(dir.path().join("notes.txt"), "not read").expect("write");
    let sets: BTreeMap<String, String> = [("cpu.matvec.q2_k".to_string(), "h2".to_string())]
        .into_iter()
        .collect();
    let got = read_host_kernels(dir.path(), "cpu", "x86_64", &sets).expect("a receipt dir");
    assert_eq!(got.len(), 2);
    assert!(got["cpu.matvec.q2_k"].fresh, "listed with the same key");
    assert!(
        !got["cpu.matvec.q3_k"].fresh,
        "unlisted in the input sets: stale"
    );
    assert!(got["cpu.matvec.q2_k"].pass && got["cpu.matvec.q2_k"].arch_match);
    // Same kernel twice.
    std::fs::write(dir.path().join("c.json"), &q2k).expect("write");
    assert!(read_host_kernels(dir.path(), "cpu", "x86_64", &sets).is_err());
    std::fs::remove_file(dir.path().join("c.json")).expect("rm");
    // A json file that is not a parity receipt.
    std::fs::write(dir.path().join("d.json"), "{}").expect("write");
    assert!(read_host_kernels(dir.path(), "cpu", "x86_64", &sets).is_err());
    let none =
        read_host_kernels(&dir.path().join("absent"), "cpu", "x86_64", &sets).expect("absent");
    assert!(none.is_empty());
}

const SMOKE_SHA: &str = "abababababababababababababababababababababababababababababababab";

fn smoke_json(host: &str, apr_sha: &str, verdict: &str, kernel_path: &str) -> String {
    format!(
        r#"{{"schema":"{SMOKE_SCHEMA}","host":"{host}","model_sha256":"{SMOKE_SHA}","apr_sha":"{apr_sha}",
            "verdict":"{verdict}","kernel_path":{kernel_path}}}"#
    )
}

fn kpath(source: &str, ids: &[&str]) -> String {
    let entries: Vec<String> = ids
        .iter()
        .map(|k| {
            format!(
                r#"{{"op":"gemv","kernel_id":"{k}","qtype":"q4_k","layout":"row_major","arch":"sm_89","shape_class":"m1","precision":"f32"}}"#
            )
        })
        .collect();
    format!(
        r#"{{"source":"{source}","entries":[{}]}}"#,
        entries.join(",")
    )
}

/// RR2-F3's input: the smoke judge. `kernel_path` is the `apr-kernel-path-v1` object (OBS-15, aprender#4574).
#[test]
fn the_smoke_judge_case_table() {
    let judge = |json: &str| judge_smoke_receipt("s.json", json.as_bytes(), "lambda", "mc");
    let ok = |json: &str| judge(json).expect("a smoke receipt").1;
    let good = ok(&smoke_json(
        "lambda",
        "mc",
        "pass",
        &kpath("kreg", &["cuda.gemv.q4_k", "cuda.gemv.q6_k"]),
    ));
    assert_eq!(
        (good.pass, good.fresh, good.dispatched),
        (true, true, Some(ids(&["cuda.gemv.q4_k", "cuda.gemv.q6_k"])))
    );
    assert_eq!(
        judge(&smoke_json("lambda", "mc", "pass", "null"))
            .expect("null path")
            .0,
        SMOKE_SHA
    );
    for (json, want, why) in [
        (
            smoke_json("lambda", "old", "pass", "null"),
            (true, false, None),
            "another commit: stale",
        ),
        (
            smoke_json("lambda", "mc", "fail", "null"),
            (false, true, None),
            "verdict fail",
        ),
        (
            smoke_json("lambda", "mc", "pass", &kpath("trace", &["q4k_gemv@q4_k"])),
            (true, true, None),
            "trace labels are not registry ids",
        ),
        (
            smoke_json("lambda", "mc", "pass", "null").replace(r#","kernel_path":null"#, ""),
            (true, true, None),
            "absent path",
        ),
    ] {
        let r = ok(&json);
        assert_eq!((r.pass, r.fresh, r.dispatched), want, "{why}");
    }
    let good_json = smoke_json("lambda", "mc", "pass", &kpath("kreg", &["cuda.gemv.q4_k"]));
    for (bad, why) in [
        ("[]".to_string(), "not an object"),
        (
            good_json.replace(SMOKE_SCHEMA, "rr2-smoke-receipt/v0"),
            "schema",
        ),
        (
            smoke_json("gx10", "mc", "pass", "null"),
            "another host's receipt",
        ),
        (good_json.replace(SMOKE_SHA, "AB"), "short sha"),
        (
            good_json.replace(SMOKE_SHA, &SMOKE_SHA.to_uppercase()),
            "uppercase sha",
        ),
        (good_json.replace(r#""apr_sha":"mc","#, ""), "no apr_sha"),
        (good_json.replace(r#""verdict":"pass","#, ""), "no verdict"),
        (
            smoke_json("lambda", "mc", "pass", "7"),
            "path not an object",
        ),
        (
            smoke_json("lambda", "mc", "pass", &kpath("guess", &["x"])),
            "unknown source",
        ),
        (
            smoke_json("lambda", "mc", "pass", r#"{"source":"kreg","entries":[]}"#),
            "no entries",
        ),
        (
            smoke_json("lambda", "mc", "pass", &kpath("kreg", &["unknown"])),
            "entry names no kernel",
        ),
        (
            smoke_json("lambda", "mc", "pass", &kpath("kreg", &[""])),
            "empty kernel id",
        ),
    ] {
        assert!(judge(&bad).is_err(), "{why} must be refused");
    }
    // `against`: unpredicted is dispatched minus uses; no path is known-false, never "ran nothing".
    let uses = ids(&["cuda.gemv.q4_k"]);
    let r = SmokeReceipt {
        pass: true,
        fresh: true,
        dispatched: Some(ids(&["cuda.gemv.q4_k", "cuda.gemv.q6_k"])),
    };
    let j = r.against(&uses);
    assert!(j.kernel_path_known);
    assert_eq!(j.unpredicted, ids(&["cuda.gemv.q6_k"]));
    let j = SmokeReceipt {
        dispatched: None,
        ..r
    }
    .against(&uses);
    assert!(!j.kernel_path_known && j.unpredicted.is_empty());
}

#[test]
fn read_host_smokes_keys_by_model_and_refuses_a_second_receipt() {
    let dir = tempfile::tempdir().expect("tempdir");
    let json = smoke_json("lambda", "mc", "pass", &kpath("kreg", &["cuda.gemv.q4_k"]));
    std::fs::write(dir.path().join("a.json"), &json).expect("write");
    std::fs::write(dir.path().join("notes.txt"), "not read").expect("write");
    let got = read_host_smokes(dir.path(), "lambda", "mc").expect("a smoke dir");
    assert_eq!(got.keys().collect::<Vec<_>>(), vec![SMOKE_SHA]);
    std::fs::write(dir.path().join("b.json"), &json).expect("write");
    assert!(
        read_host_smokes(dir.path(), "lambda", "mc").is_err(),
        "two receipts, one model"
    );
    assert!(read_host_smokes(&dir.path().join("absent"), "lambda", "mc")
        .expect("absent")
        .is_empty());
}

/// A v2 sanitizer run on lambda: `path` is its `kernel_path`, `rows` its tool rows.
fn san_run(host: &str, utc: &str, path: &str, rows: &[String]) -> String {
    format!(
        r#"{{"schema":"{SANITIZER_SCHEMA_V2}","host":"{host}","utc":"{utc}","kernel_path":{path},"tools":[{}]}}"#,
        rows.join(",")
    )
}

fn san_row(tool: &str, verdict: &str, filter: &str, covers: Option<&[&str]>) -> String {
    let covers = covers.map_or_else(String::new, |ks| {
        let q: Vec<String> = ks.iter().map(|k| format!("\"{k}\"")).collect();
        format!(r#","covers":[{}]"#, q.join(","))
    });
    format!(r#"{{"tool":"{tool}","verdict":"{verdict}","filter":"{filter}"{covers}}}"#)
}

/// The four tools, unfiltered, with `racecheck` given as `race`.
fn san_rows(race: String) -> Vec<String> {
    vec![
        san_row("memcheck", "CLEAN", "none", None),
        race,
        san_row("initcheck", "ADVISORY_RED", "none", None),
        san_row("synccheck", "CLEAN", "none", None),
    ]
}

/// S-SAN attribution: a run checks the kernels its `kreg` path dispatched, and a `--kernel-name`-filtered
/// tool checks only the ids in its `covers`.
#[test]
fn the_sanitizer_run_judge_case_table() {
    const Q4: &str = "cuda.gemv.q4_k";
    const ROPE: &str = "cuda.rope.f32";
    let path = kpath("kreg", &[Q4, ROPE]);
    let now = Some("2026-09-28T16:00:00Z");
    let judge = |json: &str| judge_sanitizer_run("r.json", json.as_bytes(), "lambda", now);

    let run = judge(&san_run(
        "lambda",
        "2026-09-28T12:00:00Z",
        &path,
        &san_rows(san_row("racecheck", "CLEAN", "regex=rope", Some(&[ROPE]))),
    ))
    .expect("judged");
    assert!(run.fresh);
    assert_eq!(
        run.tools["memcheck"],
        (true, ids(&[Q4, ROPE])),
        "unfiltered: every dispatched kernel"
    );
    assert_eq!(
        run.tools["racecheck"],
        (true, ids(&[ROPE])),
        "filtered: only covers"
    );
    assert!(
        run.tools["initcheck"].0,
        "ADVISORY_RED is the recorded advisory policy"
    );

    let no_covers = judge(&san_run(
        "lambda",
        "2026-09-28T12:00:00Z",
        &path,
        &san_rows(san_row("racecheck", "CLEAN", "regex=rope", None)),
    ))
    .expect("judged");
    assert!(
        no_covers.tools["racecheck"].1.is_empty(),
        "a filter of unknown reach checked nothing"
    );

    let dirty = judge(&san_run(
        "lambda",
        "2026-09-28T12:00:00Z",
        &path,
        &san_rows(san_row("racecheck", "RED", "none", None)),
    ))
    .expect("judged");
    assert!(!dirty.tools["racecheck"].0);

    let old = san_run(
        "lambda",
        "2026-09-20T12:00:00Z",
        &path,
        &san_rows(san_row("racecheck", "CLEAN", "none", None)),
    );
    assert!(!judge(&old).expect("judged").fresh, "8 days old");
    assert!(
        !judge_sanitizer_run("r.json", old.as_bytes(), "lambda", None)
            .expect("judged")
            .fresh,
        "no gate time: never fresh"
    );

    let clean = san_rows(san_row("racecheck", "CLEAN", "none", None));
    for (name, json) in [
        (
            "another host",
            san_run("gx10", "2026-09-28T12:00:00Z", &path, &clean),
        ),
        ("v1 schema", SAN_RC.to_string()),
        (
            "trace path",
            san_run(
                "lambda",
                "2026-09-28T12:00:00Z",
                &kpath("trace", &["q4k_gemv@q4_k"]),
                &clean,
            ),
        ),
        (
            "null path",
            san_run("lambda", "2026-09-28T12:00:00Z", "null", &clean),
        ),
        (
            "covers a kernel not dispatched",
            san_run(
                "lambda",
                "2026-09-28T12:00:00Z",
                &path,
                &san_rows(san_row(
                    "racecheck",
                    "CLEAN",
                    "regex=x",
                    Some(&["cuda.gemv.q6_k"]),
                )),
            ),
        ),
        (
            "a tool twice",
            san_run(
                "lambda",
                "2026-09-28T12:00:00Z",
                &path,
                &[
                    clean.clone(),
                    vec![san_row("memcheck", "CLEAN", "none", None)],
                ]
                .concat(),
            ),
        ),
        (
            "a row with no filter",
            san_run(
                "lambda",
                "2026-09-28T12:00:00Z",
                &path,
                &[r#"{"tool":"memcheck","verdict":"CLEAN"}"#.to_string()],
            ),
        ),
    ] {
        assert!(judge(&json).is_err(), "{name} must be refused");
    }
}

/// A kernel is clean only when every tool checked it somewhere and no check of it was dirty; fresh only
/// when every tool checked it in a fresh run. A kernel no run checked has no evidence.
#[test]
fn attribution_needs_every_tool_to_check_the_kernel() {
    let run = |fresh: bool, tools: &[(&str, bool, &[&str])]| SanitizerRun {
        fresh,
        tools: tools
            .iter()
            .map(|(t, ok, ks)| ((*t).to_string(), (*ok, ids(ks))))
            .collect(),
    };
    let all = |ok: bool, ks: &'static [&'static str]| {
        SANITIZER_TOOLS
            .iter()
            .map(move |t| (*t, ok, ks))
            .collect::<Vec<_>>()
    };
    let ev = |clean, fresh| SanitizerEvidence { clean, fresh };

    // racecheck filtered to rope: gemv is not clean, rope is.
    let got = attribute_sanitizer_runs(&[run(
        true,
        &[
            ("memcheck", true, &["gemv", "rope"][..]),
            ("racecheck", true, &["rope"][..]),
            ("initcheck", true, &["gemv", "rope"][..]),
            ("synccheck", true, &["gemv", "rope"][..]),
        ],
    )]);
    assert_eq!(got["rope"], ev(true, true));
    assert_eq!(
        got["gemv"],
        ev(false, false),
        "no racecheck ever checked it"
    );

    // A second, gemv-only racecheck run clears it; a stale one clears clean but not fresh.
    let base = [
        ("memcheck", true, &["gemv"][..]),
        ("racecheck", true, &[][..]),
        ("initcheck", true, &["gemv"][..]),
        ("synccheck", true, &["gemv"][..]),
    ];
    let race = |fresh| run(fresh, &[("racecheck", true, &["gemv"][..])]);
    assert_eq!(
        attribute_sanitizer_runs(&[run(true, &base), race(true)])["gemv"],
        ev(true, true)
    );
    assert_eq!(
        attribute_sanitizer_runs(&[run(true, &base), race(false)])["gemv"],
        ev(true, false)
    );

    // One dirty check anywhere is not clean, whatever the other runs say.
    let got = attribute_sanitizer_runs(&[
        run(true, &all(true, &["gemv"])),
        run(true, &[("memcheck", false, &["gemv"][..])]),
    ]);
    assert_eq!(got["gemv"], ev(false, true));

    assert!(
        !attribute_sanitizer_runs(&[run(true, &all(true, &["gemv"]))]).contains_key("rope"),
        "a kernel no run checked has no evidence"
    );
    assert!(attribute_sanitizer_runs(&[]).is_empty());
}

#[test]
fn read_host_sanitizers_attributes_the_directory() {
    let t = tempfile::tempdir().expect("tmp");
    let dir = t.path().join("sanitizer");
    let now = Some("2026-09-28T16:00:00Z");
    assert!(read_host_sanitizers(&dir, "lambda", now)
        .expect("absent dir")
        .is_empty());
    std::fs::create_dir_all(&dir).expect("mkdir");
    let path = kpath("kreg", &["cuda.gemv.q4_k"]);
    std::fs::write(
        dir.join("a.json"),
        san_run(
            "lambda",
            "2026-09-28T12:00:00Z",
            &path,
            &san_rows(san_row("racecheck", "CLEAN", "none", None)),
        ),
    )
    .expect("write");
    let got = read_host_sanitizers(&dir, "lambda", now).expect("read");
    assert_eq!(
        got["cuda.gemv.q4_k"],
        SanitizerEvidence {
            clean: true,
            fresh: true
        }
    );
    std::fs::write(dir.join("b.json"), SAN_RC).expect("write v1");
    assert!(
        read_host_sanitizers(&dir, "lambda", now).is_err(),
        "a v1 run names no kernel: refused"
    );
}
