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

/// Case table: (backend, host arch, model types) → (kernels used, unregistered types).
#[test]
fn the_static_map_case_table() {
    let rows = excerpt();
    let cases: &[(&str, &[u32], &[&str], &[u32])] = &[
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

/// Edges and literals only; no verdict predicate is written.
#[test]
fn emit_writes_edges_and_unregistered_literals_only() {
    let mut g = Graph::new();
    let cell = iri("model-cell", "lambda/qwen");
    let m = static_kernel_map(&excerpt(), "cuda", "x86_64", &types(&[11, 12, 14]));
    emit_model_cell(&mut g, &cell, &m);
    let used: BTreeSet<&str> = g
        .objects(&cell, &rel("usesKernel"))
        .into_iter()
        .filter_map(Term::as_iri)
        .collect();
    assert_eq!(
        used,
        [kernel_cell("cuda.gemv.q4_k"), kernel_cell("cuda.gemv.q6_k")]
            .iter()
            .map(String::as_str)
            .collect()
    );
    let unreg = g.objects(&cell, &rel("unregisteredQtype"));
    assert_eq!(unreg, vec![&Term::integer(11)]);
    assert_eq!(
        g.predicates_of(&cell),
        [rel("usesKernel"), rel("unregisteredQtype")]
            .iter()
            .map(String::as_str)
            .collect()
    );
}
