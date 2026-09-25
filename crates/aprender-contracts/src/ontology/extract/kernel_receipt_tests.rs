//! OXIDE-001 O-2 (aprender#3522) — the extractor's case table. Each case is a tree one edit away from the
//! committed `evidence/kernels/gdn_gated_rmsnorm/`, and says which edit.

use super::*;

const SAFE_SRC: &str = "
mod kernels {
    fn helper(x: &[f32]) -> f32 { x[0] }
    #[kernel(launch_context = c)]
    pub fn k_a(x: &[f32]) -> f32 { helper(x) }
    #[kernel]
    pub fn k_b(x: &[f32]) -> f32 { x[1] }
}
fn cpu_ref(x: &[f32]) -> f32 { x[0] }
";

fn manifest() -> serde_json::Value {
    serde_json::json!({
        "schema": MANIFEST_SCHEMA,
        "kernel": "kx",
        "authoring": "oxide",
        "source": "src/k.rs",
        "module": "kernels",
        "entries": ["k_a", "k_b"],
        "reference": "cpu_ref",
        "required_hosts": ["h1", "h2"]
    })
}

fn row(host: &str, entry: &str, parity: bool) -> serde_json::Value {
    serde_json::json!({
        "schema": RECEIPT_SCHEMA, "kernel": "kx", "variant": entry, "entry": entry, "authoring": "oxide",
        "parity": {"pass": parity}, "timing": {"pass": true},
        "register_budget": {"oxide": 30, "handptx": 27},
        "host": host, "sha": "abc", "tree_dirty_paths": 0, "foreign_gpu_procs": ""
    })
}

fn receipt(_host: &str, rows: &[serde_json::Value]) -> serde_json::Value {
    serde_json::json!({"schema": RECEIPT_SCHEMA, "kernel": "kx", "receipts": rows})
}

/// A tree: `src/k.rs` holds `src`, and `evidence/kernels/kx/` holds each `(file, json)`.
fn tree(src: &str, files: &[(&str, serde_json::Value)]) -> tempfile::TempDir {
    let d = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(d.path().join("src")).expect("mkdir src");
    std::fs::write(d.path().join("src/k.rs"), src).expect("write src");
    let kd = d.path().join(EVIDENCE_DIR).join("kx");
    std::fs::create_dir_all(&kd).expect("mkdir kx");
    for (name, v) in files {
        std::fs::write(kd.join(name), v.to_string()).expect("write json");
    }
    d
}

fn green_files() -> Vec<(&'static str, serde_json::Value)> {
    vec![
        (MANIFEST_FILE, manifest()),
        (
            "h1.json",
            receipt("h1", &[row("h1", "k_a", true), row("h1", "k_b", true)]),
        ),
        (
            "h2.json",
            receipt("h2", &[row("h2", "k_a", true), row("h2", "k_b", true)]),
        ),
    ]
}

fn run(d: &tempfile::TempDir) -> (Graph, KernelStats) {
    let mut g = Graph::new();
    let stats = extract(d.path(), &mut g);
    (g, stats)
}

fn lits(g: &Graph, prop: &str) -> Vec<String> {
    g.objects(&iri_path("kernel", &["kx"]), &kernel(prop))
        .iter()
        .filter_map(|t| t.as_literal().map(|l| l.0.to_string()))
        .collect()
}

#[test]
fn the_green_tree_is_one_oxide_kernel_with_four_receipts_and_nothing_to_refuse() {
    let (g, stats) = run(&tree(SAFE_SRC, &green_files()));
    assert_eq!(
        (stats.kernels, stats.receipts),
        (1, 4),
        "{:?}",
        stats.errors
    );
    assert!(stats.errors.is_empty());
    let node = iri_path("kernel", &["kx"]);
    assert_eq!(
        g.objects(&node, RDF_TYPE)[0].as_iri(),
        Some(kernel("OxideKernel").as_str())
    );
    assert_eq!(g.objects(&node, &kernel("receipt")).len(), 4);
    for prop in [
        "missingReceipt",
        "sourceMissing",
        "missingEntry",
        "missingReference",
        "unsafeSite",
        "rawPointer",
    ] {
        assert!(lits(&g, prop).is_empty(), "{prop}: {:?}", lits(&g, prop));
    }
    assert_eq!(lits(&g, "reference"), vec!["cpu_ref".to_string()]);
}

#[test]
fn a_missing_host_file_and_a_missing_row_are_both_named() {
    let mut files = green_files();
    files.remove(2); // h2.json gone
    files[1].1 = receipt("h1", &[row("h1", "k_a", true)]); // h1 has no k_b row
    let (g, _) = run(&tree(SAFE_SRC, &files));
    assert_eq!(
        lits(&g, "missingReceipt"),
        vec!["h1:k_b".to_string(), "h2".to_string()]
    );
}

#[test]
fn an_unsafe_block_in_a_helper_is_a_site_not_only_one_in_an_entry() {
    let src = SAFE_SRC.replace(
        "{ x[0] }\n    #[kernel",
        "{ unsafe { *x.as_ptr() } }\n    #[kernel",
    );
    assert_ne!(src, SAFE_SRC, "the plant must land");
    let (g, _) = run(&tree(&src, &green_files()));
    assert_eq!(
        lits(&g, "unsafeSite"),
        vec!["unsafe block in fn helper".to_string()]
    );
    assert_eq!(
        lits(&g, "rawPointer"),
        Vec::<String>::new(),
        "as_ptr() is a call, not a pointer type"
    );
}

#[test]
fn a_raw_pointer_parameter_and_an_unsafe_fn_are_sites() {
    let src = SAFE_SRC.replace("pub fn k_b(x: &[f32])", "pub unsafe fn k_b(x: *const f32)");
    let (g, _) = run(&tree(&src, &green_files()));
    assert_eq!(
        lits(&g, "unsafeSite"),
        vec!["unsafe fn in fn k_b".to_string()]
    );
    assert_eq!(
        lits(&g, "rawPointer"),
        vec!["raw pointer in fn k_b".to_string()]
    );
}

#[test]
fn unsafe_outside_the_device_module_is_the_hosts_business_not_the_kernels() {
    let src = format!("{SAFE_SRC}\nfn host() {{ unsafe {{ launch() }} }}");
    let (g, _) = run(&tree(&src, &green_files()));
    assert!(lits(&g, "unsafeSite").is_empty());
}

#[test]
fn an_entry_without_the_kernel_attribute_is_missing_and_so_is_a_renamed_reference() {
    let src = SAFE_SRC
        .replace("#[kernel]\n", "")
        .replace("fn cpu_ref", "fn cpu_ref2");
    let (g, _) = run(&tree(&src, &green_files()));
    assert_eq!(lits(&g, "missingEntry"), vec!["k_b".to_string()]);
    assert_eq!(lits(&g, "missingReference"), vec!["cpu_ref".to_string()]);
}

#[test]
fn a_missing_source_or_module_cannot_read_as_a_safe_one() {
    let mut files = green_files();
    files[0].1["source"] = "src/gone.rs".into();
    let (g, _) = run(&tree(SAFE_SRC, &files));
    assert_eq!(lits(&g, "sourceMissing"), vec!["src/gone.rs".to_string()]);

    let mut files = green_files();
    files[0].1["module"] = "device".into();
    let (g, _) = run(&tree(SAFE_SRC, &files));
    assert_eq!(
        lits(&g, "missingEntry"),
        vec!["module \"device\"".to_string()]
    );
}

#[test]
fn a_ptx_kernel_types_as_ptx_and_its_exemption_must_resolve() {
    let mut files = green_files();
    files[0].1["authoring"] = "ptx".into();
    files[0].1["ptx_exemption"] = serde_json::json!({"receipt": "evidence/kernels/kx/nope.md"});
    let (g, _) = run(&tree(SAFE_SRC, &files));
    let node = iri_path("kernel", &["kx"]);
    assert_eq!(
        g.objects(&node, RDF_TYPE)[0].as_iri(),
        Some(kernel("PtxKernel").as_str())
    );
    assert_eq!(
        lits(&g, "exemptionReceiptMissing"),
        vec!["evidence/kernels/kx/nope.md".to_string()]
    );
}

#[test]
fn a_foreign_process_is_carried_and_an_empty_one_is_not() {
    let mut files = green_files();
    files[1].1["receipts"][0]["foreign_gpu_procs"] = "123, apr, 900 MiB;".into();
    let (g, _) = run(&tree(SAFE_SRC, &files));
    let r = iri_path("kernel-receipt", &["kx", "h1", "k_a"]);
    assert_eq!(g.objects(&r, &kernel("foreignGpuProcs")).len(), 1);
    let r2 = iri_path("kernel-receipt", &["kx", "h1", "k_b"]);
    assert!(g.objects(&r2, &kernel("foreignGpuProcs")).is_empty());
}

#[test]
fn foreign_files_are_refused_by_name_never_skipped() {
    let mut files = green_files();
    files.push(("notes.json", serde_json::json!({"hello": 1})));
    files.push(("kernel2.json", manifest()));
    let (_, stats) = run(&tree(SAFE_SRC, &files));
    let named: Vec<_> = stats
        .errors
        .iter()
        .map(|e| e.file.rsplit('/').next().unwrap_or_default().to_string())
        .collect();
    assert_eq!(
        named,
        vec!["kernel2.json".to_string(), "notes.json".to_string()],
        "{:?}",
        stats.errors
    );
}

#[test]
fn receipts_with_no_manifest_and_a_receipt_for_another_kernel_are_refused() {
    let files = vec![("h1.json", receipt("h1", &[row("h1", "k_a", true)]))];
    let (_, stats) = run(&tree(SAFE_SRC, &files));
    assert_eq!((stats.kernels, stats.errors.len()), (0, 1));

    let mut files = green_files();
    files[1].1["kernel"] = "other".into();
    let (_, stats) = run(&tree(SAFE_SRC, &files));
    assert_eq!(stats.errors.len(), 1, "{:?}", stats.errors);
    assert!(stats.errors[0].what.contains("other"));
}

#[test]
fn the_denominator_pins_the_kernel_count() {
    let d = tree(SAFE_SRC, &green_files());
    let set = |n: &str| std::fs::write(d.path().join(EXPECTED_FILE), n).expect("write expected");
    set("# kernels\n1\n");
    assert_eq!(run(&d).1.wrong_corpus(), None);
    set("2\n");
    assert_eq!(run(&d).1.wrong_corpus(), Some((2, 1)));
    std::fs::remove_file(d.path().join(EXPECTED_FILE)).expect("rm");
    assert_eq!(
        run(&d).1.wrong_corpus(),
        None,
        "an ABSENT denominator is not a miss"
    );
}

#[test]
fn two_extractions_are_identical() {
    let d = tree(SAFE_SRC, &green_files());
    assert_eq!(run(&d).0.to_ntriples(), run(&d).0.to_ntriples());
}

#[test]
fn the_positive_control_fires() {
    assert!(positive_control());
}

#[test]
fn an_unsafe_method_a_trait_default_and_an_extern_block_are_sites() {
    // Sonnet quorum, O-2 round 1: `ImplItemFn` is not `ItemFn`, so an unsafe METHOD read as safe.
    let src = SAFE_SRC.replace(
        "    fn helper",
        "    struct W;\n    impl W { unsafe fn put(&self) {} }\n    trait T { unsafe fn t(&self) {} }\n    extern \"C\" { fn ext(); }\n    fn helper",
    );
    assert_ne!(src, SAFE_SRC, "the plant must land");
    let (g, _) = run(&tree(&src, &green_files()));
    assert_eq!(
        lits(&g, "unsafeSite"),
        vec![
            "extern block (module level)".to_string(),
            "unsafe fn in fn put".to_string(),
            "unsafe fn in fn t".to_string(),
        ]
    );
}

#[test]
fn unsafe_and_raw_pointers_hidden_in_a_macro_are_sites_and_a_plain_macro_is_not() {
    let plain = SAFE_SRC.replace(
        "{ x[0] }\n    #[kernel",
        "{ assert!(x.len() > 0); x[0] }\n    #[kernel",
    );
    assert_ne!(plain, SAFE_SRC, "the plant must land");
    let (g, _) = run(&tree(&plain, &green_files()));
    assert!(
        lits(&g, "unsafeSite").is_empty(),
        "{:?}",
        lits(&g, "unsafeSite")
    );

    let hidden = SAFE_SRC.replace(
        "{ x[0] }\n    #[kernel",
        "{ m!(unsafe { *(x.as_ptr() as *const f32) }) }\n    #[kernel",
    );
    let (g, _) = run(&tree(&hidden, &green_files()));
    assert_eq!(
        lits(&g, "unsafeSite"),
        vec!["unsafe in macro in fn helper".to_string()]
    );
    assert_eq!(
        lits(&g, "rawPointer"),
        vec!["raw pointer in macro in fn helper".to_string()]
    );
}
