// F-FORMAT-DISPATCH-003: apr tensors supports all 3 formats
#[test]
fn test_f_format_dispatch_003_tensors_all_formats() {
    for (name, file) in [
        ("GGUF", create_test_gguf_file()),
        ("APR", create_test_apr_file()),
        ("SafeTensors", create_test_safetensors_file()),
    ] {
        let output = apr()
            .args(["tensors", file.path().to_str().unwrap()])
            .output()
            .expect(&format!("run tensors on {name}"));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !stderr.contains("not supported") && !stderr.contains("Only GGUF"),
            "{name} tensors should not skip: {stderr}"
        );
    }
}

// F-FORMAT-DISPATCH-004: apr lint supports all 3 formats
#[test]
fn test_f_format_dispatch_004_lint_all_formats() {
    for (name, file) in [
        ("GGUF", create_test_gguf_file()),
        ("APR", create_test_apr_file()),
        ("SafeTensors", create_test_safetensors_file()),
    ] {
        let output = apr()
            .args(["lint", file.path().to_str().unwrap()])
            .output()
            .expect(&format!("run lint on {name}"));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !stderr.contains("not supported") && !stderr.contains("Only GGUF"),
            "{name} lint should not skip: {stderr}"
        );
    }
}

// F-FORMAT-DISPATCH-005: apr diff supports all format combinations
#[test]
fn test_f_format_dispatch_005_diff_all_format_combinations() {
    let gguf = create_test_gguf_file();
    let apr_file = create_test_apr_file();
    let st = create_test_safetensors_file();

    // Test all 6 combinations (excluding same-file comparisons)
    let combinations = [
        ("GGUF-APR", gguf.path(), apr_file.path()),
        ("GGUF-ST", gguf.path(), st.path()),
        ("APR-GGUF", apr_file.path(), gguf.path()),
        ("APR-ST", apr_file.path(), st.path()),
        ("ST-GGUF", st.path(), gguf.path()),
        ("ST-APR", st.path(), apr_file.path()),
    ];

    for (name, path1, path2) in combinations {
        let output = apr()
            .args(["diff", path1.to_str().unwrap(), path2.to_str().unwrap()])
            .output()
            .expect(&format!("run diff {name}"));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !stderr.contains("not supported") && !stderr.contains("Only GGUF"),
            "{name} diff should not skip: {stderr}"
        );
    }
}
