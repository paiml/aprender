/// Test graceful handling of unicode edge cases
#[test]
fn test_cli_021_chaos_unicode() {
    let history = create_temp_history(&["git status", "cargo build"]);
    let model = NamedTempFile::new().unwrap();

    aprender_shell()
        .args([
            "train",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            model.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    // Test Unicode edge cases
    let test_cases = [
        "🚀",                // Emoji
        "日本語",            // CJK characters
        "مرحبا",             // RTL text
        "\u{FEFF}git",       // BOM
        "git\u{200B}status", // Zero-width space
        "git\u{202E}status", // RTL override
        &"é".repeat(100),    // Many combining marks
    ];

    for prefix in test_cases {
        aprender_shell()
            .args(["suggest", "-m", model.path().to_str().unwrap(), prefix])
            .assert()
            .success(); // Should handle gracefully
    }
}

/// Test graceful handling of concurrent file access
#[test]
fn test_cli_021_chaos_concurrent_read() {
    use std::thread;

    let history = create_temp_history(&[
        "git status",
        "git commit",
        "git push",
        "cargo build",
        "cargo test",
    ]);
    let model = NamedTempFile::new().unwrap();

    aprender_shell()
        .args([
            "train",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            model.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    let model_path = model.path().to_str().unwrap().to_string();

    // Spawn multiple concurrent readers
    let handles: Vec<_> = (0..5)
        .map(|_| {
            let model_path = model_path.clone();
            thread::spawn(move || {
                for _ in 0..10 {
                    Command::cargo_bin("aprender-shell")
                        .unwrap()
                        .args(["suggest", "-m", &model_path, "git "])
                        .assert()
                        .success();
                }
            })
        })
        .collect();

    // All threads should complete without issues
    for handle in handles {
        handle.join().expect("Thread should complete successfully");
    }
}

/// Test graceful handling of rapid sequential calls
#[test]
fn test_cli_021_chaos_rapid_calls() {
    let history = create_temp_history(&["git status", "git commit", "cargo build"]);
    let model = NamedTempFile::new().unwrap();

    aprender_shell()
        .args([
            "train",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            model.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    // Rapid sequential calls
    for i in 0..50 {
        aprender_shell()
            .args([
                "suggest",
                "-m",
                model.path().to_str().unwrap(),
                &format!("git {:02}", i),
            ])
            .assert()
            .success();
    }
}
