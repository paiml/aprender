//! `verificar generate --language` must actually select the grammar.
//!
//! The binary maps an unrecognised `--language` onto `Language::Python` with
//! only a warning (see `parse_language` in `src/bin/verificar.rs`), so a
//! silently-ignored flag would look identical to a working one on the default
//! invocation. This test pins the observable difference: bash assignments have
//! no spaces around `=` (`x=1`) while Python's do (`x = 1`). Asserting only
//! that each run produced output would not exclude "every language emits
//! Python".

use std::process::Command;

/// Run `verificar generate` for `language` with a fixed seed and depth.
fn generate(language: &str) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_verificar"))
        .args([
            "generate",
            "--language",
            language,
            "--count",
            "3",
            "--max-depth",
            "2",
            "--seed",
            "7",
        ])
        .output()
        .expect("run verificar");

    assert!(
        out.status.success(),
        "verificar generate --language {language} exited nonzero: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf-8 stdout")
}

#[test]
fn bash_and_python_generators_emit_different_syntax() {
    let bash = generate("bash");
    let python = generate("python");

    assert_ne!(
        bash, python,
        "--language produced byte-identical output for bash and python; the \
         flag is not reaching the generator"
    );

    // Bash forbids spaces around `=` in an assignment; Python requires them by
    // convention and this generator emits them. Each check excludes the other
    // language's output shape.
    assert!(
        bash.contains("x=") && !bash.contains("x = "),
        "expected unspaced bash assignments, got:\n{bash}"
    );
    assert!(
        python.contains("x = "),
        "expected spaced python assignments, got:\n{python}"
    );
}

#[test]
fn generation_is_deterministic_for_a_fixed_seed() {
    assert_eq!(
        generate("python"),
        generate("python"),
        "two runs at --seed 7 diverged; generation is not reproducible"
    );
}
