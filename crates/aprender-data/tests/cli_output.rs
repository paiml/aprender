//! What the `aprender-data` binary PRINTS, for the commands whose whole effect is their output.
//!
//! `quality profiles` and the `hub push` warning return `Ok(())` / `()` whatever they print, so a lib test that
//! asserts only the return value passes against an empty body (mutants `quality.rs:381` → `Ok(())` and
//! `hub.rs:145` → `()`, #4588). These run the binary built from this tree and read its streams.

use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_aprender-data");

fn run(args: &[&str]) -> Output {
    Command::new(BIN)
        .args(args)
        .output()
        .expect("spawn the binary under test")
}

/// Every profile the library knows is listed by name, between the header and the usage line.
#[test]
fn quality_profiles_lists_every_profile() {
    let out = run(&["quality", "profiles"]);
    assert!(out.status.success(), "{out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Available Quality Profiles"), "{stdout}");
    let names = alimentar::quality::QualityProfile::available_profiles();
    assert!(
        !names.is_empty(),
        "a profile list that is empty proves nothing"
    );
    let (mut with_constants, mut without_constants) = (0, 0);
    for name in names {
        let head = format!("  {name} - ");
        let at = stdout
            .find(&head)
            .unwrap_or_else(|| panic!("{name} missing:\n{stdout}"));
        // A profile's block runs from its header to the blank line after it.
        let block = stdout[at..].split("\n\n").next().unwrap_or("");
        let profile =
            alimentar::quality::QualityProfile::by_name(name).expect("a listed profile loads");
        let has = |label: &str| block.contains(label);
        assert_eq!(
            has("    Expected constants: "),
            !profile.expected_constant_columns.is_empty(),
            "{name}:\n{block}"
        );
        assert_eq!(
            has("    Nullable columns: "),
            !profile.nullable_columns.is_empty(),
            "{name}:\n{block}"
        );
        if profile.expected_constant_columns.is_empty() {
            without_constants += 1;
        } else {
            with_constants += 1;
        }
    }
    // Both sides of the `is_empty` branch must be exercised, or the check above proves half of it.
    assert!(
        with_constants > 0 && without_constants > 0,
        "{with_constants}/{without_constants}"
    );
    assert!(
        stdout.contains("Usage: aprender-data quality score <path> --profile <name>"),
        "{stdout}"
    );
}

/// The publishing warning is printed BEFORE the input is checked, so a push of a file that does not exist
/// still shows it — and fails without touching the network.
#[cfg(feature = "hf-hub")]
#[test]
fn hub_push_warns_about_quality_before_anything_else() {
    let out = run(&["hub", "push", "/nonexistent/a07-4588.parquet", "owner/repo"]);
    assert!(!out.status.success(), "a missing input must fail: {out:?}");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("WARNING: Data quality is CRITICAL for ML datasets!"),
        "{stderr}"
    );
    assert!(
        stderr.contains("Minimum recommended: Grade B (85%)"),
        "{stderr}"
    );
}
