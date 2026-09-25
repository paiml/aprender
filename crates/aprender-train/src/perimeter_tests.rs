//! FALSIFY-EXT-011 (EXT-04, aprender#4386): a training process cannot reach
//! a publish token nor write the sealed manifest or the promotion contract.

use super::*;
use tempfile::TempDir;

struct Planted {
    _dir: TempDir,
    creds: PathBuf,
    sealed: PathBuf,
    work: PathBuf,
}

fn plant() -> Planted {
    let dir = TempDir::new().expect("tempdir");
    let (creds, sealed, work) =
        (dir.path().join("publish"), dir.path().join("sealed"), dir.path().join("work"));
    for d in [&creds, &sealed, &work] {
        std::fs::create_dir_all(d).expect("mkdir");
    }
    Planted { _dir: dir, creds, sealed, work }
}

fn env(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs.iter().map(|(k, v)| ((*k).to_string(), (*v).to_string())).collect()
}

#[test]
fn falsify_ext_011_train_cannot_reach_token_or_manifest() {
    let p = plant();
    let perimeter = Perimeter::at(&p.creds, &p.sealed);
    let ok_out = p.work.join("model.apr");
    let benign = env(&[("HF_TOKEN", "hf_pull_only"), ("PATH", "/usr/bin")]);

    // Clean: an empty credentials dir, no publish env, outputs outside sealed.
    // HF_TOKEN is the pull token `apr pull` uses and is not a publish credential.
    assert_eq!(perimeter.violations(benign.clone(), [ok_out.as_path()]), vec![]);

    // (1) A readable publish token.
    let token = p.creds.join("hf.token");
    std::fs::write(&token, "planted").expect("plant token");
    assert_eq!(
        perimeter.violations(benign.clone(), [ok_out.as_path()]),
        vec![PerimeterViolation::CredentialReadable(token.clone())]
    );
    std::fs::remove_file(&token).expect("unplant");

    // (2) A publish token in the environment; an empty value is not a token.
    let leaked = env(&[("APR_PUBLISH_HF_TOKEN", "planted"), ("APR_PUBLISH_GHCR_TOKEN", "")]);
    assert_eq!(
        perimeter.violations(leaked, [ok_out.as_path()]),
        vec![PerimeterViolation::CredentialInEnv("APR_PUBLISH_HF_TOKEN".into())]
    );

    // (3) Writes into the sealed store: directly, through `..`, and through a
    // symlink, before the target file exists.
    let sealed_abs = p.sealed.canonicalize().expect("canon");
    let direct = p.sealed.join("promotion-contract.yaml");
    let dotdot = p.work.join("../sealed/manifest.json");
    let link = p.work.join("out");
    std::os::unix::fs::symlink(&p.sealed, &link).expect("symlink");
    let via_link = link.join("nested/manifest.json");
    for out in [&direct, &dotdot, &via_link] {
        let bad = perimeter.violations(benign.clone(), [out.as_path()]);
        assert!(
            matches!(bad.as_slice(), [PerimeterViolation::WritesSealed(_, s)] if *s == sealed_abs),
            "{} escaped the perimeter: {bad:?}",
            out.display()
        );
    }
    // A sibling whose name only starts with "sealed" is outside.
    let sibling = p.work.join("../sealed-not/x");
    assert_eq!(perimeter.violations(benign, [sibling.as_path()]), vec![]);
}

#[test]
fn an_unreadable_token_is_the_intended_state() {
    let p = plant();
    let token = p.creds.join("hf.token");
    std::fs::write(&token, "planted").expect("plant");
    // Owned by the publisher account on the driver host: this process cannot open it.
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&token, std::fs::Permissions::from_mode(0o000)).expect("chmod");
    let readable = std::fs::File::open(&token).is_ok(); // true when the test runs as root
    let bad = Perimeter::at(&p.creds, &p.sealed).violations(Vec::new(), []);
    assert_eq!(bad.is_empty(), !readable, "{bad:?}");
    std::fs::set_permissions(&token, std::fs::Permissions::from_mode(0o600)).expect("chmod back");
}

#[test]
fn the_fixed_perimeter_cannot_be_moved_from_the_environment() {
    let fixed = Perimeter::fixed();
    assert_eq!(fixed.credentials_dir, PathBuf::from("/etc/apr/publish"));
    assert_eq!(fixed.sealed_dir, sealed_dir());
    assert!(sealed_dir().is_some_and(|s| s.ends_with(".pacha/sealed")));
}
