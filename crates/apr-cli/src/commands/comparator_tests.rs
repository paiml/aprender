//! EXT-26 (aprender#4408): FALSIFY-EXT-020 and comparator-block determinism.

use super::*;
use serde_json::json;
use tempfile::TempDir;

const H: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn full() -> Value {
    json!({
        "command": ["llama-quantize", "in.gguf", "out.gguf", "Q4_K_M"],
        "version": "b6400",
        "env_sha256": H,
        "artifact_sha256": H,
        "log_path": "/tmp/arm.log",
        "started_utc": "2026-09-25T00:00:00.000Z",
        "finished_utc": "2026-09-25T00:00:01.000Z"
    })
}

fn spec(dir: &Path, name: &str, script: &str) -> ArmSpec {
    ArmSpec {
        name: name.into(),
        command: vec!["sh".into(), "-c".into(), script.into()],
        version_command: vec!["sh".into(), "-c".into(), "echo; echo tool 1.2.3".into()],
        env: vec![
            ("PATH".into(), "/usr/bin:/bin".into()),
            ("SEED".into(), "7".into()),
        ],
        artifact: dir.join("out.bin"),
        image: None,
        workdir: dir.to_path_buf(),
    }
}

/// FALSIFY-EXT-020: an arm missing any of the five fields blocks.
#[test]
fn falsify_ext_020_incomplete_arm_blocks() {
    assert!(check_block(&full()).is_ok());
    for key in REQUIRED {
        let mut v = full();
        v.as_object_mut().expect("object").remove(key);
        let e = check_block(&v).expect_err(key);
        assert!(e.contains(key) && e.contains("FALSIFY-EXT-020"), "{e}");

        let mut blank = full();
        blank[key] = if key == "command" {
            json!([])
        } else {
            json!(" ")
        };
        assert!(check_block(&blank).expect_err(key).contains(key));
    }
    let mut short = full();
    short["artifact_sha256"] = json!("abc");
    assert!(check_block(&short).is_err());

    let ok = json!({"arm": "llama.cpp", "comparator": full()});
    assert_eq!(check_arms(&[ok.clone(), ok.clone()]), Ok(2));
    let bare = json!({"arm": "ollama"});
    let e = check_arms(&[ok, bare]).expect_err("no block");
    assert!(e.contains("ollama") && e.contains("FALSIFY-EXT-020"), "{e}");
}

/// Two identical runs give identical comparator blocks except timestamps;
/// the host env never reaches the child.
#[test]
fn two_identical_runs_give_identical_blocks_except_timestamps() {
    let dir = TempDir::new().expect("tempdir");
    let s = spec(dir.path(), "arm-a", "env | sort > out.bin; echo ran");
    let logs = dir.path().join("logs");
    std::fs::create_dir_all(&logs).expect("mkdir");

    let strip = |mut b: ComparatorBlock| {
        b.started_utc.clear();
        b.finished_utc.clear();
        b
    };
    let first = run_arm(&s, &logs).expect("run 1");
    let second = run_arm(&s, &logs).expect("run 2");
    for t in [&first.started_utc, &first.finished_utc] {
        let ts = chrono::DateTime::parse_from_rfc3339(t).expect("rfc3339");
        assert!(
            t.ends_with('Z') && ts.offset().local_minus_utc() == 0,
            "{t}"
        );
    }
    assert!(first.started_utc <= first.finished_utc);
    assert_eq!(strip(first.clone()), strip(second));
    assert_eq!(first.version, "tool 1.2.3");
    assert_eq!(first.command, s.command);
    assert_eq!(first.env_sha256, env_sha256(&s.env));
    assert!(first.log_path.ends_with("logs/arm-a.log"));
    let child_env = std::fs::read_to_string(dir.path().join("out.bin")).expect("env");
    assert!(child_env.contains("SEED=7"), "{child_env}");
    assert!(!child_env.contains("HOME="), "host env leaked: {child_env}");

    let mut other = s.clone();
    other.env[1].1 = "8".into();
    let changed = run_arm(&other, &logs).expect("run 3");
    assert_ne!(changed.env_sha256, first.env_sha256);
    assert_ne!(changed.artifact_sha256, first.artifact_sha256);
}

/// The env digest does not depend on declaration order.
#[test]
fn env_digest_is_order_free() {
    let a = vec![("A".to_string(), "1".to_string()), ("B".into(), "2".into())];
    let b = vec![a[1].clone(), a[0].clone()];
    assert_eq!(env_sha256(&a), env_sha256(&b));
    assert_ne!(env_sha256(&a), env_sha256(&a[..1]));
}

/// A failed run, a missing artifact, an empty version or a bad name is an
/// error, never a partial block.
#[test]
fn a_failed_arm_gives_no_block() {
    let dir = TempDir::new().expect("tempdir");
    let d = dir.path();
    assert!(run_arm(&spec(d, "fails", "echo x > out.bin; exit 3"), d)
        .expect_err("exit 3")
        .contains("exited"));
    // `fails` left an out.bin behind: it must not be taken as this run's.
    assert!(d.join("out.bin").exists());
    assert!(run_arm(&spec(d, "no-artifact", "true"), d)
        .expect_err("no artifact")
        .contains("artifact"));
    let mut quiet = spec(d, "quiet", "echo x > out.bin");
    quiet.version_command = vec!["true".into()];
    assert!(run_arm(&quiet, d)
        .expect_err("no version")
        .contains("version"));
    let mut lies = spec(d, "lies", "echo x > out.bin");
    lies.version_command = vec!["sh".into(), "-c".into(), "echo tool 1.2.3; exit 2".into()];
    assert!(run_arm(&lies, d)
        .expect_err("a failed version command")
        .contains("gave no version"));
    for bad in ["../x", "", "Arm", "a b"] {
        let e = run_arm(&spec(d, bad, "echo x > out.bin"), d).expect_err(bad);
        assert!(e.contains("is not [a-z0-9._-]+"), "{bad}: {e}");
    }
    for good in ["llama.cpp", "arm_2-b"] {
        assert!(
            run_arm(&spec(d, good, "echo x > out.bin"), d).is_ok(),
            "{good}"
        );
    }
    // An artifact path that cannot be removed (a directory) is refused, not ignored.
    let mut stuck = spec(d, "stuck", "true");
    stuck.artifact = d.join("stuck-dir");
    std::fs::create_dir(&stuck.artifact).expect("mkdir");
    assert!(run_arm(&stuck, d)
        .expect_err("a directory")
        .contains("stale artifact"));
}

/// R-1a: containers are digest-pinned and run with the network denied.
#[test]
fn a_container_arm_is_digest_pinned_and_offline() {
    let argv = vec!["llama-cli".to_string(), "--version".to_string()];
    let env = vec![("SEED".to_string(), "7".to_string())];
    let wd = Path::new("/w");
    assert!(container_argv("ghcr.io/ggml-org/llama.cpp:b6400", &argv, &env, wd).is_err());
    assert!(container_argv("img@sha256:abc", &argv, &env, wd).is_err());
    let pinned = format!("ghcr.io/ggml-org/llama.cpp@sha256:{H}");
    let out = container_argv(&pinned, &argv, &env, wd).expect("pinned");
    let image_at = out.iter().position(|a| *a == pinned).expect("image");
    assert!(out[..image_at].contains(&"--network=none".to_string()));
    assert!(out[..image_at].contains(&"SEED=7".to_string()));
    assert_eq!(out[image_at + 1..], argv[..]);

    let mut v = full();
    v["image"] = json!("llama.cpp:latest");
    assert!(check_block(&v).is_err());
    v["image"] = json!(pinned);
    assert!(check_block(&v).is_ok());
}
