//! `aprender-present bundle` through the real binary, with a fake `wasm-pack` first on PATH
//! (aprender#6893). The fake records the `--out-dir` it was given and writes the glue there,
//! or writes nothing. So reverting how `bundle()` builds that directory, or deleting its
//! missing-glue refusal, turns a test here red, with no wasm toolchain installed.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("present-bundle-cli-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// A `wasm-pack` that logs its `--out-dir` and, when `FAKE_GLUE=1`, writes the two glue files.
fn fake_wasm_pack(dir: &Path) {
    let script = "#!/bin/sh\n\
        out=\n\
        while [ $# -gt 0 ]; do\n\
          if [ \"$1\" = --out-dir ]; then out=$2; fi\n\
          shift\n\
        done\n\
        printf '%s' \"$out\" > \"$FAKE_LOG\"\n\
        if [ \"$FAKE_GLUE\" = 1 ]; then\n\
          mkdir -p \"$out\"\n\
          echo 'export default 0;' > \"$out/presentar.js\"\n\
          printf '\\0' > \"$out/presentar_bg.wasm\"\n\
        fi\n";
    let path = dir.join("wasm-pack");
    fs::write(&path, script).expect("fake wasm-pack");
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).expect("chmod");
}

fn bundle(cwd: &Path, output: &str, glue: bool) -> (Output, String) {
    let bin = cwd.join("bin");
    fs::create_dir_all(&bin).expect("bin");
    fake_wasm_pack(&bin);
    let log = cwd.join("out-dir.log");
    let path = format!(
        "{}:{}",
        bin.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let out = Command::new(env!("CARGO_BIN_EXE_aprender-present"))
        .args(["bundle", "--no-optimize", "--output", output])
        .current_dir(cwd)
        .env("PATH", path)
        .env("FAKE_LOG", &log)
        .env("FAKE_GLUE", if glue { "1" } else { "0" })
        .output()
        .expect("run bundle");
    let given = fs::read_to_string(&log).unwrap_or_default();
    (out, given)
}

/// FALSIFY-PRESENT_PAGE_NO_SCRIPT-009: a relative `-o` reaches the build tool as an absolute
/// glue directory inside that output, and the bundle is whole.
#[test]
fn relative_output_hands_the_build_tool_an_absolute_glue_dir() {
    let cwd = scratch("relative");
    let (out, given) = bundle(&cwd, "dist", true);
    assert!(out.status.success(), "{out:?}");
    assert_eq!(
        Path::new(&given),
        cwd.join("dist/pkg"),
        "out-dir given to wasm-pack"
    );
    assert!(cwd.join("dist/index.html").is_file());
    assert!(cwd.join("dist/pkg/boot.js").is_file());
    let _ = fs::remove_dir_all(&cwd);
}

/// An absolute `-o` is kept as given, not prefixed (it used to land under `crates/`).
#[test]
fn absolute_output_is_kept_as_given() {
    let cwd = scratch("absolute");
    let target = scratch("absolute-target");
    let (out, given) = bundle(&cwd, target.to_str().expect("utf8"), true);
    assert!(out.status.success(), "{out:?}");
    assert_eq!(Path::new(&given), target.join("pkg"));
    assert!(target.join("pkg/presentar_bg.wasm").is_file());
    let _ = fs::remove_dir_all(&cwd);
    let _ = fs::remove_dir_all(&target);
}

/// FALSIFY-PRESENT_PAGE_NO_SCRIPT-010: a build tool that succeeds but leaves no glue makes
/// `bundle` exit non-zero, name each missing file, and write no page.
#[test]
fn a_bundle_without_glue_is_refused_naming_each_file() {
    let cwd = scratch("noglue");
    let (out, _) = bundle(&cwd, "dist", false);
    assert!(!out.status.success(), "{out:?}");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("pkg/presentar.js is missing"), "{err}");
    assert!(err.contains("pkg/presentar_bg.wasm is missing"), "{err}");
    assert!(!cwd.join("dist/index.html").exists());
    let _ = fs::remove_dir_all(&cwd);
}

/// FALSIFY-PRESENT_PAGE_NO_SCRIPT-004 through the verb: a project page with script is refused,
/// the verb exits non-zero, and no `index.html` is written.
#[test]
fn a_project_page_with_script_makes_the_verb_fail() {
    let cwd = scratch("refused");
    fs::create_dir_all(cwd.join("www")).expect("www");
    fs::write(
        cwd.join("www/index.html"),
        "<html><script>x()</script></html>",
    )
    .expect("page");
    let (out, _) = bundle(&cwd, "dist", true);
    assert!(!out.status.success(), "{out:?}");
    assert!(String::from_utf8_lossy(&out.stderr).contains("refused"));
    assert!(!cwd.join("dist/index.html").exists());
    let _ = fs::remove_dir_all(&cwd);
}

/// A refused page must not leave an index.html from an earlier run beside the fresh glue.
#[test]
fn a_refused_page_removes_a_stale_index_html() {
    let cwd = scratch("stale");
    fs::create_dir_all(cwd.join("www")).expect("www");
    fs::create_dir_all(cwd.join("dist")).expect("dist");
    fs::write(cwd.join("dist/index.html"), "<script>old()</script>").expect("stale");
    fs::write(
        cwd.join("www/index.html"),
        "<html><script>x()</script></html>",
    )
    .expect("page");
    let (out, _) = bundle(&cwd, "dist", true);
    assert_eq!(out.status.code(), Some(1), "{out:?}");
    assert!(!cwd.join("dist/index.html").exists());
    let _ = fs::remove_dir_all(&cwd);
}

/// An unreadable (here non-UTF-8) project page is an error with a message, not a silent fallback.
#[test]
fn an_unreadable_project_page_makes_the_verb_fail() {
    let cwd = scratch("nonutf8");
    fs::create_dir_all(cwd.join("www")).expect("www");
    fs::write(cwd.join("www/index.html"), [0xff, 0xfe, 0x80]).expect("page");
    let (out, _) = bundle(&cwd, "dist", true);
    assert_eq!(out.status.code(), Some(1), "{out:?}");
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot read www/index.html"));
    assert!(!cwd.join("dist/index.html").exists());
    let _ = fs::remove_dir_all(&cwd);
}

const SOURCE_PAGE: &str = "<html><body>my own page</body></html>";

/// Lay out a project whose source page is `www/index.html`, bundle into `output`, and require a
/// refusal that leaves the source page byte-for-byte alone.
fn assert_source_page_survives(name: &str, output: &str, setup: impl Fn(&Path)) {
    let cwd = scratch(name);
    fs::create_dir_all(cwd.join("www")).expect("www");
    fs::write(cwd.join("www/index.html"), SOURCE_PAGE).expect("page");
    setup(&cwd);
    let (out, _) = bundle(&cwd, output, true);
    assert_ne!(out.status.code(), Some(0), "{out:?}");
    assert_eq!(
        fs::read_to_string(cwd.join("www/index.html")).expect("source page must still exist"),
        SOURCE_PAGE
    );
    let _ = fs::remove_dir_all(&cwd);
}

/// `-o www` would make the output page the source page: refuse, delete nothing.
#[test]
fn output_dir_equal_to_www_is_refused_and_keeps_the_source_page() {
    assert_source_page_survives("same-www", "www", |_| {});
}

#[test]
fn output_dir_dot_slash_www_is_refused_and_keeps_the_source_page() {
    assert_source_page_survives("same-dotwww", "./www", |_| {});
}

#[test]
fn output_dir_www_dotdot_www_is_refused_and_keeps_the_source_page() {
    assert_source_page_survives("same-dotdot", "www/../www", |_| {});
}

#[test]
fn output_dir_symlink_to_www_is_refused_and_keeps_the_source_page() {
    assert_source_page_survives("same-symlink", "dist", |cwd| {
        std::os::unix::fs::symlink(cwd.join("www"), cwd.join("dist")).expect("symlink");
    });
}

/// A refused page also removes a stale pkg/boot.js.
#[test]
fn a_refused_page_removes_a_stale_boot_js() {
    let cwd = scratch("stale-boot");
    fs::create_dir_all(cwd.join("www")).expect("www");
    fs::create_dir_all(cwd.join("dist/pkg")).expect("pkg");
    let boot = cwd.join("dist/pkg").join("boot.js");
    fs::write(&boot, "old()").expect("stale boot");
    fs::write(
        cwd.join("www/index.html"),
        "<html><script>x()</script></html>",
    )
    .expect("page");
    let (out, _) = bundle(&cwd, "dist", true);
    assert_eq!(out.status.code(), Some(1), "{out:?}");
    assert!(!boot.exists());
    let _ = fs::remove_dir_all(&cwd);
}

/// A removal that fails for a reason other than NotFound is an error naming the path.
#[test]
fn an_unremovable_stale_index_html_makes_the_verb_fail() {
    let cwd = scratch("unremovable");
    fs::create_dir_all(cwd.join("www")).expect("www");
    fs::create_dir_all(cwd.join("dist/index.html")).expect("index.html as a directory");
    fs::write(cwd.join("dist/index.html/keep"), "x").expect("keep");
    fs::write(cwd.join("www/index.html"), SOURCE_PAGE).expect("page");
    let (out, _) = bundle(&cwd, "dist", true);
    assert_ne!(out.status.code(), Some(0), "{out:?}");
    assert!(String::from_utf8_lossy(&out.stderr).contains("index.html"));
    let _ = fs::remove_dir_all(&cwd);
}
