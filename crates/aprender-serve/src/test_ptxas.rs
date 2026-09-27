//! Test-only: assemble emitted PTX with `ptxas`, portably across CUDA toolchains.
//!
//! Our kernels declare `.target sm_70`. Newer CUDA releases drop old archs from
//! ptxas (`Value 'sm_70' is not defined for option 'gpu-name'`, seen on yoga in CI
//! run 35847926651). PTX for sm_70 assembles for any later arch, so an arch that
//! ptxas does not DEFINE is skipped for the next newer one. Any other ptxas error
//! is returned, which keeps the guards able to fail (#3975).

/// Archs tried, in order, after the caller's `first` candidates.
const NEWER: [&str; 6] = ["sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_120"];

/// The `.target` a PTX module declares (default `sm_70`).
pub(crate) fn declared_target(ptx: &str) -> String {
    ptx.lines()
        .find_map(|l| l.trim().strip_prefix(".target "))
        .unwrap_or("sm_70")
        .trim()
        .to_string()
}

/// Assemble `ptx` (named `tag` in temp files) trying `first`, then [`NEWER`],
/// skipping ONLY archs this ptxas does not define. `Ok(arch)` names the arch
/// used; `Err` carries the first real ptxas error, or says no arch was defined.
pub(crate) fn assemble(ptx: &str, tag: &str, first: &[&str]) -> Result<String, String> {
    let dir = std::env::temp_dir().join(format!(
        "apr_ptxas_{}_{:?}_{tag}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).map_err(|e| format!("temp dir {}: {e}", dir.display()))?;
    let src = dir.join(format!("{tag}.ptx"));
    // A module that could not be written was not assembled: that is an error, not a pass.
    std::fs::write(&src, ptx)
        .map_err(|e| format!("could not write PTX to {} ({e})", src.display()))?;
    let mut undefined = Vec::new();
    let mut result = Err(String::new());
    for arch in first.iter().chain(NEWER.iter()) {
        // A cuda-feature build implies a CUDA toolchain: a missing ptxas is a finding.
        let out = match std::process::Command::new("ptxas")
            .args(["--gpu-name", arch, "-o"])
            .arg(dir.join(format!("{tag}.cubin")))
            .arg(&src)
            .output()
        {
            Ok(o) => o,
            Err(e) => {
                result = Err(format!("could not run ptxas ({e})"));
                break;
            },
        };
        if out.status.success() {
            result = Ok((*arch).to_string());
            break;
        }
        let err = String::from_utf8_lossy(&out.stderr).into_owned();
        if err.contains("is not defined for option 'gpu-name'") {
            undefined.push(*arch);
            continue;
        }
        result = Err(format!(
            "at {arch}: {}",
            err.trim().lines().take(3).collect::<Vec<_>>().join(" | ")
        ));
        break;
    }
    let _ = std::fs::remove_dir_all(&dir);
    result.map_err(|e| {
        if e.is_empty() {
            format!("this ptxas defines none of {undefined:?}")
        } else {
            e
        }
    })
}
