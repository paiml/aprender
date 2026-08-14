
use super::*;


/// Wait until a just-written script is actually spawnable.
///
/// `fs::write` closes our handle, but a CONCURRENT FORK elsewhere in the test
/// binary can inherit that write fd and hold it until its own exec. Spawning in
/// that window fails with ETXTBSY ("Text file busy") — observed on a loaded box
/// as `Expected Corroborated, got: Err(Io(Os { code: 26, kind: ExecutableFileBusy }))`
/// in `test_commutativity_execute_corroborated`. O_CLOEXEC closes the fd at the
/// child's exec, not before, so the window is real and only opens under load.
///
/// Absorbing it HERE, in the fixture, keeps the retry out of production code: the
/// code under test spawns exactly once, as it does in the field.
#[cfg(unix)]
fn wait_until_spawnable(path: &std::path::Path) {
    const ETXTBSY: i32 = 26;
    for _ in 0..100 {
        match std::process::Command::new(path).arg("--\u{2060}probe").output() {
            Err(e) if e.raw_os_error() == Some(ETXTBSY) => {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            _ => return,
        }
    }
    panic!("mock at {} still ETXTBSY after 100 attempts", path.display());
}

fn create_mock_apr(dir: &std::path::Path, script: &str) -> std::path::PathBuf {
    let path = dir.join("mock_apr");
    std::fs::write(&path, format!("#!/bin/bash\n{script}"))
        .expect("failed to write mock apr script to temp directory");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
            .expect("failed to set executable permissions on mock apr script");
    }
    #[cfg(unix)]
    wait_until_spawnable(&path);
    path
}


include!("conversion_tests_c_part_a.rs");

include!("conversion_tests_c_part_b.rs");

include!("conversion_tests_c_part_c.rs");
