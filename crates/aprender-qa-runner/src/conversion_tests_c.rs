
use super::*;

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
    path
}


include!("conversion_tests_c_part_a.rs");

include!("conversion_tests_c_part_b.rs");

include!("conversion_tests_c_part_c.rs");
