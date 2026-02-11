fn main() {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output();
    let sha = match output {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        _ => "unknown".to_string(),
    };
    println!("cargo:rustc-env=APR_GIT_SHA={sha}");
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/refs/heads/");
}
