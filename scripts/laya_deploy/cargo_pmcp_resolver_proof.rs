
// ---------------------------------------------------------------------------
// APPENDED by aprender's `just laya-resolver-proof` to a SCRATCH copy of
// cargo-pmcp/src/deployment/builder.rs (plan 08-10, Pitfall 1). It is never
// written into the SDK checkout: the recipe `git archive`s a recorded commit
// into a scratch directory and appends this module there.
//
// It EXECUTES cargo-pmcp's own `find_lambda_package_dir` on the aprender
// workspace named by APRENDER_WORKSPACE, with the deploy root and server name
// `just laya-deploy` uses (shared-crates-root: root `crates`, server
// `aprender-mcp-decide`), and prints the directory it returns. Building the
// decide package by name with zigbuild never calls this resolver, which is why
// that is not accepted as resolver evidence.
//
// A child module of `builder`, so the private fields and the private resolver
// are in scope without changing a line of the SDK's own code.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod aprender_resolver_proof {
    use super::BinaryBuilder;
    use std::path::{Path, PathBuf};

    const SERVER: &str = "aprender-mcp-decide";
    const EXPECTED: &str = "crates/aprender-mcp-decide-lambda";

    /// The builder exactly as `cargo pmcp deploy --manifest-path <root>` constructs it for a
    /// pmcp-run target (`BinaryBuilder::new` only adds the two OAuth flags, both false for
    /// pmcp-run with `[auth] enabled = false`, and neither is read by the resolver).
    fn builder_at(root: PathBuf) -> BinaryBuilder {
        BinaryBuilder {
            project_root: root,
            oauth_enabled: false,
            build_oauth_lambdas: false,
        }
    }

    fn rel(ws: &Path, p: &Path) -> String {
        let p = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
        p.strip_prefix(ws)
            .map_or_else(|_| p.display().to_string(), |r| r.display().to_string())
    }

    #[test]
    fn aprender_decide_resolves_from_shared_crates_root() {
        let ws = std::env::var_os("APRENDER_WORKSPACE")
            .expect("APRENDER_WORKSPACE must name the aprender checkout");
        let ws = PathBuf::from(ws)
            .canonicalize()
            .expect("APRENDER_WORKSPACE must exist");

        // The proof: root `crates`, server `aprender-mcp-decide` -> the preferred-directory
        // branch (`<project_root>/<server>-lambda`), before any workspace-wide search.
        let root = ws.join("crates");
        let got = builder_at(root.clone())
            .find_lambda_package_dir(SERVER)
            .expect("the resolver must return a directory for the decide server");
        let got_rel = rel(&ws, &got);
        println!("RESOLVED root=crates server={SERVER} -> {got_rel}");
        assert_eq!(got_rel, EXPECTED, "shared-crates-root must resolve to the decide package");

        // The control: the per-crate root the template warns about. The preferred directory
        // `<crate>/<server>-lambda` does not exist, so the resolver falls through to the
        // workspace-wide search and returns ANOTHER `*-lambda` package (the trap).
        let per_crate = root.join("aprender-mcp-decide-lambda");
        let trap = builder_at(per_crate)
            .find_lambda_package_dir(SERVER)
            .expect("the workspace search must return some *-lambda package");
        let trap_rel = rel(&ws, &trap);
        println!("CONTROL root={EXPECTED} server={SERVER} -> {trap_rel}");
        assert_ne!(
            trap_rel, EXPECTED,
            "the per-crate root no longer shows the trap; re-derive Pitfall 1 before relying on this proof"
        );
    }
}
