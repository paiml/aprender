use std::path::Path;

use provable_contracts::error::Severity;
use provable_contracts::schema::{validate_artifact, ArtifactKind};

/// `pv validate <path>` — validate whatever artifact `path` holds.
///
/// Dispatches on what the file IS, not on the assumption that everything under
/// `contracts/` is a `Contract`. Five files in the corpus are not: two pv
/// binding registries and three publish manifests, all of which failed here
/// with ``missing field `metadata` `` while the directory walkers that lint
/// them already knew to treat them differently. See
/// `provable_contracts::schema::artifact`.
pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let (kind, violations) = validate_artifact(path)?;

    let errors: Vec<_> = violations
        .iter()
        .filter(|v| v.severity == Severity::Error)
        .collect();
    let warnings: Vec<_> = violations
        .iter()
        .filter(|v| v.severity == Severity::Warning)
        .collect();

    for v in &violations {
        println!("{v}");
    }

    println!("\n{} error(s), {} warning(s)", errors.len(), warnings.len());

    if errors.is_empty() {
        println!("{} is valid.", noun(kind));
        Ok(())
    } else {
        Err(format!("{} has {} validation error(s)", noun(kind), errors.len()).into())
    }
}

/// How to name the artifact in the verdict line. Naming the kind is the point:
/// a reader who runs `pv validate contracts/binding.yaml` and is told
/// "Contract is valid." has been told something false about which rules ran.
fn noun(kind: ArtifactKind) -> &'static str {
    match kind {
        ArtifactKind::Contract => "Contract",
        ArtifactKind::Binding => "Binding registry (kind: binding)",
        ArtifactKind::PublishManifest => "Publish manifest (kind: publish-manifest)",
    }
}
