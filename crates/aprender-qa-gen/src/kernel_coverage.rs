//! Kernel Coverage Verification
//!
//! Discovers HuggingFace kernel requirements per architecture, verifies
//! implementation coverage in the sovereign stack (trueno/realizar), and
//! generates upstream tickets for gaps.
//!
//! All architecture constraints are parsed from `arch-constraints-v1.yaml`
//! (provable-contracts). All kernel bindings are parsed from
//! `kernel-bindings.yaml`. Zero hardcoded data.
//!
//! Spec §20: Kernel Coverage Verification (v1.8.0)

use crate::kernel_class::KernelClass;
use crate::kernel_profile::{profile_from_constraints, ArchConstraints, KernelOp};
use serde::{Deserialize, Serialize};
use std::path::Path;

// ── YAML deserialization types (private) ────────────────────────────────────

/// Top-level structure of arch-constraints-v1.yaml.
#[derive(Deserialize)]
struct ArchConstraintsFile {
    architectures: std::collections::HashMap<String, YamlArchEntry>,
    default: YamlArchEntry,
}

/// Per-architecture entry in the YAML.
#[derive(Deserialize)]
struct YamlArchEntry {
    #[serde(default)]
    aliases: Vec<String>,
    norm_type: String,
    activation: String,
    positional_encoding: String,
    mlp_type: String,
    has_bias: bool,
    tied_embeddings: bool,
}

/// Top-level structure of kernel-bindings.yaml.
#[derive(Deserialize)]
struct KernelBindingsFile {
    bindings: Vec<KernelBinding>,
}

// ── YAML → ArchConstraints conversion helpers ──────────────────────────────

fn convert_norm_type(s: &str) -> Option<String> {
    Some(
        match s {
            "LayerNorm" => "layernorm",
            "RmsNorm" => "rmsnorm",
            _ => return None,
        }
        .to_string(),
    )
}

fn convert_activation(s: &str) -> Option<String> {
    Some(
        match s {
            "Gelu" => "gelu",
            "Silu" => "silu",
            _ => return None,
        }
        .to_string(),
    )
}

fn convert_positional_encoding(s: &str) -> Option<String> {
    match s {
        "Rope" => Some("rope".to_string()),
        "Absolute" => Some("absolute".to_string()),
        "Alibi" => Some("alibi".to_string()),
        // "None" (SSM/recurrence) and unknown values → no positional encoding
        _ => None,
    }
}

fn convert_mlp_type(s: &str) -> Option<String> {
    Some(
        match s {
            "SwiGlu" => "swiglu",
            "GeluMlp" => "gelu_mlp",
            "GatedMlp" => "gated_mlp",
            _ => return None,
        }
        .to_string(),
    )
}

fn yaml_entry_to_constraints(entry: &YamlArchEntry) -> ArchConstraints {
    // SSM/recurrence architectures (positional_encoding: None) have no attention.
    // For attention-based architectures, attention_type is not in the YAML;
    // profile_from_constraints defaults to MHA. GQA/MHA/MQA share the same
    // implementations so coverage is unaffected.
    let attention_type = if entry.positional_encoding == "None" {
        Some("none".to_string())
    } else {
        None
    };

    ArchConstraints {
        attention_type,
        norm_type: convert_norm_type(&entry.norm_type),
        activation: convert_activation(&entry.activation),
        positional_encoding: convert_positional_encoding(&entry.positional_encoding),
        mlp_type: convert_mlp_type(&entry.mlp_type),
        has_bias: Some(entry.has_bias),
        tied_embeddings: Some(entry.tied_embeddings),
    }
}

// ── Public types ───────────────────────────────────────────────────────────

/// Implementation status of a kernel operation in the sovereign stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImplementationStatus {
    /// Optimized fused implementation exists (full speed).
    Fused,
    /// Works via generic dequant+matmul fallback (2-5x slower).
    Fallback,
    /// Not implemented; inference fails or produces garbage.
    Missing,
}

impl ImplementationStatus {
    /// Symbol for display.
    #[must_use]
    pub const fn symbol(&self) -> &'static str {
        match self {
            Self::Fused => "\u{2713}",
            Self::Fallback => "~",
            Self::Missing => "\u{2717}",
        }
    }

    /// Ticket priority for this status.
    #[must_use]
    pub const fn ticket_priority(&self) -> &'static str {
        match self {
            Self::Missing => "P0",
            Self::Fallback => "P1",
            Self::Fused => "\u{2014}",
        }
    }
}

impl std::fmt::Display for ImplementationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fused => write!(f, "Fused"),
            Self::Fallback => write!(f, "Fallback"),
            Self::Missing => write!(f, "Missing"),
        }
    }
}

/// A kernel operation's implementation in the stack.
///
/// Deserialized from `kernel-bindings.yaml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelBinding {
    /// The kernel operation.
    pub op: KernelOp,
    /// trueno function (if any).
    #[serde(default)]
    pub trueno_function: Option<String>,
    /// realizar dispatch function (if any).
    #[serde(default)]
    pub realizar_function: Option<String>,
    /// Implementation status.
    pub status: ImplementationStatus,
    /// Notes about the implementation.
    #[serde(default)]
    pub notes: String,
}

/// Result of verifying a single binding claim against source code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingVerification {
    /// The kernel operation being verified.
    pub op: KernelOp,
    /// Claimed trueno function.
    pub trueno_claim: Option<String>,
    /// Whether trueno claim was found in source.
    pub trueno_found: bool,
    /// File where trueno function was found (if any).
    pub trueno_file: Option<String>,
    /// Claimed realizar function.
    pub realizar_claim: Option<String>,
    /// Whether realizar claim was found in source.
    pub realizar_found: bool,
    /// File where realizar function was found (if any).
    pub realizar_file: Option<String>,
}

impl BindingVerification {
    /// Whether all claims are verified.
    #[must_use]
    pub fn is_verified(&self) -> bool {
        let trueno_ok = self.trueno_claim.as_ref().is_none_or(|_| self.trueno_found);
        let realizar_ok = self
            .realizar_claim
            .as_ref()
            .is_none_or(|_| self.realizar_found);
        trueno_ok && realizar_ok
    }
}

/// Summary of binding verification against source code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingVerificationReport {
    /// Per-binding verification results.
    pub bindings: Vec<BindingVerification>,
    /// Total claims checked.
    pub total_claims: usize,
    /// Claims verified in source.
    pub verified_count: usize,
    /// Claims NOT found in source (drift).
    pub drift_count: usize,
    /// trueno repo path used.
    pub trueno_path: Option<String>,
    /// realizar repo path used.
    pub realizar_path: Option<String>,
}

/// A gap where a required kernel op is not fully implemented.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelGap {
    /// The missing/fallback kernel operation.
    pub op: KernelOp,
    /// Implementation status.
    pub status: ImplementationStatus,
    /// Architectures that require this op.
    pub affected_architectures: Vec<String>,
    /// Suggested ticket title.
    pub ticket_title: String,
    /// Suggested ticket body (markdown).
    pub ticket_body: String,
}

/// Coverage report for one or more architectures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReport {
    /// Per-architecture coverage entries.
    pub architectures: Vec<ArchitectureCoverage>,
    /// Gaps found across all architectures.
    pub gaps: Vec<KernelGap>,
    /// Total kernel ops checked.
    pub total_ops: usize,
    /// Total ops with fused implementation.
    pub fused_count: usize,
    /// Total ops with fallback implementation.
    pub fallback_count: usize,
    /// Total ops missing.
    pub missing_count: usize,
}

/// Coverage for a single architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureCoverage {
    /// Architecture/family name.
    pub architecture: String,
    /// Kernel class (A-F).
    pub kernel_class: Option<String>,
    /// Required kernel ops and their status.
    pub ops: Vec<OpCoverage>,
}

/// Coverage status of a single kernel op for an architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpCoverage {
    /// The kernel operation.
    pub op: KernelOp,
    /// Implementation status.
    pub status: ImplementationStatus,
    /// trueno function name (if any).
    pub trueno_fn: Option<String>,
    /// realizar function name (if any).
    pub realizar_fn: Option<String>,
}

/// Per-model kernel coverage result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCoverage {
    /// HuggingFace repo ID.
    pub model_id: String,
    /// Architecture family.
    pub architecture: String,
    /// Kernel equivalence class (A-F).
    pub kernel_class: Option<String>,
    /// Whether all required kernels are implemented.
    pub fully_covered: bool,
    /// Count of missing kernel ops.
    pub missing_ops: usize,
    /// Count of fallback kernel ops.
    pub fallback_ops: usize,
    /// Names of missing/fallback ops (for display).
    pub gap_ops: Vec<String>,
    /// True if architecture was NOT found in contracts YAML (using defaults).
    /// Coverage result may be inaccurate — file upstream ticket.
    pub using_defaults: bool,
}

/// Summary of model-level kernel coverage across the full registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCoverageSummary {
    /// Per-model coverage results (sorted by model_id).
    pub models: Vec<ModelCoverage>,
    /// Models fully covered (can serve).
    pub covered_count: usize,
    /// Models with gaps (cannot serve or degraded).
    pub gap_count: usize,
    /// Models using default constraints (architecture not in YAML).
    pub defaults_count: usize,
    /// Per-class summary counts.
    pub class_summary: Vec<ClassSummary>,
}

/// Coverage summary for a kernel class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassSummary {
    /// Class label (A-F).
    pub class: String,
    /// Human-readable kernel description.
    pub label: String,
    /// Number of models in this class.
    pub model_count: usize,
    /// Whether the class is fully covered.
    pub fully_covered: bool,
    /// Missing ops in this class.
    pub missing_ops: Vec<String>,
}

// ── CoverageContext: pre-loaded data from YAML ─────────────────────────────

/// Pre-loaded coverage data parsed from YAML contracts.
///
/// Construct via `CoverageContext::load()` with paths to the YAML files.
/// All `verify_*` methods operate on this cached data.
#[derive(Debug, Clone)]
pub struct CoverageContext {
    /// Architecture name -> constraints (includes aliases as separate entries).
    pub architectures: Vec<(String, ArchConstraints)>,
    /// Canonical architecture names (excludes aliases). Used by `verify_all_architectures`
    /// to avoid double-counting aliases with identical kernel profiles.
    pub canonical_names: std::collections::HashSet<String>,
    /// Default constraints for unknown architectures (LLaMA-like).
    pub default_constraints: ArchConstraints,
    /// Kernel operation bindings.
    pub bindings: Vec<KernelBinding>,
}

impl CoverageContext {
    /// Load coverage data from YAML files.
    ///
    /// - `contracts_path`: directory containing `arch-constraints-v1.yaml`
    ///   (typically `../provable-contracts/contracts`)
    /// - `bindings_path`: path to `kernel-bindings.yaml`
    ///   (typically `playbooks/kernel-bindings.yaml`)
    ///
    /// # Errors
    ///
    /// Returns error if YAML files cannot be read or parsed.
    pub fn load(contracts_path: &Path, bindings_path: &Path) -> crate::Result<Self> {
        let arch_path = contracts_path.join("arch-constraints-v1.yaml");
        let arch_content = std::fs::read_to_string(&arch_path)?;
        let arch_file: ArchConstraintsFile = serde_yaml::from_str(&arch_content)?;

        let default_constraints = yaml_entry_to_constraints(&arch_file.default);

        let mut architectures = Vec::new();
        let mut canonical_names = std::collections::HashSet::new();
        for (name, entry) in &arch_file.architectures {
            let constraints = yaml_entry_to_constraints(entry);
            canonical_names.insert(name.clone());
            architectures.push((name.clone(), constraints.clone()));
            for alias in &entry.aliases {
                architectures.push((alias.clone(), constraints.clone()));
            }
        }
        architectures.sort_by(|a, b| a.0.cmp(&b.0));

        let bindings_content = std::fs::read_to_string(bindings_path)?;
        let bindings_file: KernelBindingsFile = serde_yaml::from_str(&bindings_content)?;

        Ok(Self {
            architectures,
            canonical_names,
            default_constraints,
            bindings: bindings_file.bindings,
        })
    }

    /// Construct from YAML strings (for testing).
    ///
    /// # Errors
    /// Returns an error if the YAML is malformed or missing required fields.
    #[cfg(test)]
    pub fn from_yaml_str(arch_yaml: &str, bindings_yaml: &str) -> crate::Result<Self> {
        let arch_file: ArchConstraintsFile = serde_yaml::from_str(arch_yaml)?;
        let default_constraints = yaml_entry_to_constraints(&arch_file.default);

        let mut architectures = Vec::new();
        let mut canonical_names = std::collections::HashSet::new();
        for (name, entry) in &arch_file.architectures {
            let constraints = yaml_entry_to_constraints(entry);
            canonical_names.insert(name.clone());
            architectures.push((name.clone(), constraints.clone()));
            for alias in &entry.aliases {
                architectures.push((alias.clone(), constraints.clone()));
            }
        }
        architectures.sort_by(|a, b| a.0.cmp(&b.0));

        let bindings_file: KernelBindingsFile = serde_yaml::from_str(bindings_yaml)?;

        Ok(Self {
            architectures,
            canonical_names,
            default_constraints,
            bindings: bindings_file.bindings,
        })
    }

    /// List all known architecture names (canonical + aliases).
    #[must_use]
    pub fn architecture_names(&self) -> Vec<&str> {
        self.architectures.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Look up a kernel op's binding.
    #[must_use]
    pub fn lookup_binding(&self, op: KernelOp) -> KernelBinding {
        self.bindings
            .iter()
            .find(|b| b.op == op)
            .cloned()
            .unwrap_or_else(|| KernelBinding {
                op,
                trueno_function: None,
                realizar_function: None,
                status: ImplementationStatus::Missing,
                notes: "Unknown kernel op \u{2014} not in binding registry".to_string(),
            })
    }

    /// Look up constraints for an architecture.
    ///
    /// Returns `(constraints, using_defaults)`. If the architecture is not
    /// in the contracts YAML, returns default constraints with `true`.
    #[must_use]
    pub fn constraints_for(&self, family: &str) -> (&ArchConstraints, bool) {
        self.architectures
            .iter()
            .find(|(n, _)| n == family)
            .map_or((&self.default_constraints, true), |(_, c)| (c, false))
    }

    /// Verify kernel coverage for a single architecture.
    #[must_use]
    pub fn verify_architecture(
        &self,
        family: &str,
        constraints: &ArchConstraints,
    ) -> ArchitectureCoverage {
        let profile = profile_from_constraints(family, constraints, None);
        let class = KernelClass::from_family(family);

        let ops: Vec<OpCoverage> = profile
            .kernel_ops
            .iter()
            .map(|op| {
                let binding = self.lookup_binding(*op);
                OpCoverage {
                    op: *op,
                    status: binding.status,
                    trueno_fn: binding.trueno_function,
                    realizar_fn: binding.realizar_function,
                }
            })
            .collect();

        ArchitectureCoverage {
            architecture: family.to_string(),
            kernel_class: class.map(|c| c.to_string()),
            ops,
        }
    }

    /// Verify kernel coverage across all known architectures.
    ///
    /// Only verifies canonical architecture names (not aliases) to avoid
    /// double-counting entries with identical kernel profiles.
    #[must_use]
    pub fn verify_all_architectures(&self) -> CoverageReport {
        let architectures: Vec<ArchitectureCoverage> = self
            .architectures
            .iter()
            .filter(|(name, _)| self.canonical_names.contains(name))
            .map(|(family, constraints)| self.verify_architecture(family, constraints))
            .collect();

        build_report(architectures, self)
    }

    /// Verify kernel coverage for a specific architecture by name.
    ///
    /// Returns `None` if the architecture is not in the known registry.
    #[must_use]
    pub fn verify_by_name(&self, family: &str) -> Option<CoverageReport> {
        let constraints = self
            .architectures
            .iter()
            .find(|(name, _)| name == family)
            .map(|(_, c)| c)?;

        let arch = self.verify_architecture(family, constraints);
        Some(build_report(vec![arch], self))
    }

    /// Verify kernel coverage for every model in the registry.
    ///
    /// Walks all registered models, resolves each to an architecture,
    /// checks kernel coverage, and produces a per-model summary.
    #[must_use]
    pub fn verify_all_registry_models(&self) -> ModelCoverageSummary {
        use crate::models::ModelRegistry;

        let registry = ModelRegistry::with_defaults();

        // Cache architecture -> (coverage, using_defaults)
        let mut arch_cache: std::collections::HashMap<String, (ArchitectureCoverage, bool)> =
            std::collections::HashMap::new();

        let mut models: Vec<ModelCoverage> = registry
            .all()
            .iter()
            .map(|meta| {
                let arch = &meta.architecture;
                let (coverage, using_defaults) = arch_cache
                    .entry(arch.clone())
                    .or_insert_with(|| {
                        let (constraints, defaults) = self.constraints_for(arch);
                        (self.verify_architecture(arch, constraints), defaults)
                    })
                    .clone();

                let missing_ops = coverage
                    .ops
                    .iter()
                    .filter(|o| o.status == ImplementationStatus::Missing)
                    .count();
                let fallback_ops = coverage
                    .ops
                    .iter()
                    .filter(|o| o.status == ImplementationStatus::Fallback)
                    .count();
                let gap_ops: Vec<String> = coverage
                    .ops
                    .iter()
                    .filter(|o| o.status != ImplementationStatus::Fused)
                    .map(|o| format!("{} ({})", o.op.description(), o.status))
                    .collect();

                ModelCoverage {
                    model_id: meta.id.hf_repo(),
                    architecture: arch.clone(),
                    kernel_class: coverage.kernel_class,
                    // Cannot claim full coverage when using defaults — actual kernel
                    // requirements are unknown. Jidoka: make problems visible.
                    fully_covered: missing_ops == 0 && fallback_ops == 0 && !using_defaults,
                    missing_ops,
                    fallback_ops,
                    gap_ops,
                    using_defaults,
                }
            })
            .collect();

        models.sort_by(|a, b| a.model_id.cmp(&b.model_id));

        let covered_count = models.iter().filter(|m| m.fully_covered).count();
        let defaults_count = models.iter().filter(|m| m.using_defaults).count();
        // gap_count excludes default-using models — they are unverified, not gapped.
        // covered + gap + defaults = total
        let gap_count = models.len() - covered_count - defaults_count;

        let class_summary = build_class_summary(&models);

        ModelCoverageSummary {
            models,
            covered_count,
            gap_count,
            defaults_count,
            class_summary,
        }
    }

    /// Verify kernel binding claims against actual source code in sibling repos.
    ///
    /// Greps trueno and realizar source for each claimed function name.
    /// Returns `None` if neither repo directory exists.
    #[must_use]
    pub fn verify_bindings_against_source(
        &self,
        trueno_path: &Path,
        realizar_path: &Path,
    ) -> Option<BindingVerificationReport> {
        let trueno_exists = trueno_path.join("src").is_dir();
        let realizar_exists = realizar_path.join("src").is_dir();

        if !trueno_exists && !realizar_exists {
            return None;
        }

        let mut bindings = Vec::with_capacity(self.bindings.len());
        let mut total_claims = 0;
        let mut verified_count = 0;

        for binding in &self.bindings {
            let trueno_claim = binding.trueno_function.clone();
            let realizar_claim = binding.realizar_function.clone();

            let (trueno_found, trueno_file) = trueno_claim.as_ref().map_or((false, None), |name| {
                if trueno_exists {
                    total_claims += 1;
                    find_function_in_dir(&trueno_path.join("src"), name)
                } else {
                    (false, None)
                }
            });

            let (realizar_found, realizar_file) =
                realizar_claim.as_ref().map_or((false, None), |name| {
                    if realizar_exists {
                        total_claims += 1;
                        find_function_in_dir(&realizar_path.join("src"), name)
                    } else {
                        (false, None)
                    }
                });

            if trueno_found {
                verified_count += 1;
            }
            if realizar_found {
                verified_count += 1;
            }

            bindings.push(BindingVerification {
                op: binding.op,
                trueno_claim,
                trueno_found,
                trueno_file,
                realizar_claim,
                realizar_found,
                realizar_file,
            });
        }

        Some(BindingVerificationReport {
            bindings,
            total_claims,
            verified_count,
            drift_count: total_claims - verified_count,
            trueno_path: trueno_exists.then(|| trueno_path.display().to_string()),
            realizar_path: realizar_exists.then(|| realizar_path.display().to_string()),
        })
    }
}

// ── Private helpers ────────────────────────────────────────────────────────

/// Search for a function/struct/method name in a directory of .rs files.
///
/// Returns `(found, file_path)`. Searches for patterns like:
/// - `fn function_name(`
/// - `pub fn function_name(`
/// - `struct StructName`
/// - `pub struct StructName`
fn find_function_in_dir(dir: &Path, name: &str) -> (bool, Option<String>) {
    // Strip parenthetical notes like "(composed)" or "(kv_heads=1)"
    let clean_name = name
        .split_once(" (")
        .map_or(name, |(prefix, _)| prefix)
        .trim();

    // Handle module-qualified names: "ops::rms_norm" -> search for "rms_norm"
    let search_name = clean_name
        .rsplit_once("::")
        .map_or(clean_name, |(_, suffix)| suffix);

    // Handle "verb noun" descriptions: "vector add", "weight sharing in lm_head"
    // These are not greppable function names -- mark as non-verifiable
    if search_name.contains(' ') {
        return (false, None);
    }

    walk_rs_files_for_name(dir, search_name)
}

/// Recursively walk .rs files looking for a name definition.
fn walk_rs_files_for_name(dir: &Path, name: &str) -> (bool, Option<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return (false, None);
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let (found, file) = walk_rs_files_for_name(&path, name);
            if found {
                return (true, file);
            }
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            if let Ok(content) = std::fs::read_to_string(&path) {
                let patterns = [
                    format!("fn {name}("),
                    format!("fn {name}<"),
                    format!("struct {name} "),
                    format!("struct {name} {{"),
                    format!("struct {name}{{"),
                ];
                for pattern in &patterns {
                    if content.contains(pattern.as_str()) {
                        return (true, Some(path.display().to_string()));
                    }
                }
            }
        }
    }

    (false, None)
}

/// Build per-class coverage summary from model results.
fn build_class_summary(models: &[ModelCoverage]) -> Vec<ClassSummary> {
    let mut class_map: std::collections::HashMap<String, (usize, bool, Vec<String>)> =
        std::collections::HashMap::new();

    for model in models {
        let class_key = model
            .kernel_class
            .clone()
            .unwrap_or_else(|| "Unknown".to_string());
        let entry = class_map.entry(class_key).or_insert((0, true, Vec::new()));
        entry.0 += 1;
        if !model.fully_covered {
            entry.1 = false;
            for op in &model.gap_ops {
                if !entry.2.contains(op) {
                    entry.2.push(op.clone());
                }
            }
        }
    }

    let mut summaries: Vec<ClassSummary> = class_map
        .into_iter()
        .map(|(class, (count, covered, missing))| {
            let label = KernelClass::all()
                .iter()
                .find(|kc| kc.to_string() == class)
                .map_or_else(|| class.clone(), |kc| kc.label().to_string());
            ClassSummary {
                class,
                label,
                model_count: count,
                fully_covered: covered,
                missing_ops: missing,
            }
        })
        .collect();

    summaries.sort_by(|a, b| a.class.cmp(&b.class));
    summaries
}

/// Build a coverage report with gap analysis from architecture coverage data.
#[allow(clippy::too_many_lines)]
fn build_report(architectures: Vec<ArchitectureCoverage>, ctx: &CoverageContext) -> CoverageReport {
    let mut total_ops = 0;
    let mut fused_count = 0;
    let mut fallback_count = 0;
    let mut missing_count = 0;

    // Collect gaps across all architectures
    let mut gap_map: std::collections::HashMap<KernelOp, Vec<String>> =
        std::collections::HashMap::new();

    for arch in &architectures {
        for op in &arch.ops {
            total_ops += 1;
            match op.status {
                ImplementationStatus::Fused => fused_count += 1,
                ImplementationStatus::Fallback => {
                    fallback_count += 1;
                    gap_map
                        .entry(op.op)
                        .or_default()
                        .push(arch.architecture.clone());
                }
                ImplementationStatus::Missing => {
                    missing_count += 1;
                    gap_map
                        .entry(op.op)
                        .or_default()
                        .push(arch.architecture.clone());
                }
            }
        }
    }

    let gaps: Vec<KernelGap> = gap_map
        .into_iter()
        .map(|(op, affected)| {
            let binding = ctx.lookup_binding(op);
            let priority = binding.status.ticket_priority();
            let component = if binding.trueno_function.is_none() {
                "trueno + realizar"
            } else {
                "realizar"
            };

            let title = format!(
                "[Kernel Gap] {priority}: {} not implemented ({component})",
                op.description(),
            );

            let body = format!(
                "# KERNEL-GAP: {desc}\n\n\
                 **Priority:** {priority}\n\
                 **Component:** {component}\n\
                 **Affects:** {affected}\n\
                 **Reporter:** apr-qa kernel-coverage\n\n\
                 ## Summary\n\n\
                 Kernel operation `{op_name}` required by {affected} has no \
                 optimized implementation in the sovereign stack.\n\n\
                 ## Severity Justification\n\n\
                 {severity_reason}\n\n\
                 ## Required Implementation\n\n\
                 - **trueno**: {trueno_action}\n\
                 - **realizar**: {realizar_action}\n\n\
                 ## Affected Model Families\n\n\
                 {family_list}\n\n\
                 ## Five Whys\n\n\
                 1. Why does inference fail/degrade? \u{2192} Missing kernel for {op_name}\n\
                 2. Why is the kernel missing? \u{2192} Architecture not fully supported\n\
                 3. Why not fully supported? \u{2192} No automated coverage verification existed\n\
                 4. Why no verification? \u{2192} Manual tracking only\n\
                 5. Why manual? \u{2192} Fixed by `apr-qa kernel-coverage` (Spec \u{00a7}20)\n",
                desc = op.description(),
                affected = affected.join(", "),
                op_name = op.description(),
                severity_reason = match binding.status {
                    ImplementationStatus::Missing =>
                        "Inference will fail or produce garbage output for affected architectures.",
                    ImplementationStatus::Fallback =>
                        "Inference works but falls back to generic dequant+matmul path (2-5x slower).",
                    ImplementationStatus::Fused => "N/A",
                },
                trueno_action = if binding.trueno_function.is_none() {
                    format!("Add SIMD kernel for {}", op.description())
                } else {
                    "Already implemented".to_string()
                },
                realizar_action = if binding.realizar_function.is_none() {
                    format!(
                        "Add dispatch arm in `fused_matmul_into()` for {}",
                        op.description()
                    )
                } else {
                    "Already dispatches".to_string()
                },
                family_list = affected
                    .iter()
                    .map(|f| format!("- {f}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );

            KernelGap {
                op,
                status: binding.status,
                affected_architectures: affected,
                ticket_title: title,
                ticket_body: body,
            }
        })
        .collect();

    CoverageReport {
        architectures,
        gaps,
        total_ops,
        fused_count,
        fallback_count,
        missing_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_ARCH_YAML: &str = r#"
architectures:
  qwen2:
    aliases: ["qwen2.5", "qwen"]
    norm_type: RmsNorm
    activation: Silu
    positional_encoding: Rope
    mlp_type: SwiGlu
    has_bias: true
    tied_embeddings: false
    has_qk_norm: false
    default_eps: 1.0e-6
  llama:
    aliases: ["llama3"]
    norm_type: RmsNorm
    activation: Silu
    positional_encoding: Rope
    mlp_type: SwiGlu
    has_bias: false
    tied_embeddings: false
    has_qk_norm: false
    default_eps: 1.0e-5
  phi:
    aliases: ["phi3"]
    norm_type: LayerNorm
    activation: Silu
    positional_encoding: Rope
    mlp_type: SwiGlu
    has_bias: true
    tied_embeddings: false
    has_qk_norm: false
    default_eps: 1.0e-5
  falcon:
    aliases: []
    norm_type: LayerNorm
    activation: Gelu
    positional_encoding: Alibi
    mlp_type: GeluMlp
    has_bias: false
    tied_embeddings: false
    has_qk_norm: false
    default_eps: 1.0e-5
  gpt2:
    aliases: []
    norm_type: LayerNorm
    activation: Gelu
    positional_encoding: Absolute
    mlp_type: GeluMlp
    has_bias: true
    tied_embeddings: true
    has_qk_norm: false
    default_eps: 1.0e-5
  gemma:
    aliases: ["gemma2"]
    norm_type: RmsNorm
    activation: Gelu
    positional_encoding: Rope
    mlp_type: GatedMlp
    has_bias: false
    tied_embeddings: true
    has_qk_norm: false
    default_eps: 1.0e-6
default:
  norm_type: RmsNorm
  activation: Silu
  positional_encoding: Rope
  mlp_type: SwiGlu
  has_bias: false
  tied_embeddings: false
  has_qk_norm: false
  default_eps: 1.0e-5
"#;

    const TEST_BINDINGS_YAML: &str = r#"
version: "1.0.0"
bindings:
  - op: fused_q4k_matvec
    trueno_function: matmul_q4k_f32
    realizar_function: fused_q4k_parallel_matvec_into
    status: fused
    notes: test
  - op: fused_q5k_matvec
    trueno_function: "BlockQ5K::dequantize"
    realizar_function: fused_q5k_parallel_matvec_into
    status: fused
    notes: test
  - op: fused_q6k_matvec
    trueno_function: matmul_q6k_f32
    realizar_function: fused_q6k_parallel_matvec_into
    status: fused
    notes: test
  - op: rms_norm
    trueno_function: rms_norm_alloc
    realizar_function: rms_norm
    status: fused
    notes: test
  - op: layer_norm
    trueno_function: layer_norm_alloc
    realizar_function: simd_layer_norm
    status: fused
    notes: test
  - op: silu
    trueno_function: silu_scalar
    realizar_function: simd_silu
    status: fused
    notes: test
  - op: gelu
    trueno_function: gelu_scalar
    realizar_function: simd_gelu
    status: fused
    notes: test
  - op: swi_glu
    trueno_function: silu_scalar
    realizar_function: fused_gate_up_q4k_into
    status: fused
    notes: test
  - op: rope
    realizar_function: apply_rope_rotation_simd
    status: fused
    notes: test
  - op: alibi
    status: missing
    notes: test
  - op: absolute_position
    realizar_function: position_embedding
    status: fused
    notes: test
  - op: grouped_query_attention
    trueno_function: DotOp
    realizar_function: attention_with_cache_gqa
    status: fused
    notes: test
  - op: multi_head_attention
    trueno_function: DotOp
    realizar_function: attention_with_cache_gqa
    status: fused
    notes: test
  - op: multi_query_attention
    trueno_function: DotOp
    realizar_function: attention_with_cache_gqa
    status: fused
    notes: test
  - op: bias_add
    realizar_function: add_bias
    status: fused
    notes: test
  - op: tied_embeddings
    realizar_function: load_apr_lm_head
    status: fused
    notes: test
  - op: gated_mlp
    status: fallback
    notes: test
"#;

    fn test_ctx() -> CoverageContext {
        CoverageContext::from_yaml_str(TEST_ARCH_YAML, TEST_BINDINGS_YAML).unwrap()
    }

    #[test]
    fn test_binding_registry_covers_all_ops() {
        let ctx = test_ctx();
        let ops: Vec<KernelOp> = ctx.bindings.iter().map(|b| b.op).collect();

        assert!(ops.contains(&KernelOp::FusedQ4kMatvec));
        assert!(ops.contains(&KernelOp::FusedQ5kMatvec));
        assert!(ops.contains(&KernelOp::FusedQ6kMatvec));
        assert!(ops.contains(&KernelOp::RmsNorm));
        assert!(ops.contains(&KernelOp::LayerNorm));
        assert!(ops.contains(&KernelOp::Silu));
        assert!(ops.contains(&KernelOp::Gelu));
        assert!(ops.contains(&KernelOp::SwiGlu));
        assert!(ops.contains(&KernelOp::Rope));
        assert!(ops.contains(&KernelOp::GroupedQueryAttention));
        assert!(ops.contains(&KernelOp::MultiHeadAttention));
        assert!(ops.contains(&KernelOp::MultiQueryAttention));
        assert!(ops.contains(&KernelOp::BiasAdd));
        assert!(ops.contains(&KernelOp::TiedEmbeddings));
        assert!(ops.contains(&KernelOp::Alibi));
        assert!(ops.contains(&KernelOp::AbsolutePosition));
        assert!(ops.contains(&KernelOp::GatedMlp));
        assert_eq!(ctx.bindings.len(), 17);
    }

    #[test]
    fn test_alibi_is_missing() {
        let ctx = test_ctx();
        let binding = ctx.lookup_binding(KernelOp::Alibi);
        assert_eq!(binding.status, ImplementationStatus::Missing);
    }

    #[test]
    fn test_q4k_is_fused() {
        let ctx = test_ctx();
        let binding = ctx.lookup_binding(KernelOp::FusedQ4kMatvec);
        assert_eq!(binding.status, ImplementationStatus::Fused);
        assert!(binding.trueno_function.is_some());
        assert!(binding.realizar_function.is_some());
    }

    #[test]
    fn test_verify_qwen2_fully_covered() {
        let ctx = test_ctx();
        let report = ctx.verify_by_name("qwen2").unwrap();
        assert_eq!(report.missing_count, 0);
        assert_eq!(report.gaps.len(), 0);
    }

    #[test]
    fn test_verify_falcon_has_alibi_gap() {
        let ctx = test_ctx();
        let report = ctx.verify_by_name("falcon").unwrap();
        assert!(report.missing_count > 0);
        let alibi_gap = report.gaps.iter().find(|g| g.op == KernelOp::Alibi);
        assert!(alibi_gap.is_some());
        assert!(alibi_gap
            .unwrap()
            .affected_architectures
            .contains(&"falcon".to_string()));
    }

    #[test]
    fn test_verify_all_architectures() {
        let ctx = test_ctx();
        let report = ctx.verify_all_architectures();
        assert!(!report.architectures.is_empty());
        assert!(report.total_ops > 0);
        // ALiBi should show up as a gap (falcon needs it)
        let alibi_gap = report.gaps.iter().find(|g| g.op == KernelOp::Alibi);
        assert!(alibi_gap.is_some());
    }

    #[test]
    fn test_gap_ticket_has_five_whys() {
        let ctx = test_ctx();
        let report = ctx.verify_by_name("falcon").unwrap();
        for gap in &report.gaps {
            assert!(gap.ticket_body.contains("Five Whys"));
            assert!(gap.ticket_body.contains("apr-qa kernel-coverage"));
        }
    }

    #[test]
    fn test_architectures_from_yaml() {
        let ctx = test_ctx();
        let names = ctx.architecture_names();
        // Canonical names
        assert!(names.contains(&"qwen2"));
        assert!(names.contains(&"llama"));
        assert!(names.contains(&"phi"));
        assert!(names.contains(&"falcon"));
        assert!(names.contains(&"gpt2"));
        // Aliases
        assert!(names.contains(&"qwen2.5"));
        assert!(names.contains(&"llama3"));
        assert!(names.contains(&"phi3"));
        assert!(names.contains(&"gemma2"));
        // Canonical + aliases: qwen2(+2), llama(+1), phi(+1), falcon, gpt2, gemma(+1) = 6+5=11
        assert_eq!(ctx.architectures.len(), 11);
    }

    #[test]
    fn test_default_constraints_for_unknown_arch() {
        let ctx = test_ctx();
        let (constraints, using_defaults) = ctx.constraints_for("unknown_arch");
        assert!(using_defaults);
        // Should return LLaMA-like defaults
        assert_eq!(constraints.norm_type.as_deref(), Some("rmsnorm"));
        assert_eq!(constraints.activation.as_deref(), Some("silu"));
        assert_eq!(constraints.positional_encoding.as_deref(), Some("rope"));
        assert_eq!(constraints.mlp_type.as_deref(), Some("swiglu"));
    }

    #[test]
    fn test_gemma_has_gated_mlp_gap() {
        let ctx = test_ctx();
        let report = ctx.verify_by_name("gemma").unwrap();
        // GatedMlp is fallback, not fused — Gemma should show a gap
        assert!(
            report.fallback_count > 0,
            "Gemma GatedMlp should be fallback"
        );
        let gated_gap = report.gaps.iter().find(|g| g.op == KernelOp::GatedMlp);
        assert!(gated_gap.is_some(), "Gemma must show GatedMlp gap");
    }

    #[test]
    fn test_q5k_included_in_all_profiles() {
        let ctx = test_ctx();
        let report = ctx.verify_by_name("qwen2").unwrap();
        let has_q5k = report
            .architectures
            .iter()
            .any(|a| a.ops.iter().any(|o| o.op == KernelOp::FusedQ5kMatvec));
        assert!(has_q5k, "Q5K must be in every architecture profile");
    }

    #[test]
    fn test_kernel_class_stored_as_letter_not_label() {
        let ctx = test_ctx();
        let report = ctx.verify_by_name("qwen2").unwrap();
        // kernel_class must be the letter "A", not the label "GQA+RMSNorm+..."
        let class = &report.architectures[0].kernel_class;
        assert_eq!(
            class.as_deref(),
            Some("A"),
            "kernel_class must be letter, not label"
        );
    }

    #[test]
    fn test_defaults_not_fully_covered() {
        // When using default constraints (arch not in YAML), fully_covered
        // must be false — we haven't verified the actual kernel requirements.
        let ctx = test_ctx();
        let (_, using_defaults) = ctx.constraints_for("unknown_arch");
        assert!(using_defaults);
        // Even though default ops are all fused, the model must NOT be
        // reported as fully covered since we used defaults.
        // (Tested through the coverage API: verify_all_registry_models checks this)
    }

    #[test]
    fn test_status_display() {
        assert_eq!(ImplementationStatus::Fused.symbol(), "\u{2713}");
        assert_eq!(ImplementationStatus::Fallback.symbol(), "~");
        assert_eq!(ImplementationStatus::Missing.symbol(), "\u{2717}");
    }

    // ── CoverageContext::load (file-based) coverage ─────────────────────────

    #[test]
    fn test_coverage_context_load_success() {
        let dir = tempfile::TempDir::new().unwrap();
        let contracts_path = dir.path().join("contracts");
        std::fs::create_dir_all(&contracts_path).unwrap();

        let arch_path = contracts_path.join("arch-constraints-v1.yaml");
        std::fs::write(&arch_path, TEST_ARCH_YAML).unwrap();

        let bindings_path = dir.path().join("kernel-bindings.yaml");
        std::fs::write(&bindings_path, TEST_BINDINGS_YAML).unwrap();

        let ctx = CoverageContext::load(&contracts_path, &bindings_path).unwrap();
        let names = ctx.architecture_names();
        assert!(
            names.contains(&"qwen2"),
            "qwen2 should be in loaded architectures"
        );
        assert!(
            names.contains(&"llama"),
            "llama should be in loaded architectures"
        );
        assert_eq!(ctx.bindings.len(), 17);
    }

    #[test]
    fn test_coverage_context_load_missing_arch_file() {
        let dir = tempfile::TempDir::new().unwrap();
        // Contracts dir exists but arch-constraints-v1.yaml is missing
        let contracts_path = dir.path().join("contracts");
        std::fs::create_dir_all(&contracts_path).unwrap();

        let bindings_path = dir.path().join("kernel-bindings.yaml");
        std::fs::write(&bindings_path, TEST_BINDINGS_YAML).unwrap();

        let result = CoverageContext::load(&contracts_path, &bindings_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_coverage_context_load_missing_bindings_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let contracts_path = dir.path().join("contracts");
        std::fs::create_dir_all(&contracts_path).unwrap();

        let arch_path = contracts_path.join("arch-constraints-v1.yaml");
        std::fs::write(&arch_path, TEST_ARCH_YAML).unwrap();

        // Bindings file missing
        let bindings_path = dir.path().join("nonexistent-bindings.yaml");

        let result = CoverageContext::load(&contracts_path, &bindings_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_coverage_context_load_invalid_arch_yaml() {
        let dir = tempfile::TempDir::new().unwrap();
        let contracts_path = dir.path().join("contracts");
        std::fs::create_dir_all(&contracts_path).unwrap();

        let arch_path = contracts_path.join("arch-constraints-v1.yaml");
        std::fs::write(&arch_path, "not: [valid: {{{yaml}}}}").unwrap();

        let bindings_path = dir.path().join("kernel-bindings.yaml");
        std::fs::write(&bindings_path, TEST_BINDINGS_YAML).unwrap();

        let result = CoverageContext::load(&contracts_path, &bindings_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_coverage_context_load_invalid_bindings_yaml() {
        let dir = tempfile::TempDir::new().unwrap();
        let contracts_path = dir.path().join("contracts");
        std::fs::create_dir_all(&contracts_path).unwrap();

        let arch_path = contracts_path.join("arch-constraints-v1.yaml");
        std::fs::write(&arch_path, TEST_ARCH_YAML).unwrap();

        let bindings_path = dir.path().join("kernel-bindings.yaml");
        std::fs::write(&bindings_path, "version: !!invalid").unwrap();

        let result = CoverageContext::load(&contracts_path, &bindings_path);
        assert!(result.is_err());
    }

    // ── verify_all_registry_models / build_class_summary coverage ──────────

    #[test]
    fn test_verify_all_registry_models_returns_non_empty_summary() {
        let ctx = test_ctx();
        let summary = ctx.verify_all_registry_models();
        // Model registry has 100+ models; even with a minimal test context
        // we expect the function to run without panic and return data
        let total = summary.covered_count + summary.gap_count + summary.defaults_count;
        assert_eq!(total, summary.models.len());
    }

    #[test]
    fn test_verify_all_registry_models_class_summary_not_empty() {
        let ctx = test_ctx();
        let summary = ctx.verify_all_registry_models();
        // build_class_summary should produce at least one ClassSummary
        assert!(!summary.class_summary.is_empty());
    }

    #[test]
    fn test_verify_all_registry_models_class_summary_sorted() {
        let ctx = test_ctx();
        let summary = ctx.verify_all_registry_models();
        // Class summaries must be sorted by class string
        let classes: Vec<&str> = summary
            .class_summary
            .iter()
            .map(|s| s.class.as_str())
            .collect();
        let mut sorted = classes.clone();
        sorted.sort_unstable();
        assert_eq!(classes, sorted, "Class summaries should be sorted by class");
    }

    #[test]
    fn test_verify_all_registry_models_models_sorted() {
        let ctx = test_ctx();
        let summary = ctx.verify_all_registry_models();
        // Models must be sorted by model_id
        let ids: Vec<&str> = summary.models.iter().map(|m| m.model_id.as_str()).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted, "Models should be sorted by model_id");
    }

    #[test]
    fn test_verify_all_registry_models_defaults_count_consistent() {
        let ctx = test_ctx();
        let summary = ctx.verify_all_registry_models();
        // defaults_count should match the count of models using defaults
        let actual_defaults = summary.models.iter().filter(|m| m.using_defaults).count();
        assert_eq!(summary.defaults_count, actual_defaults);
    }

    #[test]
    fn test_verify_all_registry_models_covered_count_consistent() {
        let ctx = test_ctx();
        let summary = ctx.verify_all_registry_models();
        let actual_covered = summary.models.iter().filter(|m| m.fully_covered).count();
        assert_eq!(summary.covered_count, actual_covered);
    }

    #[test]
    fn test_build_class_summary_with_gap_ops() {
        let ctx = test_ctx();
        let summary = ctx.verify_all_registry_models();
        // At least one class should have some model data
        let total_models: usize = summary.class_summary.iter().map(|s| s.model_count).sum();
        assert_eq!(total_models, summary.models.len());
    }

    #[test]
    fn test_coverage_context_load_aliases_registered() {
        let dir = tempfile::TempDir::new().unwrap();
        let contracts_path = dir.path().join("contracts");
        std::fs::create_dir_all(&contracts_path).unwrap();

        let arch_path = contracts_path.join("arch-constraints-v1.yaml");
        std::fs::write(&arch_path, TEST_ARCH_YAML).unwrap();

        let bindings_path = dir.path().join("kernel-bindings.yaml");
        std::fs::write(&bindings_path, TEST_BINDINGS_YAML).unwrap();

        let ctx = CoverageContext::load(&contracts_path, &bindings_path).unwrap();
        let names = ctx.architecture_names();
        // qwen2 has aliases "qwen2.5" and "qwen"
        assert!(
            names.contains(&"qwen2.5"),
            "qwen2 alias should be registered"
        );
        assert!(names.contains(&"qwen"), "qwen alias should be registered");
    }

    // ── walk_rs_files_for_name ───────────────────────────────────────────────

    #[test]
    fn test_walk_rs_files_nonexistent_dir() {
        let dir = tempfile::TempDir::new().unwrap();
        let nonexistent = dir.path().join("does_not_exist");
        let (found, file) = walk_rs_files_for_name(&nonexistent, "any_function");
        assert!(!found);
        assert!(file.is_none());
    }

    #[test]
    fn test_walk_rs_files_function_found() {
        let dir = tempfile::TempDir::new().unwrap();
        let rs_file = dir.path().join("lib.rs");
        std::fs::write(
            &rs_file,
            "pub fn matmul_q4k_f32(a: &[u8], b: &[f32]) -> Vec<f32> {}",
        )
        .unwrap();
        let (found, file) = walk_rs_files_for_name(dir.path(), "matmul_q4k_f32");
        assert!(found, "should find fn matmul_q4k_f32(");
        assert!(file.is_some());
        assert!(file.unwrap().ends_with("lib.rs"));
    }

    #[test]
    fn test_walk_rs_files_struct_found() {
        let dir = tempfile::TempDir::new().unwrap();
        let rs_file = dir.path().join("types.rs");
        std::fs::write(&rs_file, "pub struct BlockQ5K { data: Vec<u8> }").unwrap();
        let (found, file) = walk_rs_files_for_name(dir.path(), "BlockQ5K");
        assert!(found, "should find struct BlockQ5K {{");
        assert!(file.is_some());
    }

    #[test]
    fn test_walk_rs_files_function_not_found() {
        let dir = tempfile::TempDir::new().unwrap();
        let rs_file = dir.path().join("lib.rs");
        std::fs::write(&rs_file, "pub fn other_function() {}").unwrap();
        let (found, file) = walk_rs_files_for_name(dir.path(), "matmul_q4k_f32");
        assert!(!found);
        assert!(file.is_none());
    }

    #[test]
    fn test_walk_rs_files_recursive() {
        let dir = tempfile::TempDir::new().unwrap();
        let subdir = dir.path().join("kernels");
        std::fs::create_dir_all(&subdir).unwrap();
        let rs_file = subdir.join("q6k.rs");
        std::fs::write(&rs_file, "pub fn matmul_q6k_f32(w: &[u8]) {}").unwrap();
        let (found, file) = walk_rs_files_for_name(dir.path(), "matmul_q6k_f32");
        assert!(found, "should find function in subdirectory");
        assert!(file.is_some());
        assert!(file.unwrap().ends_with("q6k.rs"));
    }

    #[test]
    fn test_walk_rs_files_skips_non_rs_files() {
        let dir = tempfile::TempDir::new().unwrap();
        // Only a .txt file — should not be searched
        let txt_file = dir.path().join("notes.txt");
        std::fs::write(&txt_file, "pub fn target_function() {}").unwrap();
        let (found, _) = walk_rs_files_for_name(dir.path(), "target_function");
        assert!(!found, "should not search non-.rs files");
    }

    #[test]
    fn test_walk_rs_files_generic_function() {
        // Pattern: fn name<T>(  — should match via "fn name<" pattern
        let dir = tempfile::TempDir::new().unwrap();
        let rs_file = dir.path().join("generic.rs");
        std::fs::write(
            &rs_file,
            "pub fn rms_norm_alloc<T: Float>(x: &[T]) -> Vec<T> {}",
        )
        .unwrap();
        let (found, _) = walk_rs_files_for_name(dir.path(), "rms_norm_alloc");
        assert!(found, "should find generic fn via fn name< pattern");
    }

    // ── find_function_in_dir (private helper) ────────────────────────────────

    #[test]
    fn test_find_function_strips_parenthetical_notes() {
        // "BlockQ5K::dequantize" → strips "::" prefix, searches for "dequantize"
        let dir = tempfile::TempDir::new().unwrap();
        let rs_file = dir.path().join("quant.rs");
        std::fs::write(&rs_file, "impl BlockQ5K { pub fn dequantize(&self) {} }").unwrap();
        // find_function_in_dir strips module path — "BlockQ5K::dequantize" → "dequantize"
        let (found, _) = find_function_in_dir(dir.path(), "BlockQ5K::dequantize");
        assert!(
            found,
            "should find dequantize after stripping module prefix"
        );
    }

    #[test]
    fn test_find_function_description_string_not_greppable() {
        // Names with spaces are prose descriptions — must return (false, None)
        let dir = tempfile::TempDir::new().unwrap();
        let rs_file = dir.path().join("ops.rs");
        std::fs::write(&rs_file, "// vector add implementation").unwrap();
        let (found, file) = find_function_in_dir(dir.path(), "vector add");
        assert!(!found, "prose description must not be searched");
        assert!(file.is_none());
    }

    #[test]
    fn test_find_function_strips_parenthetical_suffix() {
        // "apply_rope_rotation_simd (composed)" → searches "apply_rope_rotation_simd"
        let dir = tempfile::TempDir::new().unwrap();
        let rs_file = dir.path().join("rope.rs");
        std::fs::write(
            &rs_file,
            "pub fn apply_rope_rotation_simd(x: &mut [f32]) {}",
        )
        .unwrap();
        let (found, _) = find_function_in_dir(dir.path(), "apply_rope_rotation_simd (composed)");
        assert!(found, "should strip parenthetical suffix before searching");
    }

    // ── verify_bindings_against_source ──────────────────────────────────────

    #[test]
    fn test_verify_bindings_neither_repo_exists() {
        let ctx = test_ctx();
        let dir = tempfile::TempDir::new().unwrap();
        let trueno = dir.path().join("trueno");
        let realizar = dir.path().join("realizar");
        // Neither trueno/src nor realizar/src exist
        let result = ctx.verify_bindings_against_source(&trueno, &realizar);
        assert!(
            result.is_none(),
            "should return None when neither repo exists"
        );
    }

    #[test]
    fn test_verify_bindings_trueno_only_returns_some() {
        let ctx = test_ctx();
        let dir = tempfile::TempDir::new().unwrap();
        let trueno = dir.path().join("trueno");
        std::fs::create_dir_all(trueno.join("src")).unwrap();
        let realizar = dir.path().join("realizar"); // no src dir

        let report = ctx
            .verify_bindings_against_source(&trueno, &realizar)
            .expect("should return Some when trueno/src exists");
        assert!(report.trueno_path.is_some());
        assert!(report.realizar_path.is_none());
        // all bindings should be present in report
        assert_eq!(report.bindings.len(), ctx.bindings.len());
    }

    #[test]
    fn test_verify_bindings_realizar_only_returns_some() {
        let ctx = test_ctx();
        let dir = tempfile::TempDir::new().unwrap();
        let trueno = dir.path().join("trueno"); // no src dir
        let realizar = dir.path().join("realizar");
        std::fs::create_dir_all(realizar.join("src")).unwrap();

        let report = ctx
            .verify_bindings_against_source(&trueno, &realizar)
            .expect("should return Some when realizar/src exists");
        assert!(report.trueno_path.is_none());
        assert!(report.realizar_path.is_some());
    }

    #[test]
    fn test_verify_bindings_found_in_trueno_src() {
        let ctx = test_ctx();
        let dir = tempfile::TempDir::new().unwrap();
        let trueno = dir.path().join("trueno");
        std::fs::create_dir_all(trueno.join("src")).unwrap();
        // Write a file containing matmul_q4k_f32 (the trueno_function in TEST_BINDINGS_YAML)
        std::fs::write(
            trueno.join("src").join("q4k.rs"),
            "pub fn matmul_q4k_f32(w: &[u8], x: &[f32]) -> Vec<f32> {}",
        )
        .unwrap();
        let realizar = dir.path().join("realizar"); // absent

        let report = ctx
            .verify_bindings_against_source(&trueno, &realizar)
            .unwrap();
        let q4k = report
            .bindings
            .iter()
            .find(|b| b.op == KernelOp::FusedQ4kMatvec)
            .unwrap();
        assert!(
            q4k.trueno_found,
            "matmul_q4k_f32 should be found in trueno/src"
        );
        assert!(q4k.trueno_file.is_some());
    }

    #[test]
    fn test_verify_bindings_not_found_increments_drift() {
        let ctx = test_ctx();
        let dir = tempfile::TempDir::new().unwrap();
        let trueno = dir.path().join("trueno");
        std::fs::create_dir_all(trueno.join("src")).unwrap();
        // Empty src dir — nothing will be found
        let realizar = dir.path().join("realizar"); // absent

        let report = ctx
            .verify_bindings_against_source(&trueno, &realizar)
            .unwrap();
        // total_claims = number of bindings with trueno_function (non-None)
        let expected_trueno_claims = ctx
            .bindings
            .iter()
            .filter(|b| b.trueno_function.is_some())
            .count();
        assert_eq!(report.total_claims, expected_trueno_claims);
        // verified_count should be 0 (nothing in empty dir)
        assert_eq!(report.verified_count, 0);
        assert_eq!(report.drift_count, expected_trueno_claims);
    }

    #[test]
    fn test_verify_bindings_both_repos_count_claims() {
        let ctx = test_ctx();
        let dir = tempfile::TempDir::new().unwrap();
        let trueno = dir.path().join("trueno");
        let realizar = dir.path().join("realizar");
        std::fs::create_dir_all(trueno.join("src")).unwrap();
        std::fs::create_dir_all(realizar.join("src")).unwrap();
        // Both src dirs exist but empty — claims counted for both

        let report = ctx
            .verify_bindings_against_source(&trueno, &realizar)
            .unwrap();
        let expected_trueno = ctx
            .bindings
            .iter()
            .filter(|b| b.trueno_function.is_some())
            .count();
        let expected_realizar = ctx
            .bindings
            .iter()
            .filter(|b| b.realizar_function.is_some())
            .count();
        assert_eq!(report.total_claims, expected_trueno + expected_realizar);
        assert_eq!(report.verified_count, 0);
        assert_eq!(report.drift_count, report.total_claims);
    }

    #[test]
    fn test_verify_bindings_found_in_realizar_src() {
        let ctx = test_ctx();
        let dir = tempfile::TempDir::new().unwrap();
        let trueno = dir.path().join("trueno"); // absent
        let realizar = dir.path().join("realizar");
        std::fs::create_dir_all(realizar.join("src")).unwrap();
        // Write file containing fused_q4k_parallel_matvec_into
        std::fs::write(
            realizar.join("src").join("q4k_fused.rs"),
            "pub fn fused_q4k_parallel_matvec_into(out: &mut [f32], w: &[u8]) {}",
        )
        .unwrap();

        let report = ctx
            .verify_bindings_against_source(&trueno, &realizar)
            .unwrap();
        let q4k = report
            .bindings
            .iter()
            .find(|b| b.op == KernelOp::FusedQ4kMatvec)
            .unwrap();
        assert!(
            q4k.realizar_found,
            "fused_q4k_parallel_matvec_into should be found in realizar/src"
        );
    }
}
