//! KREG-001 (aprender#4539): the kernel registry at run time.
//!
//! `kernel-registry.json` (next to this crate's `Cargo.toml`, so `include_str!` works in a
//! crates.io build) lists every (backend, arch, isa_features, qtype, layout) a selector may
//! dispatch. Its shape is
//! `contracts/kernel-registry-v1.yaml`, checked by `pv lint --gate shapes`. This module is the
//! other half: a selector asks [`admit`] before it dispatches, and a combination with no row is an
//! `Err` naming it — never a silent fallback. A wrong kernel is then unloadable rather than
//! merely wrong.
//!
//! The table is built once, keyed by `(backend, ggml type id)`; each cell holds the rows for that
//! pair, and a [`Target`] (arch + ISA features, APR-OBS-001 §2.10 AC-1/AC-2) picks the most
//! specific row it satisfies. The only admissible layout is row-major (LAYOUT-001).

use std::sync::OnceLock;

use serde::Deserialize;

use crate::error::{RealizarError, Result};

/// The registry document, embedded at compile time.
const REGISTRY_JSON: &str = include_str!("../kernel-registry.json");

/// Type ids at or above this have no slot; every GGML/APR id in use is below it.
const MAX_TYPE_ID: usize = 256;

/// Row indices are `u16`; a document this long cannot be indexed.
const NO_ROW: u16 = u16::MAX;

/// Where a kernel runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// realizar's CPU (trueno SIMD) kernels.
    Cpu,
    /// realizar's CUDA kernels.
    Cuda,
    /// trueno's wgpu kernels.
    Wgpu,
    /// Apple Metal. In the key now (la-73, 0.73 E5); no row is registered yet, so every Metal
    /// combination is refused.
    Metal,
}

impl Backend {
    const ALL: [Self; 4] = [Self::Cpu, Self::Cuda, Self::Wgpu, Self::Metal];

    fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Wgpu => "wgpu",
            Self::Metal => "metal",
        }
    }

    fn slot(self) -> usize {
        match self {
            Self::Cpu => 0,
            Self::Cuda => 1,
            Self::Wgpu => 2,
            Self::Metal => 3,
        }
    }

    fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|b| b.as_str() == s)
    }
}

/// The machine a kernel is asked to run on: arch plus the ISA features it has (AC-1/AC-2).
/// A row with `arch: any` and `isa_features: none` matches every target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target {
    arch: String,
    isa: Vec<String>,
}

impl Target {
    /// A target with an explicit arch and feature set.
    pub fn new(arch: &str, isa: &[&str]) -> Self {
        Self {
            arch: arch.to_string(),
            isa: isa.iter().map(|f| (*f).to_string()).collect(),
        }
    }

    /// A target nothing is known about: it matches only `arch: any`, `isa_features: none` rows.
    /// Used for device backends until the device's arch (e.g. `sm_89`) is plumbed through.
    pub fn generic() -> Self {
        Self::new("any", &[])
    }

    /// The CPU this process runs on, detected once.
    pub fn host() -> &'static Self {
        static HOST: OnceLock<Target> = OnceLock::new();
        HOST.get_or_init(|| Self::new(std::env::consts::ARCH, &host_isa_features()))
    }

    fn satisfies(&self, row: &KernelRow) -> bool {
        (row.arch == "any" || row.arch == self.arch)
            && row_isa(row).all(|f| self.isa.iter().any(|h| h == f))
    }
}

/// The ISA features a CPU row may require, as detected on this host.
fn host_isa_features() -> Vec<&'static str> {
    let mut found = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        let probes: [(&'static str, bool); 6] = [
            ("sse4.1", std::arch::is_x86_feature_detected!("sse4.1")),
            ("avx2", std::arch::is_x86_feature_detected!("avx2")),
            ("fma", std::arch::is_x86_feature_detected!("fma")),
            ("avx512f", std::arch::is_x86_feature_detected!("avx512f")),
            ("avx512bw", std::arch::is_x86_feature_detected!("avx512bw")),
            (
                "avx512vnni",
                std::arch::is_x86_feature_detected!("avx512vnni"),
            ),
        ];
        found.extend(probes.iter().filter(|p| p.1).map(|p| p.0));
    }
    #[cfg(target_arch = "aarch64")]
    {
        let probes: [(&'static str, bool); 4] = [
            ("neon", std::arch::is_aarch64_feature_detected!("neon")),
            (
                "dotprod",
                std::arch::is_aarch64_feature_detected!("dotprod"),
            ),
            ("i8mm", std::arch::is_aarch64_feature_detected!("i8mm")),
            ("sve", std::arch::is_aarch64_feature_detected!("sve")),
        ];
        found.extend(probes.iter().filter(|p| p.1).map(|p| p.0));
    }
    found
}

/// A row's required ISA features (`none` is the empty set).
fn row_isa(row: &KernelRow) -> impl Iterator<Item = &str> {
    row.isa_features.split('+').filter(|f| *f != "none")
}

/// How specific a row is: a named arch outranks `any`, then more required features win.
fn specificity(row: &KernelRow) -> usize {
    usize::from(row.arch != "any") * 1024 + row_isa(row).count()
}

/// The memory layout of the weight the kernel will read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// The only layout APR and realizar hold (LAYOUT-001/002).
    RowMajor,
    /// GGML's native layout. Named so a caller holding it is refused by name, not by accident.
    ColMajor,
}

impl Layout {
    fn as_str(self) -> &'static str {
        match self {
            Self::RowMajor => "row_major",
            Self::ColMajor => "col_major",
        }
    }
}

/// One registry row, as `contracts/kernel-registry-v1.yaml` shapes it.
#[derive(Debug, Clone, Deserialize)]
pub struct KernelRow {
    /// Stable id, `<backend>.<op>.<qtype>[.<arch>[.<isa>]]` — what OBS-15 records as
    /// `kernel_path.entries[].kernel_id` (AC-4).
    pub kernel_id: String,
    /// `matvec` or `gemv`.
    pub op: String,
    /// The qtype name, e.g. `Q4_K`.
    pub qtype: String,
    /// The GGML (or APR) type id the selector matches on.
    pub ggml_type: u32,
    /// Always `row_major` for an admitted row.
    pub layout: String,
    /// `cpu`, `cuda` or `wgpu`.
    pub backend: String,
    /// Target architecture; `any` until a row is arch-specific.
    pub arch: String,
    /// `+`-joined ISA features the row requires; `none` for baseline.
    pub isa_features: String,
    /// Elements per quant block.
    pub block_elems: u32,
    /// Accumulator precision.
    pub accumulate: String,
    /// Activation precision inside the kernel (AC-4).
    pub precision: String,
    /// The M the kernel serves: `m1` or `m_any` (AC-4).
    pub shape_class: String,
    /// `unmeasured` or the path of a tolerance receipt.
    pub tolerance: String,
    /// Repo-relative file holding the kernel.
    pub source_file: String,
    /// The kernel's function name in `source_file`.
    pub source_fn: String,
    /// The selector that dispatches to it.
    pub selector: String,
    /// The contract that governs it.
    pub contract: String,
}

#[derive(Deserialize)]
struct Document {
    kernels: Vec<KernelRow>,
}

/// The parsed registry and its `(backend, type id) -> rows` table.
pub struct Registry {
    rows: Vec<KernelRow>,
    table: Vec<Vec<u16>>,
}

/// The table cell for `(backend slot, type id)`.
fn cell_index(slot: (usize, usize)) -> usize {
    slot.0 * MAX_TYPE_ID + slot.1
}

/// Two rows collide when they share backend, type id, arch and ISA feature set.
fn same_key(a: &KernelRow, b: &KernelRow) -> bool {
    a.arch == b.arch && isa_set(a) == isa_set(b)
}

/// A row's ISA features, order-free.
fn isa_set(row: &KernelRow) -> Vec<&str> {
    let mut f: Vec<&str> = row_isa(row).collect();
    f.sort_unstable();
    f
}

impl Registry {
    /// Parse and index a registry document. Refuses — rather than skips — a row this table
    /// could not answer for honestly: an unknown backend, a non-row-major layout, an id with no
    /// slot, or a second row for the same `(backend, type id, arch, isa_features)`.
    pub fn parse(json: &str) -> std::result::Result<Self, String> {
        let doc: Document =
            serde_json::from_str(json).map_err(|e| format!("kernel registry: {e}"))?;
        if doc.kernels.len() >= usize::from(NO_ROW) {
            return Err(format!(
                "kernel registry: {} rows overflow the table",
                doc.kernels.len()
            ));
        }
        let mut table = vec![Vec::new(); Backend::ALL.len() * MAX_TYPE_ID];
        for (i, row) in doc.kernels.iter().enumerate() {
            let cell = &mut table[cell_index(index_slot(row)?)];
            if let Some(&other) = cell
                .iter()
                .find(|&&j| same_key(&doc.kernels[usize::from(j)], row))
            {
                return Err(format!(
                    "kernel registry: rows `{}` and `{}` both claim ({}, type {}, {}, {})",
                    doc.kernels[usize::from(other)].kernel_id,
                    row.kernel_id,
                    row.backend,
                    row.ggml_type,
                    row.arch,
                    row.isa_features
                ));
            }
            cell.push(u16::try_from(i).map_err(|e| format!("kernel registry: {e}"))?);
        }
        Ok(Self {
            rows: doc.kernels,
            table,
        })
    }

    /// Every row, in document order.
    pub fn rows(&self) -> &[KernelRow] {
        &self.rows
    }

    /// The row for `(backend, type id, layout)` on the backend's default target: the host CPU
    /// for [`Backend::Cpu`], [`Target::generic`] otherwise.
    pub fn admit(&self, backend: Backend, ggml_type: u32, layout: Layout) -> Result<&KernelRow> {
        let target = match backend {
            Backend::Cpu => Target::host().clone(),
            _ => Target::generic(),
        };
        self.admit_for(backend, &target, ggml_type, layout)
    }

    /// The most specific row for `(backend, target, type id, layout)`, or an error naming the
    /// combination.
    pub fn admit_for(
        &self,
        backend: Backend,
        target: &Target,
        ggml_type: u32,
        layout: Layout,
    ) -> Result<&KernelRow> {
        let row = usize::try_from(ggml_type)
            .ok()
            .filter(|&t| t < MAX_TYPE_ID && layout == Layout::RowMajor)
            .and_then(|t| {
                self.table[cell_index((backend.slot(), t))]
                    .iter()
                    .map(|&i| &self.rows[usize::from(i)])
                    .filter(|r| target.satisfies(r))
                    .max_by_key(|r| specificity(r))
            });
        row.ok_or_else(|| refusal(backend, target, ggml_type, layout))
    }
}

/// `(backend slot, type id)` for a row, or why it has none.
fn index_slot(row: &KernelRow) -> std::result::Result<(usize, usize), String> {
    let backend = Backend::parse(&row.backend).ok_or_else(|| {
        format!(
            "kernel registry: row `{}` has unknown backend `{}`",
            row.kernel_id, row.backend
        )
    })?;
    if row.layout != Layout::RowMajor.as_str() {
        return Err(format!(
            "kernel registry: row `{}` has layout `{}`; only row_major is admissible (LAYOUT-001)",
            row.kernel_id, row.layout
        ));
    }
    let id = usize::try_from(row.ggml_type)
        .ok()
        .filter(|&t| t < MAX_TYPE_ID)
        .ok_or_else(|| {
            format!(
                "kernel registry: row `{}` type id {} has no slot",
                row.kernel_id, row.ggml_type
            )
        })?;
    Ok((backend.slot(), id))
}

fn refusal(backend: Backend, target: &Target, ggml_type: u32, layout: Layout) -> RealizarError {
    RealizarError::UnsupportedOperation {
        operation: "kernel_registry::admit".to_string(),
        reason: format!(
            "no registered kernel for backend={} arch={} isa={} ggml_type={ggml_type} layout={} — add a row to \
             crates/aprender-serve/kernel-registry.json (shape: contracts/kernel-registry-v1.yaml) \
             before dispatching it (KREG-001, aprender#4539)",
            backend.as_str(),
            target.arch,
            target.isa.join("+"),
            layout.as_str()
        ),
    }
}

/// The embedded registry, parsed once.
pub fn registry() -> Result<&'static Registry> {
    static REGISTRY: OnceLock<std::result::Result<Registry, String>> = OnceLock::new();
    REGISTRY
        .get_or_init(|| Registry::parse(REGISTRY_JSON))
        .as_ref()
        .map_err(|e| RealizarError::InvalidConfiguration(e.clone()))
}

/// Ask the embedded registry whether `(backend, type id, layout)` may be dispatched.
pub fn admit(backend: Backend, ggml_type: u32, layout: Layout) -> Result<&'static KernelRow> {
    registry()?.admit(backend, ggml_type, layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GGUF_TYPE_Q4_K, GGUF_TYPE_Q6_K};

    fn row_json(id: &str, backend: &str, ggml_type: u32, layout: &str) -> String {
        arch_row_json(id, backend, ggml_type, layout, "any", "none")
    }

    fn arch_row_json(
        id: &str,
        backend: &str,
        ggml_type: u32,
        layout: &str,
        arch: &str,
        isa: &str,
    ) -> String {
        format!(
            r#"{{"kernel_id":"{id}","op":"matvec","qtype":"Q4_K","ggml_type":{ggml_type},"layout":"{layout}",
            "backend":"{backend}","arch":"{arch}","isa_features":"{isa}","block_elems":256,
            "accumulate":"f32","precision":"f32","shape_class":"m_any",
            "tolerance":"unmeasured","source_file":"crates/x/src/a.rs","source_fn":"f",
            "selector":"selector::fn","contract":"contracts/tensor-layout-v1.yaml"}}"#
        )
    }

    fn doc(rows: &[String]) -> String {
        format!(r#"{{"kernels":[{}]}}"#, rows.join(","))
    }

    #[test]
    fn the_embedded_registry_parses_and_is_non_empty() {
        let r = registry().expect("embedded registry parses");
        assert!(!r.rows().is_empty());
    }

    #[test]
    fn every_row_is_admitted_by_its_own_key() {
        let r = registry().expect("registry");
        for row in r.rows() {
            let backend = Backend::parse(&row.backend).expect("known backend");
            let isa: Vec<&str> = row_isa(row).collect();
            let target = Target::new(&row.arch, &isa);
            let got = r
                .admit_for(backend, &target, row.ggml_type, Layout::RowMajor)
                .expect("admitted");
            assert_eq!(got.kernel_id, row.kernel_id);
        }
    }

    /// FALSIFY-KREG-004: a col-major Q4_K/Q6_K kernel on GGUF data is refused on every backend.
    #[test]
    fn falsify_kreg_004_col_major_is_refused() {
        for backend in Backend::ALL {
            for t in [GGUF_TYPE_Q4_K, GGUF_TYPE_Q6_K] {
                let err = admit(backend, t, Layout::ColMajor).expect_err("col-major admitted");
                assert!(err.to_string().contains("layout=col_major"), "{err}");
            }
        }
    }

    /// FALSIFY-KREG-004: an id with no row is refused, including ids past the table's end.
    #[test]
    fn falsify_kreg_004_an_unregistered_type_id_is_refused() {
        for backend in Backend::ALL {
            for t in [99, 255, 256, u32::MAX] {
                let err =
                    admit(backend, t, Layout::RowMajor).expect_err("unregistered id admitted");
                assert!(err.to_string().contains(&format!("ggml_type={t}")), "{err}");
            }
        }
    }

    #[test]
    fn the_key_includes_the_backend() {
        assert!(admit(Backend::Cuda, GGUF_TYPE_Q6_K, Layout::RowMajor).is_ok());
        assert!(admit(Backend::Wgpu, GGUF_TYPE_Q6_K, Layout::RowMajor).is_err());
        assert!(admit(Backend::Metal, GGUF_TYPE_Q4_K, Layout::RowMajor).is_err());
    }

    #[test]
    fn a_col_major_row_is_refused_at_parse() {
        let e = Registry::parse(&doc(&[row_json("cpu.matvec.q4_k", "cpu", 12, "col_major")]))
            .err()
            .expect("col-major row indexed");
        assert!(e.contains("only row_major"), "{e}");
    }

    #[test]
    fn a_duplicate_key_is_refused_at_parse() {
        let rows = [
            row_json("cpu.matvec.a", "cpu", 12, "row_major"),
            row_json("cpu.matvec.b", "cpu", 12, "row_major"),
        ];
        let e = Registry::parse(&doc(&rows))
            .err()
            .expect("duplicate indexed");
        assert!(e.contains("both claim"), "{e}");
    }

    #[test]
    fn an_unknown_backend_and_an_unslotted_id_are_refused_at_parse() {
        let e = Registry::parse(&doc(&[row_json("tpu.matvec.x", "tpu", 12, "row_major")]))
            .err()
            .expect("unknown backend indexed");
        assert!(e.contains("unknown backend"), "{e}");
        let e = Registry::parse(&doc(&[row_json("cpu.matvec.x", "cpu", 256, "row_major")]))
            .err()
            .expect("id 256 indexed");
        assert!(e.contains("has no slot"), "{e}");
    }

    #[test]
    fn a_parsed_row_is_found_and_its_neighbours_are_not() {
        let r = Registry::parse(&doc(&[row_json("cpu.matvec.q4_k", "cpu", 12, "row_major")]))
            .expect("parses");
        assert!(r.admit(Backend::Cpu, 12, Layout::RowMajor).is_ok());
        assert!(r.admit(Backend::Cpu, 11, Layout::RowMajor).is_err());
        assert!(r.admit(Backend::Cpu, 13, Layout::RowMajor).is_err());
        assert!(r.admit(Backend::Cuda, 12, Layout::RowMajor).is_err());
    }

    /// AC-1/AC-2: the key carries arch and ISA features; the most specific satisfied row wins,
    /// an arch-specific row never serves another arch, and a missing feature falls back to the
    /// baseline row — or is refused when there is none.
    #[test]
    fn the_key_includes_arch_and_isa_features() {
        let rows = [
            arch_row_json("cpu.matvec.q4_k", "cpu", 12, "row_major", "any", "none"),
            arch_row_json(
                "cpu.matvec.q4_k.x86_64.avx2",
                "cpu",
                12,
                "row_major",
                "x86_64",
                "avx2+fma",
            ),
            arch_row_json(
                "cpu.matvec.q4_k.aarch64.neon",
                "cpu",
                12,
                "row_major",
                "aarch64",
                "neon",
            ),
            arch_row_json(
                "cpu.matvec.q4_k.x86_64.avx512",
                "cpu",
                12,
                "row_major",
                "x86_64",
                "avx512f",
            ),
            arch_row_json(
                "cpu.matvec.q6_k.aarch64.sve",
                "cpu",
                14,
                "row_major",
                "aarch64",
                "sve",
            ),
        ];
        let r = Registry::parse(&doc(&rows)).expect("parses");
        let pick = |arch: &str, isa: &[&str], t: u32| {
            r.admit_for(Backend::Cpu, &Target::new(arch, isa), t, Layout::RowMajor)
                .map(|k| k.kernel_id.clone())
        };
        assert_eq!(
            pick("x86_64", &["avx2", "fma"], 12).expect("x86"),
            "cpu.matvec.q4_k.x86_64.avx2"
        );
        assert_eq!(
            pick("x86_64", &["avx2"], 12).expect("no fma"),
            "cpu.matvec.q4_k"
        );
        assert_eq!(
            pick("aarch64", &["neon"], 12).expect("arm"),
            "cpu.matvec.q4_k.aarch64.neon"
        );
        assert_eq!(
            pick("x86_64", &["avx512f"], 12).expect("same arch, other features"),
            "cpu.matvec.q4_k.x86_64.avx512"
        );
        assert_eq!(
            pick("riscv64", &[], 12).expect("generic"),
            "cpu.matvec.q4_k"
        );
        assert!(pick("aarch64", &["neon"], 14).is_err());
        assert_eq!(
            pick("aarch64", &["sve"], 14).expect("sve"),
            "cpu.matvec.q6_k.aarch64.sve"
        );
        let named = [
            arch_row_json("cpu.matvec.q8_0", "cpu", 8, "row_major", "any", "avx2"),
            arch_row_json(
                "cpu.matvec.q8_0.x86_64",
                "cpu",
                8,
                "row_major",
                "x86_64",
                "none",
            ),
        ];
        let n = Registry::parse(&doc(&named)).expect("parses");
        let t = Target::new("x86_64", &["avx2"]);
        let got = n
            .admit_for(Backend::Cpu, &t, 8, Layout::RowMajor)
            .expect("q8_0");
        assert_eq!(
            got.kernel_id, "cpu.matvec.q8_0.x86_64",
            "a named arch outranks features"
        );
        let arm = Target::new("aarch64", &["avx2"]);
        let got = n
            .admit_for(Backend::Cpu, &arm, 8, Layout::RowMajor)
            .expect("q8_0");
        assert_eq!(
            got.kernel_id, "cpu.matvec.q8_0",
            "an x86_64 row served aarch64"
        );
        let e = pick("x86_64", &["avx2", "fma"], 14).expect_err("x86 got an sve kernel");
        assert!(e.to_string().contains("arch=x86_64"), "{e}");
    }

    #[test]
    fn the_same_arch_and_feature_set_is_a_duplicate_in_any_order() {
        let rows = [
            arch_row_json("cpu.matvec.a", "cpu", 12, "row_major", "x86_64", "avx2+fma"),
            arch_row_json("cpu.matvec.b", "cpu", 12, "row_major", "x86_64", "fma+avx2"),
        ];
        let e = Registry::parse(&doc(&rows))
            .err()
            .expect("duplicate indexed");
        assert!(e.contains("both claim"), "{e}");
    }

    /// AC-4: every embedded row carries a stable kernel_id that starts with its own backend.
    #[test]
    fn every_kernel_id_is_unique_and_names_its_backend() {
        let r = registry().expect("registry");
        let mut seen = std::collections::HashSet::new();
        for row in r.rows() {
            assert!(
                row.kernel_id.starts_with(&format!("{}.", row.backend)),
                "{}",
                row.kernel_id
            );
            assert!(
                seen.insert(row.kernel_id.as_str()),
                "duplicate {}",
                row.kernel_id
            );
        }
    }

    /// KREG coverage (CUDA): the registry's `cuda` rows and the ids `WeightQuantType` declares
    /// (one `gemv_dispatch` arm each — that match is exhaustive) are the same set, derived here
    /// over every id rather than from a hand-kept count. A variant with no row would be
    /// unreachable; a row with no variant would claim a kernel that does not exist.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_rows_equal_the_declared_weight_quant_types() {
        use crate::cuda::types::WeightQuantType;
        let r = registry().expect("registry");
        let mut declared = 0;
        for t in 0..=u32::try_from(MAX_TYPE_ID).expect("fits") {
            let has_arm = WeightQuantType::declared(t).is_some();
            let has_row = r.admit(Backend::Cuda, t, Layout::RowMajor).is_ok();
            assert_eq!(
                has_arm, has_row,
                "ggml type {t}: gemv arm={has_arm}, registry row={has_row}"
            );
            assert_eq!(
                WeightQuantType::from_ggml_type(t).is_some(),
                has_row,
                "ggml type {t}"
            );
            declared += usize::from(has_arm);
        }
        let rows = r.rows().iter().filter(|k| k.backend == "cuda").count();
        assert_eq!(declared, rows);
        assert!(rows > 0);
    }

    /// KREG gate (CUDA): removing a row makes that type unloadable on CUDA even though
    /// `WeightQuantType` still declares it — the registry is the authority, not the enum.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_cuda_type_with_no_row_is_not_loadable() {
        use crate::cuda::types::WeightQuantType;
        let only_q4k =
            Registry::parse(&doc(&[row_json("cuda.gemv.q4_k", "cuda", 12, "row_major")]))
                .expect("parses");
        assert!(WeightQuantType::declared(GGUF_TYPE_Q6_K).is_some());
        assert!(WeightQuantType::admitted_by(&only_q4k, GGUF_TYPE_Q6_K).is_none());
        assert!(WeightQuantType::admitted_by(&only_q4k, GGUF_TYPE_Q4_K).is_some());
    }

    /// FALSIFY-KREG-005: every row names a function that exists in its source file.
    #[test]
    fn falsify_kreg_005_every_row_names_a_real_fn() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for row in registry().expect("registry").rows() {
            let src = std::fs::read_to_string(root.join(&row.source_file))
                .unwrap_or_else(|e| panic!("{}: {}: {e}", row.kernel_id, row.source_file));
            let needle = format!("fn {}", row.source_fn);
            let found = src
                .match_indices(&needle)
                .any(|(i, _)| src[i + needle.len()..].starts_with(['(', '<']));
            assert!(
                found,
                "{}: `{needle}` not in {}",
                row.kernel_id, row.source_file
            );
        }
    }
}
