//! `apr devices [--json]` — backend discovery: probe → enumerate → print
//! (PP-066 R-0a, #2904, PMAT-989). Every kind of `{cpu, cuda, wgpu, metal,
//! hip}` is a line, Ready or `Unavailable(reason)`; the selection is printed
//! with its reason (REG-8); overrides are loud; discovery never fails the
//! process (REG-1). Resolution — refusing a `--backend` that is not Ready —
//! is R-0b (#3002).
//!
//! Overrides (each is printed when active):
//! * `APR_RESERVE_BYTES=<n|nK|nM|nG>` — the REG-7 reserve.
//! * `APR_REGISTRY_FIXTURE=<path.json>` — read the registry from a fixture
//!   (the CI case table and host dogfood); the output says `source=fixture(...)`.

use crate::error::{CliError, Result};
use trueno::registry::{default_factories, BackendRegistry};

/// Run `apr devices`.
///
/// # Errors
/// Only a malformed override (`APR_RESERVE_BYTES` that does not parse, an
/// unreadable or unparseable `APR_REGISTRY_FIXTURE`) is an error — exit code 4.
/// A machine with no GPU is not an error: it prints five lines and selects cpu.
pub fn run(json: bool) -> Result<()> {
    let reserve = reserve_override()?;
    let fixture = std::env::var("APR_REGISTRY_FIXTURE")
        .ok()
        .filter(|p| !p.is_empty());
    let registry = match &fixture {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| CliError::InvalidInput(format!("APR_REGISTRY_FIXTURE {path}: {e}")))?;
            let reg =
                BackendRegistry::from_fixture_json(&text, path).map_err(CliError::InvalidInput)?;
            match reserve {
                Some(r) => reg.with_reserve(r, "APR_RESERVE_BYTES override"),
                None => reg,
            }
        }
        None => BackendRegistry::discover_with(&default_factories(), reserve),
    };
    if json {
        println!("{}", registry.to_json().map_err(CliError::InvalidInput)?);
        return Ok(());
    }
    // REG-8: overrides are loud — and printed from what the REGISTRY holds, not
    // from the request (review quorum 2026-09-06, lanes 2 and 3).
    if reserve.is_some() {
        println!(
            "override: APR_RESERVE_BYTES -> reserve_bytes={} basis={}",
            registry.reserve_bytes, registry.reserve_basis
        );
    }
    if fixture.is_some() {
        println!(
            "override: APR_REGISTRY_FIXTURE -> source={}",
            registry.source
        );
    }
    print!("{}", registry.render_block(env!("CARGO_PKG_VERSION")));
    Ok(())
}

fn reserve_override() -> Result<Option<u64>> {
    let Ok(raw) = std::env::var("APR_RESERVE_BYTES") else {
        return Ok(None);
    };
    if raw.trim().is_empty() {
        return Ok(None);
    }
    parse_bytes(&raw).map(Some).ok_or_else(|| {
        CliError::InvalidInput(format!(
            "APR_RESERVE_BYTES={raw}: expected <n>, <n>K, <n>M or <n>G"
        ))
    })
}

/// `123`, `512M`, `999G` (binary multiples).
pub(crate) fn parse_bytes(raw: &str) -> Option<u64> {
    let s = raw.trim();
    let (num, mul) = match s.chars().last()? {
        'k' | 'K' => (&s[..s.len() - 1], 1u64 << 10),
        'm' | 'M' => (&s[..s.len() - 1], 1u64 << 20),
        'g' | 'G' => (&s[..s.len() - 1], 1u64 << 30),
        _ => (s, 1u64),
    };
    num.trim().parse::<u64>().ok()?.checked_mul(mul)
}

#[cfg(test)]
mod tests {
    use super::parse_bytes;

    #[test]
    fn sizes_parse_with_binary_suffixes_and_refuse_garbage() {
        assert_eq!(parse_bytes("123"), Some(123));
        assert_eq!(parse_bytes("512M"), Some(512 << 20));
        assert_eq!(parse_bytes("999G"), Some(999 << 30));
        assert_eq!(parse_bytes("2k"), Some(2048));
        assert_eq!(parse_bytes("lots"), None);
        assert_eq!(parse_bytes(""), None);
    }
}
