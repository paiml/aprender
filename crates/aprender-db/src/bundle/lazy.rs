//! Lazy tables: fetch a bundle part on first use, verify it, cache it.
//!
//! The fetch function is supplied by the caller (a URL fetch in the browser, a
//! file read natively). It is called at most once per successful table: a
//! verification failure is returned to the caller and is not cached, so the
//! next `get` fetches again.

use std::collections::btree_map::Entry;
use std::collections::BTreeMap;

use super::manifest::{Manifest, PartKind};
use super::BundleError;

/// Lazily loaded, verified, cached view of a bundle's lazy parts.
pub struct LazyTables<F> {
    manifest: Manifest,
    fetch: F,
    cache: BTreeMap<String, Vec<u8>>,
    fetches: BTreeMap<String, u32>,
}

impl<F> LazyTables<F>
where
    F: FnMut(&str) -> Result<Vec<u8>, BundleError>,
{
    /// Wrap `manifest` and the fetch function. Nothing is fetched yet.
    pub fn new(manifest: Manifest, fetch: F) -> Self {
        Self { manifest, fetch, cache: BTreeMap::new(), fetches: BTreeMap::new() }
    }

    /// The verified bytes of lazy part `name`.
    ///
    /// The first call fetches, verifies and caches. Later calls return the
    /// cached bytes without fetching.
    ///
    /// # Errors
    ///
    /// [`BundleError::UnknownPart`] for a name the manifest lacks,
    /// [`BundleError::EagerPart`] for an eager part, any error from the fetch
    /// function, or a length or hash mismatch from verification. A failed
    /// `get` caches nothing.
    pub fn get(&mut self, name: &str) -> Result<&[u8], BundleError> {
        let Self { manifest, fetch, cache, fetches } = self;
        match cache.entry(name.to_owned()) {
            Entry::Occupied(slot) => Ok(slot.into_mut().as_slice()),
            Entry::Vacant(slot) => {
                let bytes = load(manifest, fetch, fetches, name)?;
                Ok(slot.insert(bytes).as_slice())
            }
        }
    }

    /// How many times the fetch function has been called for `name`.
    #[must_use]
    pub fn fetch_count(&self, name: &str) -> u32 {
        self.fetches.get(name).copied().unwrap_or(0)
    }
}

/// Fetch and verify one lazy part. Counts the request before calling `fetch`,
/// so a failed fetch or verification still counts as one request.
fn load<F>(
    manifest: &Manifest,
    fetch: &mut F,
    fetches: &mut BTreeMap<String, u32>,
    name: &str,
) -> Result<Vec<u8>, BundleError>
where
    F: FnMut(&str) -> Result<Vec<u8>, BundleError>,
{
    let kind = manifest
        .entry(name)
        .ok_or_else(|| BundleError::UnknownPart { part: name.to_owned() })?
        .kind;
    if kind == PartKind::Eager {
        return Err(BundleError::EagerPart { part: name.to_owned() });
    }
    *fetches.entry(name.to_owned()).or_insert(0) += 1;
    let bytes = fetch(name)?;
    manifest.verify_part(name, &bytes)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    type Log = Rc<RefCell<Vec<String>>>;

    fn manifest() -> Manifest {
        Manifest::from_parts([
            ("cells", PartKind::Lazy, &b"cell-rows"[..]),
            ("meta", PartKind::Eager, &b"eager"[..]),
            ("zones", PartKind::Lazy, &b"zone-rows"[..]),
        ])
        .expect("unique names")
    }

    fn payload(name: &str) -> &'static [u8] {
        match name {
            "cells" => b"cell-rows",
            "meta" => b"eager",
            "zones" => b"zone-rows",
            _ => b"",
        }
    }

    /// A fetch source that logs every request. When `corrupt` is set, the
    /// `zones` bytes come back with one flipped bit.
    fn source(
        log: Log,
        corrupt: Rc<Cell<bool>>,
    ) -> impl FnMut(&str) -> Result<Vec<u8>, BundleError> {
        move |name: &str| {
            log.borrow_mut().push(name.to_owned());
            let mut bytes = payload(name).to_vec();
            if corrupt.get() && name == "zones" {
                bytes[0] ^= 0x01;
            }
            Ok(bytes)
        }
    }

    #[test]
    fn nothing_is_requested_before_first_use() {
        let log: Log = Rc::default();
        let tables = LazyTables::new(manifest(), source(log.clone(), Rc::default()));
        assert!(log.borrow().is_empty());
        assert_eq!(tables.fetch_count("zones"), 0);
    }

    #[test]
    fn first_use_fetches_once_and_later_uses_hit_the_cache() {
        let log: Log = Rc::default();
        let mut tables = LazyTables::new(manifest(), source(log.clone(), Rc::default()));
        assert_eq!(tables.get("zones").expect("verified"), b"zone-rows");
        assert_eq!(tables.get("zones").expect("cached"), b"zone-rows");
        assert_eq!(*log.borrow(), vec!["zones".to_owned()]);
        assert_eq!(tables.fetch_count("zones"), 1);
        assert_eq!(tables.fetch_count("cells"), 0);
    }

    #[test]
    fn verification_failure_is_refused_and_not_cached() {
        let log: Log = Rc::default();
        let corrupt = Rc::new(Cell::new(true));
        let mut tables = LazyTables::new(manifest(), source(log.clone(), corrupt.clone()));
        let err = tables.get("zones").expect_err("flipped byte");
        assert_eq!(err, BundleError::HashMismatch { part: "zones".to_owned() });
        assert_eq!(tables.fetch_count("zones"), 1);

        corrupt.set(false);
        assert_eq!(tables.get("zones").expect("clean retry"), b"zone-rows");
        assert_eq!(tables.fetch_count("zones"), 2);
        assert_eq!(log.borrow().len(), 2);
    }

    #[test]
    fn length_mismatch_is_refused_and_named() {
        let mut tables = LazyTables::new(manifest(), |_: &str| -> Result<Vec<u8>, BundleError> {
            Ok(b"short".to_vec())
        });
        assert_eq!(
            tables.get("cells").expect_err("truncated"),
            BundleError::LengthMismatch { part: "cells".to_owned(), expected: 9, actual: 5 }
        );
    }

    #[test]
    fn eager_part_is_rejected_without_a_fetch() {
        let log: Log = Rc::default();
        let mut tables = LazyTables::new(manifest(), source(log.clone(), Rc::default()));
        assert_eq!(
            tables.get("meta").expect_err("eager"),
            BundleError::EagerPart { part: "meta".to_owned() }
        );
        assert!(log.borrow().is_empty());
        assert_eq!(tables.fetch_count("meta"), 0);
    }

    #[test]
    fn unknown_part_is_rejected_without_a_fetch() {
        let log: Log = Rc::default();
        let mut tables = LazyTables::new(manifest(), source(log.clone(), Rc::default()));
        assert_eq!(
            tables.get("absent").expect_err("unknown"),
            BundleError::UnknownPart { part: "absent".to_owned() }
        );
        assert!(log.borrow().is_empty());
    }

    #[test]
    fn fetch_error_propagates_and_is_not_cached() {
        let calls = Rc::new(Cell::new(0_u32));
        let counter = calls.clone();
        let mut tables =
            LazyTables::new(manifest(), move |name: &str| -> Result<Vec<u8>, BundleError> {
                counter.set(counter.get() + 1);
                Err(BundleError::UnknownPart { part: name.to_owned() })
            });
        assert!(matches!(tables.get("zones"), Err(BundleError::UnknownPart { .. })));
        assert!(tables.get("zones").is_err());
        assert_eq!(calls.get(), 2);
        assert_eq!(tables.fetch_count("zones"), 2);
    }
}
