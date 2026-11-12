//! Test artifact: the aprender release pipeline publishes this crate nightly to rehearse its own
//! publish path (clean-room read, upload, index read-back). Not for use.
#![forbid(unsafe_code)]

/// The name this crate is published under; the rehearsal reads the sparse index for it.
#[must_use]
pub fn crate_name() -> &'static str {
    "aprender-publish-rehearsal"
}

#[cfg(test)]
mod tests {
    #[test]
    fn crate_name_is_the_package_name() {
        assert_eq!(super::crate_name(), env!("CARGO_PKG_NAME"));
    }
}
