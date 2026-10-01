//! The ONE door through which `apr-cli` reads `setfit-apr-v1` bytes off disk.
//!
//! # The rule, in one sentence
//!
//! **No code in `apr-cli` may call `fs::read` on an artifact path.** Every surface
//! that needs artifact bytes — 04-07's `predict` / `inspect` / `eval`, 04-08's
//! `serve` startup, and anything after them — calls
//! [`read_setfit_apr_file_bounded`] instead.
//!
//! # Why a shared door rather than three careful callers
//!
//! Review finding B5 was that the artifact size cap ran only AFTER the file had
//! already been read into memory, so a hostile 40 GB file exhausted memory before
//! the check that exists to prevent exactly that could fire. `aprender-core` fixed
//! the library half by exposing
//! [`read_setfit_apr_bytes_bounded`](aprender::setfit::read_setfit_apr_bytes_bounded),
//! whose doc comment states that every filesystem and stream adapter is required to
//! call it. This module is the CLI half: three adapters each writing their own
//! `fs::metadata` + `File::open` + bounded-read sequence is three places for the
//! bound to be forgotten, and the one that forgets it looks identical to the two
//! that did not.
//!
//! # Two checks, in this order, and the order is the point
//!
//! 1. `fs::metadata` — a missing path, a directory or a non-regular file is refused
//!    typed, here, with the path named. This also produces the DECLARED LENGTH.
//! 2. The declared length is handed to the library reader, which refuses an
//!    over-cap length **without touching the file**, and then reads through
//!    `take(cap + 1)` anyway so a length that lies (a FIFO, a growing file, a
//!    filesystem reporting zero) still cannot exhaust memory.
//!
//! Passing `None` for the declared length would compile and would still be bounded
//! by check (2) — and would read MAX_ARTIFACT_BYTES of a hostile file before refusing it.
//! Passing the stat'd length is what makes the refusal free.

use std::fs;
use std::path::Path;

use aprender::setfit::{read_setfit_apr_bytes_bounded, SetFitArtifactError, MAX_ARTIFACT_BYTES};

use crate::error::CliError;

/// Appended to a refusal so the message names something the operator can do.
///
/// The library cannot name a CLI flag or a sibling command, and an error that
/// states a constraint without naming the knob that satisfies it is diagnosable but
/// not actionable — the same reasoning `data_contrastive::pair_config_error` applies
/// to the pair budget.
const ARTIFACT_REMEDY: &str = "Artifacts are produced by `apr setfit train --output <FILE>`; \
     verify the path points at that file and not at a directory, a partial download or a \
     plain APR.";

/// Read a `setfit-apr-v1` artifact from disk, bounded before the allocation.
///
/// This is the ONLY sanctioned crossing from the filesystem into `apr-cli`'s memory
/// for artifact bytes. See the module documentation for the rule and for why the
/// declared length is passed rather than `None`.
///
/// # Errors
///
/// [`CliError::FileNotFound`] for an absent path; [`CliError::NotAFile`] for a
/// directory or any other non-regular file; [`CliError::InvalidFormat`] when the
/// file is larger than the contracted cap (naming which of the library's two length
/// checks fired); [`CliError::ModelLoadFailed`] for a read failure or any other
/// typed library refusal; [`CliError::Io`] for a stat failure that is neither of the
/// above.
pub(crate) fn read_setfit_apr_file_bounded(path: &Path) -> Result<Vec<u8>, CliError> {
    // (1) STAT FIRST. Three things come out of this call and all three are load
    //     bearing: the existence check, the file-type check, and the declared length
    //     that makes step (2)'s refusal free.
    let metadata = fs::metadata(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            CliError::FileNotFound(path.to_path_buf())
        } else {
            CliError::Io(error)
        }
    })?;
    if !metadata.is_file() {
        // A directory read would fail later with an opaque OS error, and a FIFO would
        // block forever while `is_file()` is false for both. Refusing here names the
        // path and the reason.
        return Err(CliError::NotAFile(path.to_path_buf()));
    }

    // (2) THE BOUNDED READ, through the library's door. `File::open` rather than
    //     `fs::read`: the reader is handed a stream so the cap applies to the read
    //     itself, which is the whole of review B5's finding.
    let file = fs::File::open(path).map_err(CliError::Io)?;
    read_setfit_apr_bytes_bounded(file, Some(metadata.len()))
        .map_err(|error| artifact_read_error(path, &error))
}

/// Map a typed library refusal onto the CLI surface without losing the diagnosis.
///
/// The library's messages already name the check that fired and the values it
/// observed; this adds the path and the remedy, and chooses the exit code.
fn artifact_read_error(path: &Path, error: &SetFitArtifactError) -> CliError {
    match error {
        SetFitArtifactError::ArtifactTooLarge {
            what,
            observed,
            cap,
        } => {
            // InvalidFormat (exit 4) rather than ModelLoadFailed: nothing failed to
            // load, the input was refused for being outside the format's contracted
            // bounds. `what` distinguishes "the declared length was over cap, so no
            // byte was read" from "the declared length lied and the stream was".
            CliError::InvalidFormat(format!(
                "{}: artifact is larger than the contracted cap — {what} observed {observed} \
                 bytes against a cap of {cap} ({MAX_ARTIFACT_BYTES}). {ARTIFACT_REMEDY}",
                path.display()
            ))
        }
        SetFitArtifactError::ArtifactRead { reason } => CliError::ModelLoadFailed(format!(
            "{}: could not read the artifact — {reason}. {ARTIFACT_REMEDY}",
            path.display()
        )),
        // `SetFitArtifactError` is `#[non_exhaustive]` and the loader's rungs land in
        // the same enum, so this arm is reachable by a library that grows a variant
        // this build predates. Forwarding the library's own `Debug` rendering is
        // strictly better than a sentence this module invents about a failure it does
        // not know: the operator gets the variant name and its fields either way.
        other => CliError::ModelLoadFailed(format!(
            "{}: the artifact was refused — {other:?}. {ARTIFACT_REMEDY}",
            path.display()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    /// This module's own source, for the source assertions below.
    const SETFIT_IO_SOURCE: &str = include_str!("setfit_io.rs");

    /// Assemble a search needle from fragments at RUNTIME.
    ///
    /// These scans read the file they live in. A whole literal needle would appear IN
    /// the source being scanned and the count would come back one too high — the
    /// self-match hazard 04-14 committed once inside the very helper that warns about
    /// it. The discipline extends to doc comments, which is why this one describes the
    /// needles without spelling any of them.
    fn needle(fragments: &[&str]) -> String {
        fragments.concat()
    }

    fn write_file(dir: &Path, name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = dir.join(name);
        let mut file = fs::File::create(&path).expect("fixture file is creatable");
        file.write_all(bytes).expect("fixture file is writable");
        file.sync_all().expect("fixture file syncs");
        path
    }

    #[test]
    fn setfit_io_returns_the_file_bytes_verbatim() {
        let temp = TempDir::new().expect("tempdir");
        // Not a real artifact: this door's job is to produce BYTES bounded by the cap.
        // Deciding whether those bytes are a valid artifact belongs to the loader, and
        // a reader that also validated would be a second parse-ladder implementation.
        let payload: Vec<u8> = (0..=255_u8).cycle().take(4096).collect();
        let path = write_file(temp.path(), "artifact.apr", &payload);

        let read = read_setfit_apr_file_bounded(&path).expect("a small regular file is readable");
        assert_eq!(read, payload, "the door must not transform the bytes");
    }

    #[test]
    fn setfit_io_refuses_an_over_cap_file_from_its_declared_length() {
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("huge.apr");
        let file = fs::File::create(&path).expect("fixture file is creatable");
        // `set_len` past the cap without writing a byte: a sparse file on every
        // filesystem this repo builds on, so the test costs no disk and no time. It is
        // also the honest shape of the threat — an attacker does not have to spend
        // MAX_ARTIFACT_BYTES to make a reader spend it.
        file.set_len(MAX_ARTIFACT_BYTES + 1)
            .expect("a sparse over-cap file is creatable");
        drop(file);

        let error = read_setfit_apr_file_bounded(&path)
            .expect_err("a file past the contracted cap must be refused");
        let rendered = error.to_string();
        assert!(
            rendered.contains("declared_length"),
            "the refusal must name the check that fired, so a reader can tell that NO \
             byte was read; got: {rendered}"
        );
        assert!(
            rendered.contains(&(MAX_ARTIFACT_BYTES + 1).to_string()),
            "the refusal must name the observed length; got: {rendered}"
        );
    }

    #[test]
    fn setfit_io_refuses_a_directory_as_not_a_file() {
        let temp = TempDir::new().expect("tempdir");
        let error =
            read_setfit_apr_file_bounded(temp.path()).expect_err("a directory is not an artifact");
        assert!(
            matches!(error, CliError::NotAFile(_)),
            "a directory must be NotAFile, not an opaque IO error; got: {error}"
        );
    }

    #[test]
    fn setfit_io_reports_an_absent_path_as_file_not_found() {
        let temp = TempDir::new().expect("tempdir");
        let error = read_setfit_apr_file_bounded(&temp.path().join("absent.apr"))
            .expect_err("an absent path has no bytes");
        assert!(
            matches!(error, CliError::FileNotFound(_)),
            "an absent path must be FileNotFound (exit 3); got: {error}"
        );
    }

    #[test]
    fn setfit_io_reads_through_the_library_door_and_passes_the_statted_length() {
        // The two halves of review B5's fix, asserted on the source because neither is
        // observable from the outside: a caller cannot tell whether an in-bounds file
        // was read through the bounded door or through `fs::read`.
        let call = needle(&["read_setfit_apr_bytes_", "bounded(file, Some(metadata."]);
        assert!(
            SETFIT_IO_SOURCE.contains(&call),
            "the read must go through the library's bounded door AND pass the stat'd \
             length; passing None would still be bounded but would read the whole cap \
             before refusing"
        );

        // And the ban that makes this module THE door rather than merely A door.
        let banned = needle(&["fs::", "read("]);
        assert_eq!(
            SETFIT_IO_SOURCE.matches(&banned).count(),
            0,
            "this module must not itself contain the unbounded whole-file read it \
             exists to replace"
        );
    }
}
