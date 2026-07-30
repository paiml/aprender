// Tests for `apr pull --verify` (pull_verify.rs).

#[cfg(test)]
mod pull_verify_tests {
    use super::*;
    use std::collections::HashMap;

    fn write_manifest(dir: &Path, files: &[(&str, u64, &str)]) {
        let mut map = HashMap::new();
        for (name, size, hash) in files {
            map.insert(
                (*name).to_string(),
                crate::commands::pull::FileChecksum {
                    size: *size,
                    blake3: (*hash).to_string(),
                },
            );
        }
        let manifest = crate::commands::pull::ShardManifest {
            version: 1,
            repo: "test/repo".to_string(),
            files: map,
        };
        std::fs::write(
            dir.join(".apr-manifest.json"),
            serde_json::to_string(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");
    }

    fn blake3_of(bytes: &[u8]) -> String {
        blake3::hash(bytes).to_hex().to_string()
    }

    #[test]
    fn intact_file_verifies() {
        let dir = tempfile::tempdir().expect("tempdir");
        let body = b"model weights, entirely intact";
        std::fs::write(dir.path().join("shard.bin"), body).expect("write");
        write_manifest(
            dir.path(),
            &[("shard.bin", body.len() as u64, &blake3_of(body))],
        );
        let results = verify_manifest(&dir.path().join(".apr-manifest.json"), dir.path())
            .expect("verify runs");
        assert_eq!(results.len(), 1);
        assert!(results[0].1.is_ok(), "got {:?}", results[0].1);
        report(&results, dir.path()).expect("intact model must pass");
    }

    /// THE POINT OF THE COMMAND. Same length, different content - exactly the
    /// shape of the corrupt HF blob that motivated this (right size, tensors
    /// zeroed). The size-only check in validate.rs accepts this file.
    #[test]
    fn same_size_different_content_is_caught() {
        let dir = tempfile::tempdir().expect("tempdir");
        let good = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let corrupt = b"AAAAAAAAAAAAAAAA\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
        assert_eq!(good.len(), corrupt.len(), "the premise is equal length");

        std::fs::write(dir.path().join("shard.bin"), corrupt).expect("write");
        // Manifest records the GOOD hash and the (identical) size.
        write_manifest(
            dir.path(),
            &[("shard.bin", good.len() as u64, &blake3_of(good))],
        );

        let results = verify_manifest(&dir.path().join(".apr-manifest.json"), dir.path())
            .expect("verify runs");
        match &results[0].1 {
            FileVerdict::HashMismatch { .. } => {}
            other => panic!("expected HashMismatch, got {other:?}"),
        }
        let err = report(&results, dir.path()).expect_err("corrupt content MUST fail");
        assert!(
            format!("{err}").contains("failed verification"),
            "message should be actionable, got: {err}"
        );
    }

    /// A size check alone would still catch truncation; keep that path working
    /// and reported distinctly, because the remedy is the same but the
    /// diagnosis differs.
    #[test]
    fn truncated_file_reports_size_not_hash() {
        let dir = tempfile::tempdir().expect("tempdir");
        let good = b"0123456789abcdef";
        std::fs::write(dir.path().join("shard.bin"), b"0123").expect("write");
        write_manifest(
            dir.path(),
            &[("shard.bin", good.len() as u64, &blake3_of(good))],
        );
        let results = verify_manifest(&dir.path().join(".apr-manifest.json"), dir.path())
            .expect("verify runs");
        match &results[0].1 {
            FileVerdict::SizeMismatch { expected, actual } => {
                assert_eq!(*expected, 16);
                assert_eq!(*actual, 4);
            }
            other => panic!("expected SizeMismatch, got {other:?}"),
        }
        report(&results, dir.path()).expect_err("truncated file must fail");
    }

    #[test]
    fn missing_file_is_reported_not_silently_skipped() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_manifest(dir.path(), &[("absent.bin", 10, &blake3_of(b"whatever"))]);
        let results = verify_manifest(&dir.path().join(".apr-manifest.json"), dir.path())
            .expect("verify runs");
        assert_eq!(results[0].1, FileVerdict::Missing);
        report(&results, dir.path()).expect_err("a missing shard must fail");
    }

    /// Fail-closed: a manifest naming zero files verified nothing, and must not
    /// print success. Same defect class the command exists to expose.
    #[test]
    fn empty_manifest_fails_rather_than_reporting_success() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_manifest(dir.path(), &[]);
        let results = verify_manifest(&dir.path().join(".apr-manifest.json"), dir.path())
            .expect("verify runs");
        assert!(results.is_empty());
        let err = report(&results, dir.path()).expect_err("empty manifest must NOT pass");
        assert!(
            format!("{err}").contains("ZERO files"),
            "got: {err}"
        );
    }

    #[test]
    fn absent_manifest_is_an_error_not_a_pass() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = run_verify(dir.path()).expect_err("no manifest must not report success");
        assert!(
            format!("{err}").contains("No .apr-manifest.json"),
            "got: {err}"
        );
    }

    /// Verdicts are returned sorted, so output and assertions do not depend on
    /// HashMap iteration order.
    #[test]
    fn results_are_deterministically_ordered() {
        let dir = tempfile::tempdir().expect("tempdir");
        for name in ["c.bin", "a.bin", "b.bin"] {
            std::fs::write(dir.path().join(name), name.as_bytes()).expect("write");
        }
        write_manifest(
            dir.path(),
            &[
                ("c.bin", 5, &blake3_of(b"c.bin")),
                ("a.bin", 5, &blake3_of(b"a.bin")),
                ("b.bin", 5, &blake3_of(b"b.bin")),
            ],
        );
        let results = verify_manifest(&dir.path().join(".apr-manifest.json"), dir.path())
            .expect("verify runs");
        let names: Vec<&str> = results.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["a.bin", "b.bin", "c.bin"]);
        assert!(results.iter().all(|(_, v)| v.is_ok()));
    }

    /// Streamed hashing must agree with one-shot hashing across the 1 MiB
    /// buffer boundary - otherwise large shards would false-positive.
    #[test]
    fn streamed_hash_matches_oneshot_across_buffer_boundary() {
        let dir = tempfile::tempdir().expect("tempdir");
        let body = vec![0xA5u8; (1 << 20) + 12345];
        let p = dir.path().join("big.bin");
        std::fs::write(&p, &body).expect("write");
        assert_eq!(hash_file(&p).expect("hash"), blake3_of(&body));
    }
}
