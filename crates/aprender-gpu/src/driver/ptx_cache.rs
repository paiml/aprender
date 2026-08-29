//! PTX Disk Cache — SHA-256 keyed cubin persistence
//!
//! Caches compiled cubin blobs to `~/.cache/trueno/ptx/{sha256}.cubin`
//! to eliminate the ~35-minute JIT warmup on training restarts (~60 kernels).
//!
//! # Design
//!
//! Cache key = SHA-256(patched PTX + compute capability + driver version).
//! This module is NOT feature-gated so its tests run without CUDA hardware.
//!
//! # Citation
//!
//! SHA-256 implementation follows NIST FIPS 180-4 (2015).

use std::os::raw::c_uint;
use std::path::PathBuf;

// ============================================================================
// SHA-256 (minimal, dependency-free implementation for cache keys)
// ============================================================================

/// SHA-256 initial hash values (first 32 bits of fractional parts of sqrt of first 8 primes)
const SHA256_H: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

/// SHA-256 round constants (first 32 bits of fractional parts of cube roots of first 64 primes)
const SHA256_K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// Compute SHA-256 hash of input bytes. Returns 32-byte digest.
///
/// Minimal, dependency-free implementation for PTX cache key generation.
/// Not optimized for throughput -- called once per kernel load, correctness is paramount.
pub(crate) fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = SHA256_H;

    // Pre-processing: pad message to multiple of 512 bits (64 bytes)
    let bit_len = (data.len() as u64) * 8;
    let mut padded = data.to_vec();
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit (64-byte) block
    for block in padded.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[4 * i],
                block[4 * i + 1],
                block[4 * i + 2],
                block[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut digest = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        digest[4 * i..4 * i + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

/// Format SHA-256 digest as lowercase hex string.
pub(crate) fn hex_digest(digest: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for byte in digest {
        s.push_str(&format!("{:02x}", byte));
    }
    s
}

// ============================================================================
// PTX Disk Cache Utilities
// ============================================================================

/// Compute cache key for a PTX compilation.
///
/// Key components:
/// - Patched PTX source (post `patch_backward_branches_sm121`)
/// - Device compute capability (e.g., "sm_90")
/// - CUDA driver version (e.g., 12030)
///
/// Returns a hex SHA-256 hash string (64 characters).
pub(crate) fn ptx_cache_key(ptx: &str, jit_target: c_uint, driver_version: i32) -> String {
    // Build a canonical input string: PTX + separator + metadata
    // The separator uses a byte sequence unlikely to appear in PTX
    let mut input = ptx.as_bytes().to_vec();
    input.extend_from_slice(b"\x00\x01TRUENO_CACHE_KEY\x01\x00");
    input.extend_from_slice(format!("jit_target={jit_target}").as_bytes());
    input.push(b'\0');
    input.extend_from_slice(format!("driver_version={driver_version}").as_bytes());
    hex_digest(&sha256(&input))
}

/// Get the PTX cache directory path: `$HOME/.cache/trueno/ptx/`
pub(crate) fn ptx_cache_dir() -> Option<PathBuf> {
    // Use $HOME directly (no external deps like `dirs` crate)
    let home = std::env::var("HOME").ok()?;
    Some(
        PathBuf::from(home)
            .join(".cache")
            .join("trueno")
            .join("ptx"),
    )
}

/// Try to load a cached cubin from disk.
///
/// Returns the raw cubin bytes if the cache file exists and is readable.
pub(crate) fn load_cached_cubin(cache_key: &str) -> Option<Vec<u8>> {
    let dir = ptx_cache_dir()?;
    let path = dir.join(format!("{cache_key}.cubin"));
    std::fs::read(&path).ok()
}

/// Save a compiled cubin to the disk cache.
///
/// Creates the cache directory if it doesn't exist.
/// Uses atomic write (write to .tmp then rename) to prevent partial reads.
/// Failures are silently ignored -- caching is best-effort.
pub(crate) fn save_cached_cubin(cache_key: &str, cubin: &[u8]) {
    let Some(dir) = ptx_cache_dir() else { return };
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let path = dir.join(format!("{cache_key}.cubin"));
    let tmp_path = dir.join(format!("{cache_key}.cubin.tmp"));
    if std::fs::write(&tmp_path, cubin).is_ok() {
        let _ = std::fs::rename(&tmp_path, &path);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // SHA-256 correctness tests (NIST FIPS 180-4 test vectors)
    // ========================================================================

    #[test]
    fn test_sha256_empty() {
        // NIST FIPS 180-4 test vector: SHA-256("")
        let digest = sha256(b"");
        assert_eq!(
            hex_digest(&digest),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_sha256_abc() {
        // NIST FIPS 180-4 test vector: SHA-256("abc")
        let digest = sha256(b"abc");
        assert_eq!(
            hex_digest(&digest),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn test_sha256_448_bits() {
        // NIST FIPS 180-4 test vector
        let input = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        let digest = sha256(input);
        assert_eq!(
            hex_digest(&digest),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn test_sha256_long_message() {
        // NIST FIPS 180-4 test vector: SHA-256 of 896-bit message
        let input = b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
        let digest = sha256(input);
        assert_eq!(
            hex_digest(&digest),
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"
        );
    }

    #[test]
    fn test_sha256_deterministic() {
        // Same input must produce same hash (cache key determinism)
        let ptx = ".version 8.0\n.target sm_90\n.entry test() { ret; }";
        let d1 = sha256(ptx.as_bytes());
        let d2 = sha256(ptx.as_bytes());
        assert_eq!(d1, d2);
    }

    // ========================================================================
    // Cache key tests
    // ========================================================================

    #[test]
    fn test_cache_key_deterministic() {
        let ptx = ".version 8.0\n.target sm_90";
        let k1 = ptx_cache_key(ptx, 90, 12030);
        let k2 = ptx_cache_key(ptx, 90, 12030);
        assert_eq!(k1, k2);
        assert_eq!(k1.len(), 64); // SHA-256 hex = 64 chars
    }

    #[test]
    fn test_cache_key_different_ptx() {
        let k1 = ptx_cache_key("ptx_v1", 90, 12030);
        let k2 = ptx_cache_key("ptx_v2", 90, 12030);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_different_target() {
        let ptx = ".version 8.0";
        let k1 = ptx_cache_key(ptx, 89, 12030);
        let k2 = ptx_cache_key(ptx, 90, 12030);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_different_driver() {
        let ptx = ".version 8.0";
        let k1 = ptx_cache_key(ptx, 90, 12030);
        let k2 = ptx_cache_key(ptx, 90, 13000);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_hex_format() {
        let key = ptx_cache_key("test", 90, 12000);
        assert_eq!(key.len(), 64);
        assert!(key.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // ========================================================================
    // Cache file I/O tests
    // ========================================================================

    #[test]
    fn test_cache_roundtrip() {
        // Test save -> load roundtrip
        let key = "test_ptx_cache_roundtrip_deadbeef";
        let data = b"fake cubin data for test";

        save_cached_cubin(key, data);

        let loaded = load_cached_cubin(key);
        assert_eq!(loaded.as_deref(), Some(data.as_slice()));

        // Cleanup
        if let Some(dir) = ptx_cache_dir() {
            let _ = std::fs::remove_file(dir.join(format!("{key}.cubin")));
        }
    }

    #[test]
    fn test_cache_miss() {
        // Non-existent key should return None
        let result = load_cached_cubin("nonexistent_key_that_doesnt_exist_12345");
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_overwrite() {
        let key = "test_ptx_cache_overwrite";
        save_cached_cubin(key, b"version_1");
        save_cached_cubin(key, b"version_2");

        let loaded = load_cached_cubin(key);
        assert_eq!(loaded.as_deref(), Some(b"version_2".as_slice()));

        // Cleanup
        if let Some(dir) = ptx_cache_dir() {
            let _ = std::fs::remove_file(dir.join(format!("{key}.cubin")));
        }
    }

    #[test]
    fn test_cache_empty_data() {
        let key = "test_ptx_cache_empty";
        save_cached_cubin(key, b"");

        let loaded = load_cached_cubin(key);
        assert_eq!(loaded.as_deref(), Some(b"".as_slice()));

        // Cleanup
        if let Some(dir) = ptx_cache_dir() {
            let _ = std::fs::remove_file(dir.join(format!("{key}.cubin")));
        }
    }

    #[test]
    fn test_hex_digest_format() {
        let digest = [0u8; 32];
        let hex = hex_digest(&digest);
        assert_eq!(hex.len(), 64);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(
            hex,
            "0000000000000000000000000000000000000000000000000000000000000000"
        );
    }

    #[test]
    fn test_hex_digest_ff() {
        let digest = [0xFF; 32];
        let hex = hex_digest(&digest);
        assert_eq!(
            hex,
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        );
    }

    #[test]
    fn test_ptx_cache_dir() {
        // Should return Some when $HOME is set
        let dir = ptx_cache_dir();
        if std::env::var("HOME").is_ok() {
            assert!(dir.is_some());
            let d = dir.unwrap();
            assert!(d.ends_with("trueno/ptx"));
        }
    }
}
