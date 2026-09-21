//! Bounded reads of a model file's head (#3750, #3761): the ONE policy (first read + cap) and
//! the APR v2 header reader come from apr-format; the SafeTensors header reader lives here,
//! beside the SafeTensors format.

pub use apr_format::prefix::*;

use std::path::Path;

/// The bytes of a SafeTensors file before its tensor data: the 8-byte little-endian header
/// length and the JSON header it counts. Bounded by [`HEADER_READ_CAP`].
pub fn safetensors_header_prefix(path: &Path) -> Result<Vec<u8>, String> {
    let head = read_prefix(path, 8).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let len_bytes: [u8; 8] = head.as_slice().try_into().map_err(|_| {
        format!(
            "{} is shorter than a SafeTensors length prefix",
            path.display()
        )
    })?;
    let n = usize::try_from(u64::from_le_bytes(len_bytes))
        .ok()
        .and_then(|h| h.checked_add(8))
        .filter(|&n| n <= HEADER_READ_CAP)
        .ok_or_else(|| {
            format!(
                "SafeTensors header length {} is past the {} MiB header cap; refused rather than read whole",
                u64::from_le_bytes(len_bytes),
                HEADER_READ_CAP >> 20
            )
        })?;
    read_prefix(path, n).map_err(|e| format!("cannot read {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn file_with(bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        f.write_all(bytes).expect("write");
        f
    }

    #[test]
    fn a_safetensors_prefix_is_the_length_and_the_json_only() {
        let json = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json);
        bytes.extend_from_slice(&[0u8; 4096]); // tensor data
        let f = file_with(&bytes);
        let prefix = safetensors_header_prefix(f.path()).expect("the header");
        assert_eq!(prefix.len(), 8 + json.len());
        assert_eq!(&prefix[8..], json);
    }

    #[test]
    fn a_safetensors_length_past_the_cap_is_refused() {
        let f = file_with(&u64::MAX.to_le_bytes());
        let e = safetensors_header_prefix(f.path()).expect_err("an absurd length");
        assert!(e.contains("refused rather than read whole"), "{e}");
    }
}
