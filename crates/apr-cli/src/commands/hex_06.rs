
// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
include!("hex_options_detect.rs");
include!("make.rs");
include!("tests_blocks_and_slices.rs");

    /// FALSIFY-OPS-012: `hex --json` on an APR file lists each filtered tensor,
    /// with stats only when asked, and silently skips names that are not in the file.
    #[test]
    fn output_json_v2_lists_filtered_tensors_with_and_without_stats() {
        use aprender::format::v2::{AprV2Metadata, AprV2Writer};

        let mut writer = AprV2Writer::new(AprV2Metadata::new("hex-json"));
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        writer.add_tensor("a.weight", TensorDType::F32, vec![2, 2], data);
        let mut bytes = Vec::new();
        writer.write_to(&mut bytes).expect("write APR");
        let reader = AprV2Reader::from_bytes(&bytes).expect("read APR");

        assert!(output_json_v2(&reader, &["a.weight"], 2, true).is_ok());
        assert!(output_json_v2(&reader, &["a.weight"], 0, false).is_ok());
        assert!(output_json_v2(&reader, &["missing"], 4, true).is_ok());
        assert!(output_json_v2(&reader, &[], 4, false).is_ok());
        // the statistics the stats branch serialises
        let (min, max, mean, _std) = compute_stats(&[1.0, 2.0, 3.0, 4.0]);
        assert!((min - 1.0).abs() < f32::EPSILON && (max - 4.0).abs() < f32::EPSILON);
        assert!((mean - 2.5).abs() < 1e-6);
    }
}
