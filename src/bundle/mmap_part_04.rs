
// ============================================================================
// Property Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    proptest! {
        #[test]
        fn prop_mapped_file_slice_within_bounds(
            data in prop::collection::vec(any::<u8>(), 1..1000),
            start in 0usize..1000,
            len in 0usize..500,
        ) {
            let mut file = NamedTempFile::new().expect("create temp");
            file.write_all(&data).expect("write");

            let mapped = MappedFile::open(file.path()).expect("open");
            let end = start.saturating_add(len);

            if start <= end && end <= data.len() {
                prop_assert_eq!(mapped.slice(start, end), Some(&data[start..end]));
            } else {
                prop_assert!(mapped.slice(start, end).is_none());
            }
        }

        #[test]
        fn prop_mapped_file_full_slice_equals_as_slice(
            data in prop::collection::vec(any::<u8>(), 0..1000),
        ) {
            let mut file = NamedTempFile::new().expect("create temp");
            file.write_all(&data).expect("write");

            let mapped = MappedFile::open(file.path()).expect("open");

            prop_assert_eq!(mapped.slice(0, mapped.len()), Some(mapped.as_slice()));
        }

        #[test]
        fn prop_mapped_file_len_matches_data(
            data in prop::collection::vec(any::<u8>(), 0..10000),
        ) {
            let mut file = NamedTempFile::new().expect("create temp");
            file.write_all(&data).expect("write");

            let mapped = MappedFile::open(file.path()).expect("open");

            prop_assert_eq!(mapped.len(), data.len());
            prop_assert_eq!(mapped.is_empty(), data.is_empty());
        }
    }
}
