// ============================================================================
// Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ========================================================================
    // MappedFile Tests (New API)
    // ========================================================================

    #[test]
    fn test_mapped_file_open_and_read() {
        let mut file = NamedTempFile::new().expect("create temp");
        file.write_all(b"Hello, mmap!").expect("write");

        let mapped = MappedFile::open(file.path()).expect("open");
        assert_eq!(mapped.as_slice(), b"Hello, mmap!");
        assert_eq!(mapped.len(), 12);
        assert!(!mapped.is_empty());
    }

    #[test]
    fn test_mapped_file_slice() {
        let mut file = NamedTempFile::new().expect("create temp");
        file.write_all(b"0123456789").expect("write");

        let mapped = MappedFile::open(file.path()).expect("open");
        assert_eq!(mapped.slice(2, 7), Some(&b"23456"[..]));
        assert_eq!(mapped.slice(0, 3), Some(&b"012"[..]));
        assert_eq!(mapped.slice(8, 10), Some(&b"89"[..]));
    }

    #[test]
    fn test_mapped_file_slice_out_of_bounds() {
        let mut file = NamedTempFile::new().expect("create temp");
        file.write_all(b"short").expect("write");

        let mapped = MappedFile::open(file.path()).expect("open");
        assert!(mapped.slice(0, 100).is_none());
        assert!(mapped.slice(10, 20).is_none());
    }

    #[test]
    fn test_mapped_file_slice_invalid_range() {
        let mut file = NamedTempFile::new().expect("create temp");
        file.write_all(b"test data").expect("write");

        let mapped = MappedFile::open(file.path()).expect("open");
        // end < start should return None
        assert!(mapped.slice(5, 3).is_none());
    }

    #[test]
    fn test_mapped_file_empty() {
        let file = NamedTempFile::new().expect("create temp");
        let mapped = MappedFile::open(file.path()).expect("open");
        assert!(mapped.is_empty());
        assert_eq!(mapped.len(), 0);
        let empty: &[u8] = &[];
        assert_eq!(mapped.as_slice(), empty);
    }

    #[test]
    fn test_mapped_file_path() {
        let file = NamedTempFile::new().expect("create temp");
        let expected_path = file.path().to_string_lossy().to_string();
        let mapped = MappedFile::open(file.path()).expect("open");
        assert_eq!(mapped.path(), expected_path);
    }

    #[test]
    fn test_mapped_file_large_file() {
        let mut file = NamedTempFile::new().expect("create temp");
        let data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        file.write_all(&data).expect("write");

        let mapped = MappedFile::open(file.path()).expect("open");
        assert_eq!(mapped.len(), 100_000);
        assert_eq!(mapped.slice(50_000, 50_010), Some(&data[50_000..50_010]));
    }

    #[test]
    fn test_mapped_file_nonexistent() {
        let result = MappedFile::open("/nonexistent/path/file.bin");
        assert!(result.is_err());
    }

    // ========================================================================
    // MappedRegion Tests (Legacy API)
    // ========================================================================

    #[test]
    fn test_mapped_region_new() {
        let data = vec![1, 2, 3, 4, 5];
        let region = MappedRegion::new(data.clone(), 100);

        assert_eq!(region.as_slice(), &data);
        assert_eq!(region.offset(), 100);
        assert_eq!(region.len(), 5);
        assert!(!region.is_empty());
    }

    #[test]
    fn test_mapped_region_slice() {
        let data = vec![10, 20, 30, 40, 50];
        let region = MappedRegion::new(data, 0);

        assert_eq!(region.slice(1, 4), Some(&[20, 30, 40][..]));
        assert_eq!(region.slice(0, 2), Some(&[10, 20][..]));
        assert!(region.slice(3, 10).is_none());
    }

    // ========================================================================
    // MemoryMappedFile Tests (Legacy API)
    // ========================================================================

    #[test]
    fn test_memory_mapped_file_open() {
        let mut temp = NamedTempFile::new().expect("create temp");
        temp.write_all(b"test data").expect("write");

        let mmap = MemoryMappedFile::open(temp.path()).expect("open");
        assert_eq!(mmap.size(), 9);
    }

    #[test]
    fn test_memory_mapped_file_read_at() {
        let mut temp = NamedTempFile::new().expect("create temp");
        temp.write_all(b"0123456789").expect("write");

        let mut mmap = MemoryMappedFile::open(temp.path()).expect("open");
        let data = mmap.read_at(2, 5).expect("read");
        assert_eq!(data, b"23456");
    }

    #[test]
    fn test_memory_mapped_file_map_region() {
        let mut temp = NamedTempFile::new().expect("create temp");
        temp.write_all(b"ABCDEFGHIJ").expect("write");

        let mut mmap = MemoryMappedFile::open(temp.path()).expect("open");
        let region = mmap.map_region(3, 4).expect("map");
        assert_eq!(region.as_slice(), b"DEFG");
        assert_eq!(mmap.cached_regions(), 1);
        assert_eq!(mmap.cached_bytes(), 4);
    }

    // ========================================================================
    // PageTable Tests
    // ========================================================================

    #[test]
    fn test_page_entry() {
        let mut entry = PageEntry::new(1000, 256);
        assert_eq!(entry.offset, 1000);
        assert_eq!(entry.size, 256);
        assert_eq!(entry.access_count, 0);

        entry.touch(5);
        assert_eq!(entry.access_count, 1);
        assert_eq!(entry.last_access, 5);

        entry.touch(10);
        assert_eq!(entry.access_count, 2);
        assert_eq!(entry.last_access, 10);
    }

    #[test]
    fn test_page_table_lru() {
        let mut table = PageTable::new();

        table.add_page(100, 10);
        table.add_page(200, 20);
        table.add_page(300, 30);

        // 100 is oldest, should be LRU
        assert_eq!(table.lru_page(), Some(100));

        // Touch 100, now 200 should be LRU
        table.touch(100);
        assert_eq!(table.lru_page(), Some(200));
    }

    #[test]
    fn test_page_table_lfu() {
        let mut table = PageTable::new();

        table.add_page(100, 10);
        table.add_page(200, 20);
        table.add_page(300, 30);

        // Touch pages different numbers of times
        table.touch(100);
        table.touch(100);
        table.touch(200);

        // 300 has fewest accesses (1)
        assert_eq!(table.lfu_page(), Some(300));
    }

    #[test]
    fn test_page_table_remove() {
        let mut table = PageTable::new();
        table.add_page(100, 10);
        table.add_page(200, 20);

        assert_eq!(table.len(), 2);
        assert_eq!(table.total_size(), 30);

        let removed = table.remove(100);
        assert!(removed.is_some());
        assert_eq!(table.len(), 1);
        assert_eq!(table.total_size(), 20);
    }

    // ========================================================================
    // Additional Coverage Tests
    // ========================================================================

    #[test]
    fn test_mapped_file_debug() {
        let mut temp = NamedTempFile::new().expect("create temp");
        temp.write_all(b"test").expect("write");

        let mapped = MappedFile::open(temp.path()).expect("open");
        let debug_str = format!("{:?}", mapped);
        assert!(debug_str.contains("MappedFile"));
    }

    #[test]
    fn test_mapped_region_empty() {
        let region = MappedRegion::new(vec![], 50);
        assert!(region.is_empty());
        assert_eq!(region.len(), 0);
        assert_eq!(region.offset(), 50);
        let empty: &[u8] = &[];
        assert_eq!(region.as_slice(), empty);
    }

    #[test]
    fn test_mapped_region_debug() {
        let region = MappedRegion::new(vec![1, 2, 3], 0);
        let debug_str = format!("{:?}", region);
        assert!(debug_str.contains("MappedRegion"));
    }

    #[test]
    fn test_mapped_region_slice_empty() {
        let region = MappedRegion::new(vec![1, 2, 3, 4, 5], 0);
        // Empty slice at valid position
        assert_eq!(region.slice(2, 2), Some(&[][..]));
    }

    #[test]
    fn test_memory_mapped_file_path() {
        let mut temp = NamedTempFile::new().expect("create temp");
        temp.write_all(b"test").expect("write");

        let mmap = MemoryMappedFile::open(temp.path()).expect("open");
        assert!(mmap.path().contains(temp.path().to_str().unwrap()));
    }

    #[test]
    fn test_memory_mapped_file_clear_cache() {
        let mut temp = NamedTempFile::new().expect("create temp");
        temp.write_all(b"test data for caching").expect("write");

        let mut mmap = MemoryMappedFile::open(temp.path()).expect("open");

        // Map some regions
        let _ = mmap.map_region(0, 5).expect("map");
        let _ = mmap.map_region(5, 5).expect("map");
        assert_eq!(mmap.cached_regions(), 2);

        // Clear cache
        mmap.clear_cache();
        assert_eq!(mmap.cached_regions(), 0);
        assert_eq!(mmap.cached_bytes(), 0);
    }

    #[test]
    fn test_memory_mapped_file_debug() {
        let mut temp = NamedTempFile::new().expect("create temp");
        temp.write_all(b"test").expect("write");

        let mmap = MemoryMappedFile::open(temp.path()).expect("open");
        let debug_str = format!("{:?}", mmap);
        assert!(debug_str.contains("MemoryMappedFile"));
    }

    #[test]
    fn test_memory_mapped_file_read_past_end() {
        let mut temp = NamedTempFile::new().expect("create temp");
        temp.write_all(b"short").expect("write");

        let mut mmap = MemoryMappedFile::open(temp.path()).expect("open");
        let result = mmap.read_at(3, 10); // 3 + 10 = 13 > 5
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_mapped_file_map_region_cached() {
        let mut temp = NamedTempFile::new().expect("create temp");
        temp.write_all(b"ABCDEFGHIJ").expect("write");

        let mut mmap = MemoryMappedFile::open(temp.path()).expect("open");

        // First map
        let _ = mmap.map_region(3, 4).expect("map");
        assert_eq!(mmap.cached_regions(), 1);

        // Same region should reuse cache
        let region = mmap.map_region(3, 4).expect("map again");
        assert_eq!(region.as_slice(), b"DEFG");
        // Still 1 cached region because we reused
        assert_eq!(mmap.cached_regions(), 1);
    }

    #[test]
    fn test_memory_mapped_file_nonexistent() {
        let result = MemoryMappedFile::open("/nonexistent/path/file.bin");
        assert!(result.is_err());
    }

    #[test]
    fn test_page_table_get() {
        let mut table = PageTable::new();
        table.add_page(100, 10);

        let entry = table.get(100);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().size, 10);

        assert!(table.get(999).is_none());
    }

    #[test]
    fn test_page_table_is_empty() {
        let table = PageTable::new();
        assert!(table.is_empty());

        let mut table2 = PageTable::new();
        table2.add_page(100, 10);
        assert!(!table2.is_empty());
    }

    #[test]
    fn test_page_table_remove_nonexistent() {
        let mut table = PageTable::new();
        table.add_page(100, 10);

        let removed = table.remove(999);
        assert!(removed.is_none());
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn test_page_entry_clone() {
        let entry = PageEntry::new(100, 50);
        let cloned = entry.clone();
        assert_eq!(cloned.offset, entry.offset);
        assert_eq!(cloned.size, entry.size);
    }

    #[test]
    fn test_page_table_empty_lru_lfu() {
        let table = PageTable::new();
        assert!(table.lru_page().is_none());
        assert!(table.lfu_page().is_none());
    }

    #[test]
    fn test_page_table_touch_nonexistent() {
        let mut table = PageTable::new();
        table.add_page(100, 10);

        // Touching nonexistent page should be a no-op
        table.touch(999);
        assert_eq!(table.len(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn test_mapped_file_advise_sequential() {
        let mut temp = NamedTempFile::new().expect("create temp");
        temp.write_all(b"test data for advise").expect("write");

        let mapped = MappedFile::open(temp.path()).expect("open");
        let result = mapped.advise_sequential();
        assert!(result.is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn test_mapped_file_advise_random() {
        let mut temp = NamedTempFile::new().expect("create temp");
        temp.write_all(b"test data for advise").expect("write");

        let mapped = MappedFile::open(temp.path()).expect("open");
        let result = mapped.advise_random();
        assert!(result.is_ok());
    }
}
