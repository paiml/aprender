
impl PageEntry {
    /// Create a new page entry.
    #[must_use]
    pub(crate) fn new(offset: u64, size: usize) -> Self {
        Self {
            offset,
            size,
            access_count: 0,
            last_access: 0,
        }
    }

    /// Record an access to this page.
    pub(crate) fn touch(&mut self, timestamp: u64) {
        self.access_count += 1;
        self.last_access = timestamp;
    }
}

/// Page table for tracking memory pages.
#[derive(Debug, Default)]
pub(crate) struct PageTable {
    /// Pages indexed by offset.
    pages: std::collections::HashMap<u64, PageEntry>,
    /// Monotonic timestamp counter.
    timestamp: u64,
}

impl PageTable {
    /// Create a new empty page table.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Add a page to the table.
    pub(crate) fn add_page(&mut self, offset: u64, size: usize) {
        let mut entry = PageEntry::new(offset, size);
        entry.touch(self.next_timestamp());
        self.pages.insert(offset, entry);
    }

    /// Touch a page (record access).
    pub(crate) fn touch(&mut self, offset: u64) {
        let timestamp = self.next_timestamp();
        if let Some(entry) = self.pages.get_mut(&offset) {
            entry.touch(timestamp);
        }
    }

    /// Get the least recently used page offset.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn lru_page(&self) -> Option<u64> {
        self.pages
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(&offset, _)| offset)
    }

    /// Get the least frequently used page offset.
    #[must_use]
    pub(crate) fn lfu_page(&self) -> Option<u64> {
        self.pages
            .iter()
            .min_by_key(|(_, entry)| entry.access_count)
            .map(|(&offset, _)| offset)
    }

    /// Remove a page from the table.
    pub(crate) fn remove(&mut self, offset: u64) -> Option<PageEntry> {
        self.pages.remove(&offset)
    }

    /// Get page entry by offset.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn get(&self, offset: u64) -> Option<&PageEntry> {
        self.pages.get(&offset)
    }

    /// Get number of pages.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.pages.len()
    }

    /// Check if table is empty.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Get total size of all pages.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn total_size(&self) -> usize {
        self.pages.values().map(|p| p.size).sum()
    }

    /// Get next timestamp.
    fn next_timestamp(&mut self) -> u64 {
        self.timestamp += 1;
        self.timestamp
    }
}
