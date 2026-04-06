
impl PagedKvCache {

    /// Perform defragmentation - compact pages to reduce fragmentation
    ///
    /// This operation:
    /// 1. Identifies fragmented sequences
    /// 2. Moves pages to create contiguous allocations
    /// 3. Updates page tables accordingly
    /// 4. Returns number of pages moved
    ///
    /// Note: This is a relatively expensive operation and should be called
    /// during low-activity periods or when `should_defragment()` returns true.
    pub fn defragment(&mut self) -> usize {
        let mut pages_moved = 0;

        // Collect sequences to defragment (those with non-contiguous pages)
        let seq_ids: Vec<SeqId> = self.page_tables.keys().copied().collect();

        for seq_id in seq_ids {
            pages_moved += self.compact_sequence(seq_id);
        }

        if pages_moved > 0 {
            self.stats.defrag_operations += 1;
            self.stats.pages_moved += pages_moved as u64;
        }

        pages_moved
    }

    /// Compact a specific sequence's pages to be contiguous
    ///
    /// Returns number of pages moved.
    pub fn compact_sequence(&mut self, seq_id: SeqId) -> usize {
        let pages = match self.page_tables.get(&seq_id) {
            Some(p) => p.clone(),
            None => return 0,
        };

        if pages.is_empty() {
            return 0;
        }

        // Check if already contiguous
        let mut is_contiguous = true;
        for i in 1..pages.len() {
            let prev_id = pages[i - 1].value();
            let curr_id = pages[i].value();
            // Check if pages are adjacent (within reasonable range for contiguity)
            if curr_id != prev_id + 1 {
                is_contiguous = false;
                break;
            }
        }

        if is_contiguous {
            return 0; // Already compact
        }

        // Find target region - look for contiguous free space or lowest-numbered free pages
        let mut pages_moved = 0;

        // Strategy: For each non-contiguous page, try to move it adjacent to previous
        let mut new_page_list = vec![pages[0]];

        for i in 1..pages.len() {
            let prev_page_id = new_page_list[i - 1];
            let curr_page_id = pages[i];

            // Check if current page is already adjacent
            if curr_page_id.value() == prev_page_id.value() + 1 {
                new_page_list.push(curr_page_id);
                continue;
            }

            // Try to find a free page adjacent to previous
            let target_id = PageId::new(prev_page_id.value() + 1);
            let target_idx = target_id.value() as usize;

            if target_idx < self.total_pages && self.is_page_free(target_id) {
                // Move data from curr_page to target_page
                let curr_idx = curr_page_id.value() as usize;

                // Copy data
                let keys = self.physical_pages[curr_idx].keys.clone();
                let values = self.physical_pages[curr_idx].values.clone();
                let num_tokens = self.physical_pages[curr_idx].num_tokens;
                let ref_count = self.physical_pages[curr_idx].ref_count;

                // If current page is shared, we can't move it (COW semantics)
                if ref_count > 1 {
                    new_page_list.push(curr_page_id);
                    continue;
                }

                // Remove target from free list
                self.free_pages.retain(|&p| p != target_id);

                // Setup target page
                self.physical_pages[target_idx].keys = keys;
                self.physical_pages[target_idx].values = values;
                self.physical_pages[target_idx].num_tokens = num_tokens;
                self.physical_pages[target_idx].ref_count = 1;

                // Clear source page and return to free list
                self.physical_pages[curr_idx].num_tokens = 0;
                self.physical_pages[curr_idx].ref_count = 0;
                self.free_pages.push_back(curr_page_id);

                new_page_list.push(target_id);
                pages_moved += 1;
            } else {
                // Can't move, keep current page
                new_page_list.push(curr_page_id);
            }
        }

        // Update page table
        if let Some(entry) = self.page_tables.get_mut(&seq_id) {
            *entry = new_page_list;
        }

        pages_moved
    }

    /// Check if a page is free
    fn is_page_free(&self, page_id: PageId) -> bool {
        self.free_pages.contains(&page_id)
    }

    /// Get contiguity score for a sequence (1.0 = fully contiguous)
    pub fn sequence_contiguity(&self, seq_id: SeqId) -> Result<f32, PagedCacheError> {
        let pages = self
            .page_tables
            .get(&seq_id)
            .ok_or(PagedCacheError::SequenceNotFound(seq_id.value()))?;

        if pages.len() <= 1 {
            return Ok(1.0); // Single page is always contiguous
        }

        let mut contiguous_pairs = 0;
        for i in 1..pages.len() {
            if pages[i].value() == pages[i - 1].value() + 1 {
                contiguous_pairs += 1;
            }
        }

        Ok(contiguous_pairs as f32 / (pages.len() - 1) as f32)
    }
}

include!("mod_paged.rs");
