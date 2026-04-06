// Paged KV cache implementation with block-level allocation and free-list management.

impl PagedKvCache {
    /// Create a new PagedKvCache
    ///
    /// # Arguments
    /// * `total_pages` - Total number of physical pages to allocate
    /// * `block_size` - Tokens per page (typically 16 or 32)
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per attention head
    pub fn new(total_pages: usize, block_size: usize, num_heads: usize, head_dim: usize) -> Self {
        contract_pre_page_shape!();
        let mut physical_pages = Vec::with_capacity(total_pages);
        let mut free_pages = VecDeque::with_capacity(total_pages);

        // Pre-allocate all pages
        for i in 0..total_pages {
            let page_id = PageId::new(i as u64);
            physical_pages.push(KvPage::new(page_id, block_size, num_heads, head_dim));
            free_pages.push_back(page_id);
        }

        let result = Self {
            physical_pages,
            page_tables: HashMap::new(),
            free_pages,
            block_size,
            num_heads,
            head_dim,
            total_pages,
            stats: PagedCacheStats::default(),
        };
        contract_post_page_shape!(&result);
        result
    }

    /// Allocate pages for a new sequence
    pub fn allocate_sequence(&mut self, num_tokens: usize) -> Result<SeqId, PagedCacheError> {
        let num_pages = self.tokens_to_pages(num_tokens);

        if self.free_pages.len() < num_pages {
            return Err(PagedCacheError::OutOfMemory {
                needed: num_pages,
                available: self.free_pages.len(),
            });
        }

        let seq_id = SeqId::new();
        let mut pages = Vec::with_capacity(num_pages);

        for _ in 0..num_pages {
            if let Some(page_id) = self.free_pages.pop_front() {
                // Reset the page
                let page = &mut self.physical_pages[page_id.value() as usize];
                page.num_tokens = 0;
                page.ref_count = 1;
                pages.push(page_id);
            }
        }

        self.page_tables.insert(seq_id, pages);
        self.stats.sequences_allocated += 1;
        self.stats.pages_allocated += num_pages as u64;
        self.stats.active_sequences += 1;
        self.stats.used_pages += num_pages as u64;

        Ok(seq_id)
    }

    /// Extend sequence by allocating more pages for generation
    pub fn extend(&mut self, seq_id: SeqId, num_tokens: usize) -> Result<(), PagedCacheError> {
        // First, gather info without holding mutable borrow
        let (current_pages, current_tokens) = {
            let pages = self
                .page_tables
                .get(&seq_id)
                .ok_or(PagedCacheError::SequenceNotFound(seq_id.value()))?;

            let mut total_tokens = 0;
            for page_id in pages {
                let page = &self.physical_pages[page_id.value() as usize];
                total_tokens += page.num_tokens;
            }
            (pages.len(), total_tokens)
        };

        let current_capacity = current_pages * self.block_size;
        let total_needed = current_tokens + num_tokens;

        if total_needed <= current_capacity {
            // No new pages needed
            return Ok(());
        }

        let additional_pages = self.tokens_to_pages(total_needed) - current_pages;

        if self.free_pages.len() < additional_pages {
            return Err(PagedCacheError::OutOfMemory {
                needed: additional_pages,
                available: self.free_pages.len(),
            });
        }

        // Collect new page IDs
        let mut new_pages = Vec::with_capacity(additional_pages);
        for _ in 0..additional_pages {
            if let Some(page_id) = self.free_pages.pop_front() {
                let page = &mut self.physical_pages[page_id.value() as usize];
                page.num_tokens = 0;
                page.ref_count = 1;
                new_pages.push(page_id);
            }
        }

        // Now update page table
        if let Some(pages) = self.page_tables.get_mut(&seq_id) {
            pages.extend(new_pages);
        }

        self.stats.pages_allocated += additional_pages as u64;
        self.stats.used_pages += additional_pages as u64;

        Ok(())
    }

    /// Free sequence and return pages to pool
    pub fn free_sequence(&mut self, seq_id: SeqId) {
        if let Some(pages) = self.page_tables.remove(&seq_id) {
            for page_id in pages {
                let page = &mut self.physical_pages[page_id.value() as usize];
                page.ref_count = page.ref_count.saturating_sub(1);

                // Only return to free list if no references remain
                if page.ref_count == 0 {
                    self.free_pages.push_back(page_id);
                    self.stats.pages_freed += 1;
                    self.stats.used_pages = self.stats.used_pages.saturating_sub(1);
                }
            }
            self.stats.sequences_freed += 1;
            self.stats.active_sequences = self.stats.active_sequences.saturating_sub(1);
        }
    }

    /// Fork a sequence (copy-on-write for prefix sharing)
    pub fn fork_sequence(&mut self, parent_seq_id: SeqId) -> Result<SeqId, PagedCacheError> {
        let parent_pages = self
            .page_tables
            .get(&parent_seq_id)
            .ok_or(PagedCacheError::SequenceNotFound(parent_seq_id.value()))?
            .clone();

        // Increment reference counts for shared pages
        for page_id in &parent_pages {
            self.physical_pages[page_id.value() as usize].ref_count += 1;
        }

        let child_seq_id = SeqId::new();
        self.page_tables.insert(child_seq_id, parent_pages);

        self.stats.sequences_allocated += 1;
        self.stats.active_sequences += 1;
        self.stats.cow_operations += 1;

        Ok(child_seq_id)
    }

    /// Get the number of tokens stored for a sequence
    pub fn get_sequence_tokens(&self, seq_id: SeqId) -> Result<usize, PagedCacheError> {
        let pages = self
            .page_tables
            .get(&seq_id)
            .ok_or(PagedCacheError::SequenceNotFound(seq_id.value()))?;

        let mut total_tokens = 0;
        for page_id in pages {
            let page = &self.physical_pages[page_id.value() as usize];
            total_tokens += page.num_tokens;
        }

        Ok(total_tokens)
    }

    /// Update token count for sequence (after writing KV data)
    pub fn update_tokens(
        &mut self,
        seq_id: SeqId,
        num_tokens: usize,
    ) -> Result<(), PagedCacheError> {
        let pages = self
            .page_tables
            .get(&seq_id)
            .ok_or(PagedCacheError::SequenceNotFound(seq_id.value()))?;

        let mut remaining = num_tokens;
        for page_id in pages {
            let page = &mut self.physical_pages[page_id.value() as usize];
            let tokens_in_page = remaining.min(self.block_size);
            page.num_tokens = tokens_in_page;
            remaining = remaining.saturating_sub(self.block_size);
            if remaining == 0 {
                break;
            }
        }

        Ok(())
    }

    /// Get physical page for a logical position
    pub fn get_page(
        &self,
        seq_id: SeqId,
        token_position: usize,
    ) -> Result<&KvPage, PagedCacheError> {
        let pages = self
            .page_tables
            .get(&seq_id)
            .ok_or(PagedCacheError::SequenceNotFound(seq_id.value()))?;

        let page_index = token_position / self.block_size;
        let page_id = pages
            .get(page_index)
            .ok_or(PagedCacheError::InvalidPageAccess {
                page_id: page_index as u64,
                offset: token_position,
            })?;

        Ok(&self.physical_pages[page_id.value() as usize])
    }

    /// Get mutable physical page (handles copy-on-write)
    pub fn get_page_mut(
        &mut self,
        seq_id: SeqId,
        token_position: usize,
    ) -> Result<&mut KvPage, PagedCacheError> {
        let pages = self
            .page_tables
            .get(&seq_id)
            .ok_or(PagedCacheError::SequenceNotFound(seq_id.value()))?;

        let page_index = token_position / self.block_size;
        let page_id = *pages
            .get(page_index)
            .ok_or(PagedCacheError::InvalidPageAccess {
                page_id: page_index as u64,
                offset: token_position,
            })?;

        // Handle copy-on-write if page is shared
        let page = &self.physical_pages[page_id.value() as usize];
        if page.is_shared() {
            // Allocate a new page and copy data
            let new_page_id = self
                .free_pages
                .pop_front()
                .ok_or(PagedCacheError::OutOfMemory {
                    needed: 1,
                    available: 0,
                })?;

            // Copy data to new page
            let old_page = &self.physical_pages[page_id.value() as usize];
            let keys = old_page.keys.clone();
            let values = old_page.values.clone();
            let num_tokens = old_page.num_tokens;

            // Update old page ref count
            self.physical_pages[page_id.value() as usize].ref_count -= 1;

            // Setup new page
            let new_page = &mut self.physical_pages[new_page_id.value() as usize];
            new_page.keys = keys;
            new_page.values = values;
            new_page.num_tokens = num_tokens;
            new_page.ref_count = 1;

            // Update page table
            let pages = self
                .page_tables
                .get_mut(&seq_id)
                .ok_or(PagedCacheError::SequenceNotFound(seq_id.value()))?;
            pages[page_index] = new_page_id;

            self.stats.cow_operations += 1;
            self.stats.pages_allocated += 1;
            self.stats.used_pages += 1;

            return Ok(&mut self.physical_pages[new_page_id.value() as usize]);
        }

        Ok(&mut self.physical_pages[page_id.value() as usize])
    }

    /// Get cache statistics
    pub fn stats(&self) -> &PagedCacheStats {
        &self.stats
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let page_size = self.block_size * self.num_heads * self.head_dim * 4 * 2; // f32 = 4 bytes, K+V = 2
        self.stats.used_pages as usize * page_size
    }

    /// Get total capacity in bytes
    pub fn total_capacity(&self) -> usize {
        let page_size = self.block_size * self.num_heads * self.head_dim * 4 * 2;
        self.total_pages * page_size
    }

    /// Get utilization percentage
    pub fn utilization(&self) -> f32 {
        if self.total_pages == 0 {
            return 0.0;
        }
        (self.stats.used_pages as f32 / self.total_pages as f32) * 100.0
    }

    /// Number of free pages available
    pub fn free_page_count(&self) -> usize {
        self.free_pages.len()
    }

    /// Number of pages needed for tokens
    fn tokens_to_pages(&self, num_tokens: usize) -> usize {
        num_tokens.div_ceil(self.block_size)
    }

    // ========================================================================
    // Defragmentation (llama.cpp competitive feature)
    // ========================================================================

    /// Calculate fragmentation statistics
    ///
    /// Per llama.cpp KV cache defrag: tracks holes, wasted capacity, and
    /// fragmentation ratio to decide when defragmentation is beneficial.
    pub fn fragmentation_stats(&self) -> FragmentationStats {
        // Build a usage map: true = used, false = free
        let mut usage_map = vec![false; self.total_pages];
        let mut total_tokens = 0usize;
        let mut pages_with_tokens = 0usize;

        for pages in self.page_tables.values() {
            for page_id in pages {
                let idx = page_id.value() as usize;
                if idx < self.total_pages {
                    usage_map[idx] = true;
                    let page = &self.physical_pages[idx];
                    total_tokens += page.num_tokens;
                    if page.num_tokens > 0 {
                        pages_with_tokens += 1;
                    }
                }
            }
        }

        // Count holes (transitions from used to free in the middle of used regions)
        let mut holes = 0usize;
        let mut in_used_region = false;
        let mut current_free_run = 0usize;
        let mut largest_free_region = 0usize;
        let mut free_runs = Vec::new();

        for &used in &usage_map {
            if used {
                if in_used_region && current_free_run > 0 {
                    holes += 1;
                    free_runs.push(current_free_run);
                }
                in_used_region = true;
                current_free_run = 0;
            } else {
                current_free_run += 1;
                largest_free_region = largest_free_region.max(current_free_run);
            }
        }

        // Trailing free region
        if current_free_run > 0 {
            free_runs.push(current_free_run);
        }

        // Calculate wasted capacity (unfilled slots in used pages)
        let used_pages = self.stats.used_pages as usize;
        let max_capacity = used_pages * self.block_size;
        let wasted_capacity = max_capacity.saturating_sub(total_tokens);

        // Fragmentation ratio: based on holes relative to used pages
        let fragmentation_ratio = if used_pages > 0 {
            (holes as f32) / (used_pages as f32).max(1.0)
        } else {
            0.0
        };

        // Average tokens per page
        let avg_tokens_per_page = if pages_with_tokens > 0 {
            total_tokens as f32 / pages_with_tokens as f32
        } else {
            0.0
        };

        FragmentationStats {
            holes,
            wasted_capacity,
            fragmentation_ratio: fragmentation_ratio.min(1.0),
            largest_free_region,
            avg_tokens_per_page,
        }
    }

    /// Determine if defragmentation should be performed
    ///
    /// Heuristic based on:
    /// - Fragmentation ratio > threshold (default 0.3)
    /// - Wasted capacity > 25% of used capacity
    /// - Free page count low but fragmented
    pub fn should_defragment(&self) -> bool {
        self.should_defragment_with_threshold(0.3)
    }

    /// Determine if defragmentation should be performed with custom threshold
    pub fn should_defragment_with_threshold(&self, threshold: f32) -> bool {
        let stats = self.fragmentation_stats();

        // High fragmentation ratio
        if stats.fragmentation_ratio > threshold {
            return true;
        }

        // Significant wasted capacity (>25% of block size)
        let used_pages = self.stats.used_pages as usize;
        if used_pages > 0 {
            let max_capacity = used_pages * self.block_size;
            let waste_ratio = stats.wasted_capacity as f32 / max_capacity as f32;
            if waste_ratio > 0.25 && stats.holes > 2 {
                return true;
            }
        }

        // Low on free pages but have holes we can recover
        let free_ratio = self.free_pages.len() as f32 / self.total_pages as f32;
        if free_ratio < 0.1 && stats.holes > 0 {
            return true;
        }

        false
    }

    /// Map a (sequence, position) pair to a flat slot index.
    ///
    /// Contract: paged-kv-cache-v1 / slot_mapping
    pub fn slot(&self, seq_id: SeqId, position: usize) -> Result<usize, PagedCacheError> {
        contract_pre_slot_mapping!();
        let page_idx = position / self.block_size;
        let offset = position % self.block_size;
        let pages = self
            .page_tables
            .get(&seq_id)
            .ok_or(PagedCacheError::UnknownSequence(seq_id))?;
        if page_idx >= pages.len() {
            return Err(PagedCacheError::OutOfBounds {
                seq_id,
                position,
                allocated: pages.len() * self.block_size,
            });
        }
        let page_id = pages[page_idx];
        let result = page_id.value() as usize * self.block_size + offset;
        contract_post_slot_mapping!(&result);
        Ok(result)
    }

    /// Allocate a contiguous block of pages from the free list.
    ///
    /// Contract: paged-kv-cache-v1 / block_allocation
    pub fn allocate_blocks(&mut self, num_blocks: usize) -> Result<Vec<PageId>, PagedCacheError> {
        contract_pre_block_allocation!();
        if self.free_pages.len() < num_blocks {
            return Err(PagedCacheError::OutOfPages {
                requested: num_blocks,
                available: self.free_pages.len(),
            });
        }
        let allocated: Vec<PageId> = (0..num_blocks)
            .map(|_| self.free_pages.pop_front().unwrap())
            .collect();
        self.stats.used_pages += allocated.len() as u64;
        contract_post_block_allocation!(&allocated);
        Ok(allocated)
    }
}
