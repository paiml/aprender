
impl QuantizedPagedKvCache {
    /// Create new quantized paged KV cache
    pub fn new(
        total_pages: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        quant_type: KvQuantType,
    ) -> Self {
        let mut physical_pages = Vec::with_capacity(total_pages);
        let mut free_pages = VecDeque::with_capacity(total_pages);

        for i in 0..total_pages {
            let page_id = PageId::new(i as u64);
            physical_pages.push(QuantizedKvPage::new(
                page_id, quant_type, block_size, num_heads, head_dim,
            ));
            free_pages.push_back(page_id);
        }

        Self {
            physical_pages,
            page_tables: HashMap::new(),
            free_pages,
            quant_type,
            block_size,
            num_heads,
            head_dim,
            total_pages,
            stats: PagedCacheStats::default(),
        }
    }

    /// Get quantization type
    pub fn quant_type(&self) -> KvQuantType {
        self.quant_type
    }

    /// Allocate pages for a sequence
    pub fn allocate_sequence(&mut self, num_tokens: usize) -> Result<SeqId, PagedCacheError> {
        let num_pages = num_tokens.div_ceil(self.block_size);

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

    /// Free a sequence
    pub fn free_sequence(&mut self, seq_id: SeqId) {
        if let Some(pages) = self.page_tables.remove(&seq_id) {
            for page_id in pages {
                let page = &mut self.physical_pages[page_id.value() as usize];
                page.ref_count = page.ref_count.saturating_sub(1);

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

    /// Get page for a token position
    pub fn get_page(
        &self,
        seq_id: SeqId,
        token_position: usize,
    ) -> Result<&QuantizedKvPage, PagedCacheError> {
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

    /// Get mutable page for a token position
    pub fn get_page_mut(
        &mut self,
        seq_id: SeqId,
        token_position: usize,
    ) -> Result<&mut QuantizedKvPage, PagedCacheError> {
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

        Ok(&mut self.physical_pages[page_id.value() as usize])
    }

    /// Get total pages capacity
    pub fn total_pages(&self) -> usize {
        self.total_pages
    }

    /// Total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.physical_pages
            .iter()
            .filter(|p| p.ref_count > 0)
            .map(QuantizedKvPage::memory_bytes)
            .sum()
    }

    /// FP32 equivalent memory (for comparison)
    pub fn fp32_equivalent_memory(&self) -> usize {
        let page_size = self.block_size * self.num_heads * self.head_dim * 4 * 2;
        self.stats.used_pages as usize * page_size
    }

    /// Memory savings ratio (1.0 = no savings, 0.25 = 4x reduction)
    pub fn memory_savings(&self) -> f32 {
        let fp32_mem = self.fp32_equivalent_memory();
        if fp32_mem == 0 {
            return 1.0;
        }
        self.memory_usage() as f32 / fp32_mem as f32
    }

    /// Get statistics
    pub fn stats(&self) -> &PagedCacheStats {
        &self.stats
    }

    /// Free page count
    pub fn free_page_count(&self) -> usize {
        self.free_pages.len()
    }
}

/// Find longest matching prefix in a sequence of tokens
///
/// Returns (hash, num_matching_tokens) for the longest cached prefix
pub fn find_longest_prefix(cache: &mut PrefixCache, tokens: &[u32]) -> Option<(PrefixHash, usize)> {
    let mut best_match = None;
    let mut best_len = 0;

    // Try progressively longer prefixes (from 1 token up to full sequence)
    for len in 1..=tokens.len() {
        let prefix_hash = compute_prefix_hash(&tokens[..len]);
        if cache.contains(prefix_hash) && len > best_len {
            best_len = len;
            best_match = Some((prefix_hash, len));
        }
    }

    // Update stats if found
    if let Some((hash, _)) = best_match {
        cache.lookup(hash); // Update access time
    }

    best_match
}

// ============================================================================
// Tests
// ============================================================================

// Tests extracted to tests.rs (PMAT-802)
#[cfg(test)]
#[path = "tests.rs"]
mod paged_kv_tests;

// Additional coverage tests (part 2)
#[cfg(test)]
#[path = "tests_alloc_exact.rs"]
mod paged_kv_tests_part_02;

// Additional coverage tests (part 3)
#[cfg(test)]
#[path = "tests_write.rs"]
mod paged_kv_tests_part_03;

// Inline tests extracted to inline_tests.rs (PMAT-COMPLY)
#[cfg(test)]
#[path = "inline_tests.rs"]
mod inline_tests;
