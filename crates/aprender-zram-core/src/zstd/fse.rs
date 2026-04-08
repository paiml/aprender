//! Finite State Entropy (FSE) codec.
//!
//! FSE is an entropy coding method used by Zstandard for sequences.
//! It's based on Asymmetric Numeral Systems (ANS).

use crate::Result;

/// Maximum accuracy log for FSE tables.
pub const FSE_MAX_ACCURACY_LOG: u8 = 9;

/// Predefined literal lengths table.
pub const LL_DEFAULT_DISTRIBUTION: &[i16] = &[
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
    -1, -1, -1, -1,
];

/// Predefined match lengths table.
pub const ML_DEFAULT_DISTRIBUTION: &[i16] = &[
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
];

/// Predefined offsets table.
pub const OF_DEFAULT_DISTRIBUTION: &[i16] = &[
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
];

/// FSE decoding table entry.
#[derive(Debug, Clone, Copy, Default)]
pub struct FseEntry {
    /// Symbol to output.
    pub symbol: u8,
    /// Number of bits to read for next state.
    pub bits: u8,
    /// Base value to add to bits.
    pub baseline: u16,
}

/// FSE decoding table.
#[derive(Debug, Clone)]
pub struct FseTable {
    /// Table entries indexed by state.
    pub entries: Vec<FseEntry>,
    /// Accuracy log (table size = 1 << `accuracy_log`).
    pub accuracy_log: u8,
}

impl FseTable {
    /// Create a new FSE table from normalized counts.
    pub fn from_distribution(distribution: &[i16], accuracy_log: u8) -> Result<Self> {
        let table_size = 1usize << accuracy_log;
        let mut entries = vec![FseEntry::default(); table_size];

        // Build table using the spread algorithm
        let mut high_threshold = table_size;
        let mut position = 0usize;

        for (symbol, &count) in distribution.iter().enumerate() {
            if count == -1 {
                // Symbol with probability "less than 1"
                high_threshold -= 1;
                entries[high_threshold].symbol = symbol as u8;
            } else if count > 0 {
                for _ in 0..count {
                    entries[position].symbol = symbol as u8;
                    position =
                        (position + (table_size >> 1) + (table_size >> 3) + 3) & (table_size - 1);
                    while position >= high_threshold {
                        position = (position + (table_size >> 1) + (table_size >> 3) + 3)
                            & (table_size - 1);
                    }
                }
            }
        }

        // Build decoding info
        let mut symbol_next = vec![0u32; distribution.len()];
        for (symbol, &count) in distribution.iter().enumerate() {
            if count > 0 {
                symbol_next[symbol] = count as u32;
            } else if count == -1 {
                symbol_next[symbol] = 1;
            }
        }

        for entry in &mut entries {
            let symbol = entry.symbol as usize;
            let next = symbol_next[symbol];
            symbol_next[symbol] += 1;

            let bits = accuracy_log - next.ilog2() as u8;
            entry.bits = bits;
            entry.baseline = ((next << bits) - table_size as u32) as u16;
        }

        Ok(Self {
            entries,
            accuracy_log,
        })
    }

    /// Decode one symbol and advance state.
    #[inline]
    pub fn decode(&self, state: &mut u32, bits: &mut u64, bit_pos: &mut usize) -> u8 {
        let entry = &self.entries[*state as usize];
        let symbol = entry.symbol;

        // Read bits for next state
        let add_bits = (*bits >> *bit_pos) & ((1 << entry.bits) - 1);
        *bit_pos += entry.bits as usize;
        *state = u32::from(entry.baseline) + add_bits as u32;

        symbol
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fse_table_creation() {
        let table = FseTable::from_distribution(LL_DEFAULT_DISTRIBUTION, 6).unwrap();
        assert_eq!(table.entries.len(), 64);
    }

    #[test]
    fn test_default_distributions_valid() {
        // Verify all default distributions can build tables
        FseTable::from_distribution(LL_DEFAULT_DISTRIBUTION, 6).unwrap();
        FseTable::from_distribution(ML_DEFAULT_DISTRIBUTION, 6).unwrap();
        FseTable::from_distribution(OF_DEFAULT_DISTRIBUTION, 5).unwrap();
    }
}
