//! Huffman coding for Zstandard literals.

use crate::{Error, Result};

/// Maximum Huffman table log.
pub const HUFFMAN_MAX_TABLE_LOG: u8 = 11;

/// Maximum number of symbols.
pub const HUFFMAN_MAX_SYMBOLS: usize = 256;

/// Huffman decoding table entry.
#[derive(Debug, Clone, Copy, Default)]
pub struct HuffmanEntry {
    /// Symbol value.
    pub symbol: u8,
    /// Number of bits for this symbol.
    pub bits: u8,
}

/// Huffman decoding table.
#[derive(Debug, Clone)]
pub struct HuffmanTable {
    /// Decoding entries (indexed by bits).
    pub entries: Vec<HuffmanEntry>,
    /// Table size log.
    pub table_log: u8,
}

impl HuffmanTable {
    /// Create a Huffman table from bit lengths.
    pub fn from_weights(weights: &[u8], max_bits: u8) -> Result<Self> {
        if weights.is_empty() {
            return Err(Error::InvalidInput("empty weights".to_string()));
        }

        let table_size = 1usize << max_bits;
        let mut entries = vec![HuffmanEntry::default(); table_size];

        // Sort symbols by weight (descending)
        let mut symbols: Vec<(usize, u8)> = weights
            .iter()
            .enumerate()
            .filter(|(_, &w)| w > 0)
            .map(|(i, &w)| (i, w))
            .collect();
        symbols.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        // Assign codes
        let mut code = 0u32;
        let mut prev_bits = 0u8;

        for (symbol, bits) in symbols {
            if bits > prev_bits {
                code <<= bits - prev_bits;
            }
            prev_bits = bits;

            // Fill table entries for this symbol
            let num_entries = 1usize << (max_bits - bits);
            for i in 0..num_entries {
                let index = ((code as usize) << (max_bits - bits)) | i;
                if index < table_size {
                    entries[index] = HuffmanEntry {
                        symbol: symbol as u8,
                        bits,
                    };
                }
            }

            code += 1;
        }

        Ok(Self {
            entries,
            table_log: max_bits,
        })
    }

    /// Decode one symbol.
    #[inline]
    #[must_use]
    pub fn decode(&self, bits: u64, bit_pos: usize) -> (u8, u8) {
        let index = ((bits >> bit_pos) as usize) & ((1 << self.table_log) - 1);
        let entry = &self.entries[index];
        (entry.symbol, entry.bits)
    }
}

/// Read Huffman weights from compressed header.
pub fn read_weights(data: &[u8]) -> Result<(Vec<u8>, usize)> {
    if data.is_empty() {
        return Err(Error::CorruptedData("empty huffman header".to_string()));
    }

    let header = data[0];

    if header < 128 {
        // Compressed weights using FSE
        // For simplicity, return error - would need full FSE decoder
        Err(Error::Unsupported(
            "FSE-compressed Huffman weights".to_string(),
        ))
    } else {
        // Direct representation
        let num_symbols = (header - 127) as usize;
        let bytes_needed = num_symbols.div_ceil(2);

        if data.len() < 1 + bytes_needed {
            return Err(Error::CorruptedData(
                "truncated huffman weights".to_string(),
            ));
        }

        let mut weights = Vec::with_capacity(num_symbols);
        for i in 0..num_symbols {
            let byte_idx = 1 + i / 2;
            let weight = if i % 2 == 0 {
                data[byte_idx] >> 4
            } else {
                data[byte_idx] & 0x0F
            };
            weights.push(weight);
        }

        Ok((weights, 1 + bytes_needed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huffman_table_creation() {
        let weights = vec![4, 4, 4, 4, 3, 3, 2, 1];
        let table = HuffmanTable::from_weights(&weights, 4).unwrap();
        assert_eq!(table.entries.len(), 16);
    }

    #[test]
    fn test_read_weights_direct() {
        // Formula: num_symbols = header - 127
        // For 4 symbols: header = 127 + 4 = 131
        // bytes_needed = (4 + 1) / 2 = 2
        // Total data: 1 (header) + 2 (weights) = 3 bytes
        let data = [131, 0x44, 0x33];
        let (weights, consumed) = read_weights(&data).unwrap();
        assert_eq!(weights.len(), 4);
        assert_eq!(consumed, 3);
        assert_eq!(weights[0], 4); // first nibble of 0x44
        assert_eq!(weights[1], 4); // second nibble of 0x44
        assert_eq!(weights[2], 3); // first nibble of 0x33
        assert_eq!(weights[3], 3); // second nibble of 0x33
    }

    #[test]
    fn test_read_weights_two_symbols() {
        // 2 symbols: header = 127 + 2 = 129
        // bytes_needed = (2 + 1) / 2 = 1
        let data = [129, 0x42];
        let (weights, consumed) = read_weights(&data).unwrap();
        assert_eq!(weights.len(), 2);
        assert_eq!(weights[0], 4); // high nibble
        assert_eq!(weights[1], 2); // low nibble
        assert_eq!(consumed, 2);
    }

    #[test]
    fn test_empty_weights_error() {
        let result = HuffmanTable::from_weights(&[], 4);
        assert!(result.is_err());
    }
}
