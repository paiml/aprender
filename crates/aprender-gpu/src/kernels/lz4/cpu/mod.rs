//! CPU Reference Implementation for LZ4 compression
//!
//! Used for validation and testing. These functions implement the LZ4 block
//! format specification for correctness verification.

use super::{LZ4_HASH_BITS, LZ4_HASH_MULT, LZ4_HASH_SIZE, LZ4_MAX_OFFSET, LZ4_MIN_MATCH};

/// Read 4 bytes as little-endian u32
#[inline]
#[must_use]
pub fn read_u32_le(data: &[u8], pos: usize) -> u32 {
    debug_assert!(pos + 4 <= data.len());
    u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
}

/// LZ4 hash function: hash 4 bytes to 12-bit index
///
/// Uses Knuth multiplicative hash for good distribution.
/// Formula: hash = (val * 2654435761) >> (32 - 12)
#[inline]
#[must_use]
pub fn lz4_hash(val: u32) -> u32 {
    val.wrapping_mul(LZ4_HASH_MULT) >> (32 - LZ4_HASH_BITS)
}

/// Hash 4 bytes from a slice position
#[inline]
#[must_use]
pub fn lz4_hash_at(data: &[u8], pos: usize) -> u32 {
    lz4_hash(read_u32_le(data, pos))
}

/// Count matching bytes between two positions
///
/// Returns the number of matching bytes (minimum 0).
/// Used after finding a 4-byte hash match to extend the match.
#[inline]
#[must_use]
pub fn lz4_match_length(data: &[u8], pos1: usize, pos2: usize, limit: usize) -> usize {
    let mut len = 0;
    let max_len = limit.min(data.len() - pos1.max(pos2));

    while len < max_len && data[pos1 + len] == data[pos2 + len] {
        len += 1;
    }
    len
}

/// Compute the LZ4 token byte from literal and match lengths.
#[inline]
fn lz4_compute_token(literal_len: usize, match_length: usize) -> u8 {
    let token_lit = literal_len.min(15) as u8;
    let token_match = if match_length == 0 {
        0
    } else {
        (match_length - LZ4_MIN_MATCH as usize).min(15) as u8
    };
    (token_lit << 4) | token_match
}

/// Estimate the number of bytes needed to encode an extended length.
#[inline]
fn lz4_extended_len_size(base: usize) -> usize {
    if base < 15 {
        return 0;
    }
    let extra = base - 15;
    1 + extra / 255 + 1
}

/// Estimate total output bytes needed for an LZ4 sequence.
fn lz4_estimate_output_size(literal_len: usize, match_length: usize) -> usize {
    let token_size = 1;
    let lit_ext = lz4_extended_len_size(literal_len);
    let match_overhead = if match_length > 0 { 2 } else { 0 };
    let match_ext = if match_length > 0 {
        lz4_extended_len_size(match_length - LZ4_MIN_MATCH as usize)
    } else {
        0
    };
    token_size + lit_ext + literal_len + match_overhead + match_ext
}

/// Write an LZ4 extended length (sequence of 255-byte continuations) to output.
fn lz4_write_extended_len(output: &mut [u8], out_pos: &mut usize, base: usize) {
    if base < 15 {
        return;
    }
    let mut remaining = base - 15;
    while remaining >= 255 {
        output[*out_pos] = 255;
        *out_pos += 1;
        remaining -= 255;
    }
    output[*out_pos] = remaining as u8;
    *out_pos += 1;
}

/// Write match offset and optional extended match length to output.
fn lz4_write_match(output: &mut [u8], out_pos: &mut usize, match_offset: u16, match_length: usize) {
    if match_length == 0 {
        return;
    }
    output[*out_pos] = (match_offset & 0xFF) as u8;
    output[*out_pos + 1] = (match_offset >> 8) as u8;
    *out_pos += 2;
    lz4_write_extended_len(output, out_pos, match_length - LZ4_MIN_MATCH as usize);
}

/// Encode LZ4 sequence to output buffer
///
/// Returns number of bytes written to output.
/// Format: [token] [extra_literal_len...] [literals] [offset_lo] [offset_hi] [extra_match_len...]
pub fn lz4_encode_sequence(
    output: &mut [u8],
    out_pos: &mut usize,
    literals: &[u8],
    match_offset: u16,
    match_length: usize,
) -> Result<(), &'static str> {
    let literal_len = literals.len();
    let needed = lz4_estimate_output_size(literal_len, match_length);

    if *out_pos + needed > output.len() {
        return Err("Output buffer too small");
    }

    output[*out_pos] = lz4_compute_token(literal_len, match_length);
    *out_pos += 1;

    lz4_write_extended_len(output, out_pos, literal_len);

    output[*out_pos..*out_pos + literal_len].copy_from_slice(literals);
    *out_pos += literal_len;

    lz4_write_match(output, out_pos, match_offset, match_length);

    Ok(())
}

/// Read an LZ4 extended length (sequence of 255-byte continuations).
fn lz4_read_extended_len(
    input: &[u8],
    in_pos: &mut usize,
    base: usize,
) -> Result<usize, &'static str> {
    let mut len = base;
    loop {
        if *in_pos >= input.len() {
            return Err("Truncated extended length");
        }
        let byte = input[*in_pos] as usize;
        *in_pos += 1;
        len += byte;
        if byte != 255 {
            break;
        }
    }
    Ok(len)
}

/// Copy literal bytes from input to output during decompression.
fn lz4_copy_literals(
    input: &[u8],
    output: &mut [u8],
    in_pos: &mut usize,
    out_pos: &mut usize,
    literal_len: usize,
) -> Result<(), &'static str> {
    if *in_pos + literal_len > input.len() {
        return Err("Truncated literals");
    }
    if *out_pos + literal_len > output.len() {
        return Err("Output buffer overflow (literals)");
    }
    output[*out_pos..*out_pos + literal_len]
        .copy_from_slice(&input[*in_pos..*in_pos + literal_len]);
    *in_pos += literal_len;
    *out_pos += literal_len;
    Ok(())
}

/// Read and validate a match offset during decompression.
fn lz4_read_match_offset(
    input: &[u8],
    in_pos: &mut usize,
    out_pos: usize,
) -> Result<usize, &'static str> {
    if *in_pos + 2 > input.len() {
        return Err("Truncated match offset");
    }
    let offset = (input[*in_pos] as usize) | ((input[*in_pos + 1] as usize) << 8);
    *in_pos += 2;
    if offset == 0 {
        return Err("Invalid zero offset");
    }
    if offset > out_pos {
        return Err("Invalid offset (exceeds output)");
    }
    Ok(offset)
}

/// Copy overlapping match bytes from earlier in output buffer.
fn lz4_copy_match(
    output: &mut [u8],
    out_pos: &mut usize,
    offset: usize,
    match_len: usize,
) -> Result<(), &'static str> {
    if *out_pos + match_len > output.len() {
        return Err("Output buffer overflow (match)");
    }
    let match_start = *out_pos - offset;
    for i in 0..match_len {
        output[*out_pos + i] = output[match_start + i];
    }
    *out_pos += match_len;
    Ok(())
}

/// Decode one LZ4 sequence (literals + match) from input.
fn lz4_decode_sequence(
    input: &[u8],
    output: &mut [u8],
    in_pos: &mut usize,
    out_pos: &mut usize,
) -> Result<bool, &'static str> {
    let token = input[*in_pos];
    *in_pos += 1;

    let mut literal_len = (token >> 4) as usize;
    let match_len_base = (token & 0x0F) as usize;

    if literal_len == 15 {
        literal_len = lz4_read_extended_len(input, in_pos, literal_len)?;
    }
    if literal_len > 0 {
        lz4_copy_literals(input, output, in_pos, out_pos, literal_len)?;
    }
    if *in_pos >= input.len() {
        return Ok(false); // last sequence, no match
    }

    let offset = lz4_read_match_offset(input, in_pos, *out_pos)?;
    let mut match_len = match_len_base + LZ4_MIN_MATCH as usize;
    if match_len_base == 15 {
        match_len = lz4_read_extended_len(input, in_pos, match_len)?;
    }
    lz4_copy_match(output, out_pos, offset, match_len)?;
    Ok(true)
}

/// LZ4 decompress a block (CPU reference implementation)
///
/// Returns decompressed size, or error if decompression fails.
/// Used for F001 lossless verification.
pub fn lz4_decompress_block(input: &[u8], output: &mut [u8]) -> Result<usize, &'static str> {
    if input.is_empty() {
        return Ok(0);
    }

    let mut in_pos = 0usize;
    let mut out_pos = 0usize;

    while in_pos < input.len() {
        if !lz4_decode_sequence(input, output, &mut in_pos, &mut out_pos)? {
            break;
        }
    }

    Ok(out_pos)
}

/// Check if a hash table entry represents a valid match at the current position.
fn lz4_try_match(input: &[u8], in_pos: usize, match_pos: usize) -> Option<(usize, usize)> {
    let offset = in_pos - match_pos;
    if offset == 0 || offset > LZ4_MAX_OFFSET as usize || match_pos + 4 > input.len() {
        return None;
    }
    if read_u32_le(input, in_pos) != read_u32_le(input, match_pos) {
        return None;
    }
    let match_len =
        lz4_match_length(input, in_pos + 4, match_pos + 4, input.len() - in_pos - 4) + 4;
    Some((offset, match_len))
}

/// State for the LZ4 compression loop.
struct Lz4CompressState {
    hash_table: [u32; LZ4_HASH_SIZE as usize],
    in_pos: usize,
    out_pos: usize,
    anchor: usize,
}

impl Lz4CompressState {
    fn new() -> Self {
        Self {
            hash_table: [0u32; LZ4_HASH_SIZE as usize],
            in_pos: 0,
            out_pos: 0,
            anchor: 0,
        }
    }

    /// Process one position in the compression loop.
    /// Returns `true` if the main loop should continue, `false` to break.
    fn step(&mut self, input: &[u8], output: &mut [u8]) -> Result<bool, &'static str> {
        let h = lz4_hash_at(input, self.in_pos);
        let match_pos = self.hash_table[h as usize] as usize;
        self.hash_table[h as usize] = self.in_pos as u32;

        if let Some((offset, match_len)) = lz4_try_match(input, self.in_pos, match_pos) {
            let literals = &input[self.anchor..self.in_pos];
            lz4_encode_sequence(
                output,
                &mut self.out_pos,
                literals,
                offset as u16,
                match_len,
            )?;
            self.in_pos += match_len;
            self.anchor = self.in_pos;
        } else {
            self.in_pos += 1;
        }

        Ok(self.in_pos + 5 <= input.len())
    }

    /// Flush any remaining literal bytes after the main loop.
    fn flush_trailing_literals(
        &mut self,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<(), &'static str> {
        if self.anchor < input.len() {
            let literals = &input[self.anchor..];
            lz4_encode_sequence(output, &mut self.out_pos, literals, 0, 0)?;
        }
        Ok(())
    }
}

/// LZ4 compress a block (CPU reference implementation)
///
/// Returns compressed size, or error if compression fails.
pub fn lz4_compress_block(input: &[u8], output: &mut [u8]) -> Result<usize, &'static str> {
    if input.is_empty() {
        return Ok(0);
    }

    let mut state = Lz4CompressState::new();

    if input.len() < LZ4_MIN_MATCH as usize {
        lz4_encode_sequence(output, &mut state.out_pos, input, 0, 0)?;
        return Ok(state.out_pos);
    }

    while state.in_pos + LZ4_MIN_MATCH as usize <= input.len() {
        if !state.step(input, output)? {
            break;
        }
    }

    state.flush_trailing_literals(input, output)?;
    Ok(state.out_pos)
}

#[cfg(test)]
mod tests;
