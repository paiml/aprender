//! Minimal, dependency-free 8-bit grayscale PNG encoder.
//!
//! `apr probar tensor --format png` used to advertise `.png` files while writing
//! Netpbm `.pgm` bytes, so every path it printed was a file that did not exist.
//! This module produces a real PNG so the printed manifest is true.
//!
//! The encoder emits the smallest spec-conformant stream: a grayscale IHDR, an
//! IDAT holding a zlib wrapper around *stored* (uncompressed) DEFLATE blocks,
//! and IEND. Stored blocks keep the implementation auditable — no compression
//! state machine to get wrong — at the cost of the file being ~raster-sized.
//! Histogram strips are 256x100, so that is ~26 KB.

/// PNG file signature (RFC 2083 §3.1).
const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

/// Largest payload a single DEFLATE stored block can carry.
const MAX_STORED_BLOCK: usize = 65_535;

/// Encode an 8-bit grayscale raster as a PNG byte stream.
///
/// `pixels` is row-major, `width * height` bytes. Returns `None` when the
/// dimensions are zero or do not match the buffer length — callers must not
/// write a truncated image.
pub(crate) fn encode_grayscale(width: usize, height: usize, pixels: &[u8]) -> Option<Vec<u8>> {
    if width == 0 || height == 0 || pixels.len() != width.checked_mul(height)? {
        return None;
    }
    let w32 = u32::try_from(width).ok()?;
    let h32 = u32::try_from(height).ok()?;

    // Each scanline is prefixed with its filter-type byte (0 = None).
    let mut raw = Vec::with_capacity(height * (width + 1));
    for row in pixels.chunks_exact(width) {
        raw.push(0u8);
        raw.extend_from_slice(row);
    }

    let mut out = Vec::new();
    out.extend_from_slice(&SIGNATURE);

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&w32.to_be_bytes());
    ihdr.extend_from_slice(&h32.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(0); // color type: grayscale
    ihdr.push(0); // compression method: deflate
    ihdr.push(0); // filter method: adaptive
    ihdr.push(0); // interlace: none
    write_chunk(&mut out, b"IHDR", &ihdr);

    write_chunk(&mut out, b"IDAT", &zlib_stored(&raw));
    write_chunk(&mut out, b"IEND", &[]);
    Some(out)
}

/// Append a length-prefixed, CRC-suffixed PNG chunk.
fn write_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    let len = u32::try_from(data.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

/// Wrap `data` in a zlib stream built from DEFLATE stored blocks.
fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 64);
    // CMF: deflate, 32K window. FLG chosen so (CMF << 8 | FLG) % 31 == 0.
    out.push(0x78);
    out.push(0x01);

    if data.is_empty() {
        out.push(0x01); // BFINAL=1, BTYPE=00
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(!0u16).to_le_bytes());
    } else {
        let mut chunks = data.chunks(MAX_STORED_BLOCK).peekable();
        while let Some(chunk) = chunks.next() {
            let final_block = u8::from(chunks.peek().is_none());
            out.push(final_block); // BFINAL bit, BTYPE=00 (stored)
            let len = u16::try_from(chunk.len()).unwrap_or(u16::MAX);
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&(!len).to_le_bytes());
            out.extend_from_slice(chunk);
        }
    }

    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

/// CRC-32 (IEEE 802.3), as required for every PNG chunk.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Adler-32 checksum trailing the zlib stream.
fn adler32(data: &[u8]) -> u32 {
    let mut a = 1u32;
    let mut b = 0u32;
    for &byte in data {
        a = (a + u32::from(byte)) % 65_521;
        b = (b + a) % 65_521;
    }
    (b << 16) | a
}

#[cfg(test)]
#[path = "png_encode_tests.rs"]
mod tests;
