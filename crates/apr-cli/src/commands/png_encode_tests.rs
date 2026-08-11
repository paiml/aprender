use super::*;

/// Walk the chunk list, verifying every CRC, and return (kind, payload) pairs.
fn parse_chunks(png: &[u8]) -> Vec<(String, Vec<u8>)> {
    assert_eq!(&png[0..8], &SIGNATURE, "PNG signature");
    let mut chunks = Vec::new();
    let mut i = 8;
    while i + 8 <= png.len() {
        let len = u32::from_be_bytes([png[i], png[i + 1], png[i + 2], png[i + 3]]) as usize;
        let kind = String::from_utf8_lossy(&png[i + 4..i + 8]).into_owned();
        let data = png[i + 8..i + 8 + len].to_vec();
        let crc_at = i + 8 + len;
        let stated = u32::from_be_bytes([
            png[crc_at],
            png[crc_at + 1],
            png[crc_at + 2],
            png[crc_at + 3],
        ]);
        let mut crc_input = png[i + 4..i + 8].to_vec();
        crc_input.extend_from_slice(&data);
        assert_eq!(crc32(&crc_input), stated, "CRC of {kind} chunk");
        chunks.push((kind, data));
        i = crc_at + 4;
    }
    assert_eq!(i, png.len(), "chunk stream consumed exactly");
    chunks
}

/// Inflate a zlib stream made only of stored blocks, checking the Adler-32.
fn inflate_stored(z: &[u8]) -> Vec<u8> {
    assert_eq!(z[0], 0x78, "zlib CMF");
    assert_eq!(
        (u32::from(z[0]) * 256 + u32::from(z[1])) % 31,
        0,
        "zlib header check bits"
    );
    let mut out = Vec::new();
    let mut i = 2;
    loop {
        let header = z[i];
        assert_eq!(header & 0b110, 0, "BTYPE must be 00 (stored)");
        let len = u16::from_le_bytes([z[i + 1], z[i + 2]]) as usize;
        let nlen = u16::from_le_bytes([z[i + 3], z[i + 4]]);
        assert_eq!(!(len as u16), nlen, "NLEN is the ones-complement of LEN");
        out.extend_from_slice(&z[i + 5..i + 5 + len]);
        i += 5 + len;
        if header & 1 == 1 {
            break;
        }
    }
    let stated = u32::from_be_bytes([z[i], z[i + 1], z[i + 2], z[i + 3]]);
    assert_eq!(adler32(&out), stated, "zlib Adler-32");
    assert_eq!(i + 4, z.len(), "zlib stream consumed exactly");
    out
}

/// Decode a grayscale PNG produced by `encode_grayscale` back to (w, h, pixels).
fn decode_grayscale(png: &[u8]) -> (usize, usize, Vec<u8>) {
    let chunks = parse_chunks(png);
    let kinds: Vec<&str> = chunks.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(kinds, vec!["IHDR", "IDAT", "IEND"], "chunk order");

    let ihdr = &chunks[0].1;
    assert_eq!(ihdr.len(), 13, "IHDR length");
    let w = u32::from_be_bytes([ihdr[0], ihdr[1], ihdr[2], ihdr[3]]) as usize;
    let h = u32::from_be_bytes([ihdr[4], ihdr[5], ihdr[6], ihdr[7]]) as usize;
    assert_eq!(ihdr[8], 8, "bit depth");
    assert_eq!(ihdr[9], 0, "color type grayscale");
    assert_eq!(ihdr[10], 0, "compression method");
    assert_eq!(ihdr[11], 0, "filter method");
    assert_eq!(ihdr[12], 0, "interlace");

    let raw = inflate_stored(&chunks[1].1);
    assert_eq!(raw.len(), h * (w + 1), "scanline count");
    let mut pixels = Vec::with_capacity(w * h);
    for row in raw.chunks_exact(w + 1) {
        assert_eq!(row[0], 0, "filter type None");
        pixels.extend_from_slice(&row[1..]);
    }
    (w, h, pixels)
}

#[test]
fn encode_grayscale_round_trips_pixels() {
    let pixels: Vec<u8> = (0..(7 * 5u16)).map(|v| (v * 7 % 256) as u8).collect();
    let png = encode_grayscale(7, 5, &pixels).expect("encode");
    let (w, h, decoded) = decode_grayscale(&png);
    assert_eq!((w, h), (7, 5));
    assert_eq!(decoded, pixels, "pixel data survives the round trip");
}

#[test]
fn encode_grayscale_emits_png_magic_not_netpbm() {
    let png = encode_grayscale(2, 2, &[0, 255, 255, 0]).expect("encode");
    assert_eq!(
        &png[0..8],
        b"\x89PNG\r\n\x1a\n",
        "must start with the PNG signature, not Netpbm 'P5'"
    );
    assert_ne!(&png[0..2], b"P5", "Netpbm magic must not appear");
}

#[test]
fn encode_grayscale_spans_multiple_stored_blocks() {
    // 300x300 grayscale = 300 * 301 = 90_300 raw bytes > one 65_535 stored block.
    let w = 300;
    let h = 300;
    let pixels: Vec<u8> = (0..w * h).map(|i| (i % 251) as u8).collect();
    let png = encode_grayscale(w, h, &pixels).expect("encode");
    let (dw, dh, decoded) = decode_grayscale(&png);
    assert_eq!((dw, dh), (w, h));
    assert_eq!(decoded, pixels, "multi-block stream round trips");
}

#[test]
fn encode_grayscale_rejects_mismatched_buffer() {
    assert!(
        encode_grayscale(4, 4, &[0u8; 15]).is_none(),
        "short buffer must not produce a truncated image"
    );
    assert!(encode_grayscale(0, 4, &[]).is_none(), "zero width");
    assert!(encode_grayscale(4, 0, &[]).is_none(), "zero height");
}

#[test]
fn crc32_matches_known_vector() {
    // The IEND chunk's CRC is a fixed, published value: 0xAE426082.
    assert_eq!(crc32(b"IEND"), 0xAE42_6082);
    assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
}

#[test]
fn adler32_matches_known_vector() {
    assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    assert_eq!(adler32(b""), 1);
}
