// usage: iqdec <gguf> <offset> <nbytes> <IQ4_NL|IQ4_XS|IQ3_S> <out.f32> [flip_byte_index]
use std::io::{Read, Seek, SeekFrom, Write};
use realizar::quantize::{iq3_s::dequantize_iq3_s, iq4_nl::dequantize_iq4_nl, iq4_xs::dequantize_iq4_xs};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (off, n): (u64, usize) = (a[2].parse().unwrap(), a[3].parse().unwrap());
    let mut f = std::fs::File::open(&a[1]).unwrap();
    f.seek(SeekFrom::Start(off)).unwrap();
    let mut buf = vec![0u8; n];
    f.read_exact(&mut buf).unwrap();
    if let Some(i) = a.get(6) { let i: usize = i.parse().unwrap(); buf[i] ^= 0x01; }
    let out = match a[4].as_str() {
        "IQ4_NL" => dequantize_iq4_nl(&buf), "IQ4_XS" => dequantize_iq4_xs(&buf), "IQ3_S" => dequantize_iq3_s(&buf),
        t => panic!("type {t}"),
    }.expect("decode");
    let mut o = std::fs::File::create(&a[5]).unwrap();
    let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    o.write_all(&bytes).unwrap();
}
