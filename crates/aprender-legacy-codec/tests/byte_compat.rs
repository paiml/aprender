//! FALSIFY-LEGACY-CODEC-001..004: the codec is byte-identical to bincode 1.3.3.
//! `fixtures/sample_bincode_1_3_3.hex` was written by `bincode = "=1.3.3"`
//! (`bincode::serialize`) from the value built in `sample()`; bincode is NOT a dependency here.
use aprender_legacy_codec::{deserialize, deserialize_from, serialize, serialize_into, Error};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub enum Kind {
    Unit,
    New(u16),
    Tup(i8, String),
    Rec { a: f32, b: Option<u64> },
}
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct Wrap(pub u32);
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct Unit;
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct Sample {
    pub flag: bool,
    pub i8_: i8,
    pub i16_: i16,
    pub i32_: i32,
    pub i64_: i64,
    pub i128_: i128,
    pub u8_: u8,
    pub u16_: u16,
    pub u32_: u32,
    pub u64_: u64,
    pub u128_: u128,
    pub f32_: f32,
    pub f64_: f64,
    pub ch: char,
    pub text: String,
    pub bytes: Vec<u8>,
    pub none: Option<u32>,
    pub some: Option<String>,
    pub list: Vec<Kind>,
    pub map: BTreeMap<String, Vec<f32>>,
    pub tuple: (u8, u16, String),
    pub arr: [u16; 3],
    pub wrap: Wrap,
    pub unit: Unit,
    pub empty: (),
    pub nested: Vec<Option<Vec<i64>>>,
}

fn sample() -> Sample {
    Sample {
        flag: true,
        i8_: -5,
        i16_: -300,
        i32_: -70000,
        i64_: -5_000_000_000,
        i128_: -170141183460469231731687303715884105,
        u8_: 200,
        u16_: 65000,
        u32_: 4_000_000_000,
        u64_: 18_000_000_000_000_000_000,
        u128_: 340282366920938463463374607431768211455,
        f32_: 1.5,
        f64_: -2.25e10,
        ch: 'é',
        text: "héllo €🙂".into(),
        bytes: vec![0, 1, 254, 255],
        none: None,
        some: Some("x".into()),
        list: vec![
            Kind::Unit,
            Kind::New(7),
            Kind::Tup(-1, "t".into()),
            Kind::Rec { a: 0.5, b: Some(9) },
            Kind::Rec { a: -0.0, b: None },
        ],
        map: [("a".to_string(), vec![1.0, 2.0]), ("b".to_string(), vec![])]
            .into_iter()
            .collect(),
        tuple: (1, 2, "three".into()),
        arr: [10, 20, 30],
        wrap: Wrap(42),
        unit: Unit,
        empty: (),
        nested: vec![None, Some(vec![]), Some(vec![-1, 2])],
    }
}

fn fixture() -> Vec<u8> {
    let hex = include_str!("fixtures/sample_bincode_1_3_3.hex").trim();
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("codec"))
        .collect()
}

#[test]
fn writes_the_exact_bytes_bincode_1_3_3_wrote() {
    assert_eq!(serialize(&sample()).expect("codec"), fixture());
}

#[test]
fn reads_the_bincode_1_3_3_fixture_back_to_the_same_value() {
    assert_eq!(deserialize::<Sample>(&fixture()).expect("codec"), sample());
}

#[test]
fn stream_api_matches_slice_api_and_consumes_only_what_it_needs() {
    let mut buf = Vec::new();
    serialize_into(&mut buf, &sample()).expect("codec");
    assert_eq!(buf, fixture());
    buf.extend_from_slice(&[0xAA, 0xBB]);
    let mut rd = buf.as_slice();
    let v: Sample = deserialize_from(&mut rd).expect("codec");
    assert_eq!(v, sample());
    assert_eq!(rd, &[0xAA, 0xBB], "reader must stop after the value");
}

#[test]
fn hand_checked_primitives_pin_the_layout() {
    assert_eq!(serialize(&1u32).expect("codec"), [1, 0, 0, 0]);
    assert_eq!(
        serialize("ab").expect("codec"),
        [2, 0, 0, 0, 0, 0, 0, 0, b'a', b'b']
    );
    assert_eq!(serialize(&Some(7u8)).expect("codec"), [1, 7]);
    assert_eq!(
        serialize(&Kind::New(0x0102)).expect("codec"),
        [1, 0, 0, 0, 2, 1]
    );
}

#[test]
fn corrupt_input_is_an_error_not_a_panic_or_a_huge_allocation() {
    // bool=2, Option tag=2, string length 2^60 on 8 bytes of input, truncated u32, bad UTF-8.
    assert!(matches!(
        deserialize::<bool>(&[2]),
        Err(Error::InvalidBool(2))
    ));
    assert!(matches!(
        deserialize::<Option<u8>>(&[2]),
        Err(Error::InvalidOptionTag(2))
    ));
    assert!(deserialize::<String>(&(1u64 << 60).to_le_bytes()).is_err());
    assert!(deserialize::<Vec<u64>>(&(1u64 << 60).to_le_bytes()).is_err());
    assert!(deserialize::<u32>(&[1, 2]).is_err());
    assert!(matches!(
        deserialize::<String>(&[1, 0, 0, 0, 0, 0, 0, 0, 0xFF]),
        Err(Error::InvalidUtf8)
    ));
}

// FALSIFY-LEGACY-CODEC-005: an untrusted length prefix must not reach a Visitor as a huge
// size_hint (a visitor that pre-allocates from it would abort on a corrupt length).
mod size_hint_cap {
    use aprender_legacy_codec::{deserialize, Error};
    use serde::de::{Deserialize, Deserializer, SeqAccess, Visitor};
    use std::cell::Cell;
    use std::fmt;

    thread_local!(static HINT: Cell<Option<usize>> = const { Cell::new(None) });

    struct Probe;
    impl<'de> Deserialize<'de> for Probe {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            struct V;
            impl<'de> Visitor<'de> for V {
                type Value = Probe;
                fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("seq")
                }
                fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Probe, A::Error> {
                    HINT.with(|h| h.set(a.size_hint()));
                    while a.next_element::<u8>()?.is_some() {}
                    Ok(Probe)
                }
            }
            d.deserialize_seq(V)
        }
    }

    #[test]
    fn corrupt_huge_length_is_capped_and_errors_cleanly() {
        let mut bytes = u64::MAX.to_le_bytes().to_vec(); // corrupt length prefix, no payload
        bytes.extend_from_slice(&[1, 2, 3]);
        let r: Result<Probe, Error> = deserialize(&bytes);
        assert!(r.is_err(), "truncated payload must error, not succeed");
        let hint = HINT.with(Cell::get).expect("visitor saw a size_hint");
        assert!(hint <= 4096, "size_hint {hint} leaks the untrusted length");
    }

    #[test]
    fn honest_small_length_hint_is_exact() {
        let bytes = [3u8, 0, 0, 0, 0, 0, 0, 0, 7, 8, 9];
        assert!(deserialize::<Probe>(&bytes).is_ok());
        assert_eq!(HINT.with(Cell::get), Some(3));
    }
}
