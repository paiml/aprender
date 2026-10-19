//! Public-API edge cases for the .ald dataset format (coverage ticket, Rule 7).
//! Contract: FALSIFY-BIN-ALIMENTAR-008.

use std::sync::Arc;

use alimentar::format::{
    flags, load, save, Compression, DatasetType, Header, Metadata, SaveOptions,
    FORMAT_VERSION_MAJOR, HEADER_SIZE, MAGIC,
};
use arrow::{
    array::{Int32Array, RecordBatch, StringArray},
    datatypes::{DataType, Field, Schema},
};

fn batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
        ],
    )
    .expect("batch")
}

fn saved(opts: &SaveOptions) -> Vec<u8> {
    let mut buf = Vec::new();
    save(&mut buf, &[batch()], DatasetType::Tabular, opts).expect("save");
    buf
}

#[test]
fn dataset_type_codes_roundtrip_and_unknown_is_none() {
    let all = [
        DatasetType::Tabular,
        DatasetType::TimeSeries,
        DatasetType::Graph,
        DatasetType::Spatial,
        DatasetType::TextCorpus,
        DatasetType::TextClassification,
        DatasetType::TextPairs,
        DatasetType::SequenceLabeling,
        DatasetType::QuestionAnswering,
        DatasetType::Summarization,
        DatasetType::Translation,
        DatasetType::ImageClassification,
        DatasetType::ObjectDetection,
        DatasetType::Segmentation,
        DatasetType::ImagePairs,
        DatasetType::Video,
        DatasetType::AudioClassification,
        DatasetType::SpeechRecognition,
        DatasetType::SpeakerIdentification,
        DatasetType::UserItemRatings,
        DatasetType::ImplicitFeedback,
        DatasetType::SequentialRecs,
        DatasetType::ImageText,
        DatasetType::AudioText,
        DatasetType::VideoText,
        DatasetType::Custom,
    ];
    for t in all {
        assert_eq!(DatasetType::from_u16(t.as_u16()), Some(t), "{t:?}");
    }
    for bad in [
        0x0000, 0x0005, 0x0017, 0x0033, 0x0043, 0x0053, 0x0100, 0xFFFF,
    ] {
        assert_eq!(DatasetType::from_u16(bad), None, "0x{bad:04X}");
    }
    assert_eq!(DatasetType::Tabular.as_u16(), 1);
    assert_eq!(DatasetType::Custom.as_u16(), 0xFF);
}

#[test]
fn compression_codes_roundtrip_and_unknown_is_none() {
    for (c, code) in [
        (Compression::None, 0u8),
        (Compression::ZstdL3, 1),
        (Compression::ZstdL19, 2),
        (Compression::Lz4, 3),
    ] {
        assert_eq!(c.as_u8(), code);
        assert_eq!(Compression::from_u8(code), Some(c));
    }
    assert_eq!(Compression::from_u8(4), None);
    assert_eq!(Compression::from_u8(0xFF), None);
    assert_eq!(Compression::default(), Compression::ZstdL3);
}

#[test]
fn header_roundtrips_every_field() {
    let mut h = Header::new(DatasetType::Graph);
    h.metadata_size = 11;
    h.payload_size = 22;
    h.uncompressed_size = 33;
    h.compression = Compression::Lz4;
    h.flags = flags::SIGNED | flags::STREAMING;
    h.schema_size = 44;
    h.num_rows = 5_000_000_000;
    let bytes = h.to_bytes();
    assert_eq!(bytes.len(), HEADER_SIZE);
    assert_eq!(bytes[0..4], MAGIC);
    let back = Header::from_bytes(&bytes).expect("parse");
    assert_eq!(back.dataset_type, DatasetType::Graph);
    assert_eq!(
        (
            back.metadata_size,
            back.payload_size,
            back.uncompressed_size,
            back.schema_size
        ),
        (11, 22, 33, 44)
    );
    assert_eq!(back.compression, Compression::Lz4);
    assert_eq!(back.num_rows, 5_000_000_000);
    assert!(back.is_signed() && back.is_streaming());
    assert!(!back.is_encrypted() && !back.is_licensed() && !back.is_trueno_native());
}

#[test]
fn header_flag_accessors_are_independent() {
    for (bit, check) in [
        (
            flags::ENCRYPTED,
            Header::is_encrypted as fn(&Header) -> bool,
        ),
        (flags::SIGNED, Header::is_signed),
        (flags::STREAMING, Header::is_streaming),
        (flags::LICENSED, Header::is_licensed),
        (flags::TRUENO_NATIVE, Header::is_trueno_native),
    ] {
        let mut h = Header::new(DatasetType::Tabular);
        assert!(!check(&h));
        h.flags = bit;
        assert!(check(&h), "bit {bit:#b}");
    }
}

#[test]
fn header_rejects_malformed_input() {
    let good = Header::new(DatasetType::Tabular).to_bytes();

    assert!(
        Header::from_bytes(&good[..HEADER_SIZE - 1]).is_err(),
        "short"
    );
    assert!(Header::from_bytes(&[]).is_err(), "empty");

    let mut bad_magic = good;
    bad_magic[0] ^= 0xFF;
    assert!(Header::from_bytes(&bad_magic).is_err(), "magic");

    let mut bad_type = good;
    bad_type[6] = 0xEE;
    bad_type[7] = 0xEE;
    let e = Header::from_bytes(&bad_type).unwrap_err().to_string();
    assert!(e.contains("Unknown dataset type"), "{e}");

    let mut bad_comp = good;
    bad_comp[20] = 0x7F;
    let e = Header::from_bytes(&bad_comp).unwrap_err().to_string();
    assert!(e.contains("Unknown compression"), "{e}");

    let mut future = good;
    future[4] = FORMAT_VERSION_MAJOR + 1;
    assert!(Header::from_bytes(&future).is_err(), "future major version");
}

#[test]
fn save_rejects_empty_batch_list() {
    let mut buf = Vec::new();
    assert!(save(&mut buf, &[], DatasetType::Tabular, &SaveOptions::default()).is_err());
    assert!(buf.is_empty(), "nothing written on error");
}

#[test]
fn every_compression_roundtrips_rows_and_values() {
    for c in [
        Compression::None,
        Compression::ZstdL3,
        Compression::ZstdL19,
        Compression::Lz4,
    ] {
        let bytes = saved(&SaveOptions::default().with_compression(c));
        let loaded = load(&mut bytes.as_slice()).unwrap_or_else(|e| panic!("{c:?}: {e}"));
        assert_eq!(loaded.header.compression, c);
        assert_eq!(loaded.header.dataset_type, DatasetType::Tabular);
        assert_eq!(loaded.header.num_rows, 3);
        assert_eq!(loaded.batches.len(), 1);
        let ids = loaded.batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("int32");
        assert_eq!(ids.values().to_vec(), vec![1, 2, 3], "{c:?}");
        assert_eq!(loaded.batches[0].column(1).null_count(), 1, "{c:?}");
    }
}

#[test]
fn metadata_roundtrips() {
    let meta = Metadata {
        name: Some("demo".into()),
        tags: vec!["x".into(), "y".into()],
        ..Metadata::default()
    };
    let bytes = saved(&SaveOptions::default().with_metadata(meta));
    let loaded = load(&mut bytes.as_slice()).expect("load");
    assert_eq!(loaded.metadata.name.as_deref(), Some("demo"));
    assert_eq!(loaded.metadata.tags, vec!["x", "y"]);
}

#[test]
fn corrupted_or_truncated_file_is_rejected() {
    let bytes = saved(&SaveOptions::default());
    let mut flipped = bytes.clone();
    let mid = flipped.len() / 2;
    flipped[mid] ^= 0xFF;
    assert!(
        load(&mut flipped.as_slice()).is_err(),
        "bit flip must fail the checksum"
    );

    let truncated = &bytes[..bytes.len() - 6];
    assert!(load(&mut &truncated[..]).is_err(), "truncation");
    assert!(load(&mut &bytes[..10]).is_err(), "header-only fragment");
    assert!(load(&mut &[][..]).is_err(), "empty input");
}
