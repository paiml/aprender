//! CHARACTERIZATION — compute's one id boundary, snapshotted before PMAT-3430
//! Phases 3-4 re-export `trueno_quant::GgmlType` here.
//!
//! Written against UNCHANGED code; must pass UNCHANGED afterwards, with ONE
//! permitted edit named in advance by #3430 Q1-c: `GgmlType::from_u32(x)`
//! becomes the free function `from_u32(x)`, because an inherent method cannot
//! survive on a re-exported foreign type. The admitted SET may not change.
//!
//! Sizes are characterized here too. compute is the only one of the three
//! enums that carried block geometry, and it is the one the upstream extraction
//! agreed with exactly — so these 15 rows are the evidence that delegating to
//! `TRAITS` is value-identical rather than merely plausible.

use super::GgmlType;

/// The 15 ids compute admitted before M1, with their block geometry.
/// (id, blck_size, type_size)
const ADMITTED_TODAY: [(u32, usize, usize); 15] = [
    (0, 1, 4),
    (1, 1, 2),
    (2, 32, 18),
    (3, 32, 20),
    (6, 32, 22),
    (7, 32, 24),
    (8, 32, 34),
    (9, 32, 36),
    (10, 256, 84),
    (11, 256, 110),
    (12, 256, 144),
    (13, 256, 176),
    (14, 256, 210),
    (15, 256, 292),
    (30, 1, 2),
];

fn admitted(id: u32) -> Option<(usize, usize)> {
    ADMITTED_TODAY
        .iter()
        .find(|(candidate, _, _)| *candidate == id)
        .map(|(_, blck, size)| (*blck, *size))
}

#[test]
fn from_u32_admits_exactly_fifteen_ids_over_the_whole_byte_range() {
    for id in 0u32..=255 {
        let got = GgmlType::from_u32(id);
        assert_eq!(
            got.is_some(),
            admitted(id).is_some(),
            "qtype {id}: compute's admitted set changed"
        );
        if let Some(t) = got {
            assert_eq!(t as u32, id, "qtype {id} no longer round-trips through the enum");
        }
    }
}

#[test]
fn block_geometry_is_unchanged_for_every_admitted_id() {
    for (id, blck, size) in ADMITTED_TODAY {
        let t = GgmlType::from_u32(id).expect("an admitted id must construct");
        assert_eq!(t.block_size(), blck, "qtype {id} block_size changed");
        assert_eq!(t.block_bytes(), size, "qtype {id} block_bytes changed");
    }
}

#[test]
fn tensor_bytes_rounds_partial_blocks_up_exactly_as_it_did() {
    for (id, blck, size) in ADMITTED_TODAY {
        let t = GgmlType::from_u32(id).expect("an admitted id must construct");
        assert_eq!(t.tensor_bytes(0), 0, "qtype {id} at 0 elements");
        assert_eq!(t.tensor_bytes(blck), size, "qtype {id} at one full block");
        assert_eq!(t.tensor_bytes(blck * 4), size * 4, "qtype {id} at four blocks");
        if blck > 1 {
            assert_eq!(t.tensor_bytes(1), size, "qtype {id} rounds 1 element up to a block");
            assert_eq!(
                t.tensor_bytes(blck + 1),
                size * 2,
                "qtype {id} rounds a partial second block up"
            );
        }
    }
}
