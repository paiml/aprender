//! Tests for the prefix search: case folding, rank order, cell-id tie-break, dedupe, limits.

use super::*;

fn index(entries: &[(u32, &str)]) -> SearchIndex {
    SearchIndex::new(entries).expect("index")
}

#[test]
fn prefix_match_ignores_case_in_both_directions_and_is_a_prefix_not_a_substring() {
    let idx = index(&[(3, "Alaska"), (7, "alder"), (9, "Boston")]);
    assert_eq!(idx.prefix("ALA", 10), vec![3]);
    assert_eq!(idx.prefix("aLd", 10), vec![7]);
    assert_eq!(idx.prefix("bos", 10), vec![9]);
    assert_eq!(
        idx.prefix("ska", 10),
        Vec::<u32>::new(),
        "substring is not a prefix"
    );
}

#[test]
fn case_folding_covers_non_ascii_letters() {
    let idx = index(&[(1, "Ünterberg"), (2, "Zürich")]);
    assert_eq!(idx.prefix("ün", 10), vec![1]);
    assert_eq!(idx.prefix("zÜRICH", 10), vec![2]);
}

#[test]
fn exact_match_ranks_first_then_shorter_name_then_lower_cell_id() {
    let idx = index(&[(1, "Ohio State"), (2, "Ohio"), (3, "Oh"), (4, "Oh")]);
    // "oh": exact matches 3 and 4 first (tie by id), then "ohio" (4 chars), then "ohio state".
    assert_eq!(idx.prefix("oh", 10), vec![3, 4, 2, 1]);
    // "ohio": exact "ohio" (cell 2) outranks the longer "ohio state" (cell 1).
    assert_eq!(idx.prefix("ohio", 10), vec![2, 1]);
}

#[test]
fn ties_break_by_cell_id_whatever_the_insertion_order() {
    let forward = index(&[(9, "Maine"), (2, "Mainz"), (5, "Main")]);
    let backward = index(&[(5, "Main"), (2, "Mainz"), (9, "Maine")]);
    // Exact "main" (5) first; then the two 5-letter names by cell id: 2 then 9.
    assert_eq!(forward.prefix("main", 10), vec![5, 2, 9]);
    assert_eq!(backward.prefix("main", 10), vec![5, 2, 9]);
    let same_name = index(&[(9, "Park"), (2, "Park")]);
    assert_eq!(same_name.prefix("park", 10), vec![2, 9]);
}

#[test]
fn a_cell_matched_under_several_names_is_reported_once_at_its_best_rank() {
    let idx = index(&[(4, "Lakeside"), (4, "Lake"), (6, "Lakes")]);
    // Cell 4 is exact for "lake" through its second name, so it leads and appears once.
    assert_eq!(idx.prefix("lake", 10), vec![4, 6]);
}

#[test]
fn limit_truncates_and_empty_query_or_zero_limit_return_nothing() {
    let idx = index(&[
        (1, "Ada"),
        (2, "Aden"),
        (3, "Adams"),
        (4, "Adair"),
        (5, "Adan"),
    ]);
    // Character counts: Ada 3, Aden 4, Adan 4, Adams 5, Adair 5. No name is exact for "ad", so
    // rank is (length, cell id): 1; then 2 and 5; then 3 and 4.
    assert_eq!(idx.prefix("ad", 3), vec![1, 2, 5]);
    assert_eq!(idx.prefix("ad", 10), vec![1, 2, 5, 3, 4]);
    assert_eq!(idx.prefix("a", 0), Vec::<u32>::new());
    assert_eq!(idx.prefix("", 10), Vec::<u32>::new());
    assert_eq!(idx.prefix("zzz", 10), Vec::<u32>::new());
}

#[test]
fn an_empty_feature_name_is_refused() {
    assert_eq!(
        SearchIndex::new(&[(1, "")]),
        Err(MapError::Malformed("feature name"))
    );
}

#[test]
fn an_index_with_no_features_matches_nothing() {
    let idx = index(&[]);
    assert_eq!(idx.prefix("a", 10), Vec::<u32>::new());
}
