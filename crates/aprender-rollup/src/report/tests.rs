use super::*;

#[test]
fn empty_plan_is_one_line() {
    assert_eq!(plan_text(&Selection::default()), "rollup 0 []\n");
}

#[test]
fn lists_picked_then_each_skip_with_its_reason() {
    let sel = Selection {
        heads: vec![],
        picked: vec![1, 3],
        rejected: vec![
            (
                2,
                Reject::FileOverlap {
                    with: 1,
                    file: "a.rs".to_string(),
                },
            ),
            (4, Reject::TooLarge(201)),
            (5, Reject::NotGreen),
            (6, Reject::NotApproved),
            (7, Reject::NoOptInLabel),
            (8, Reject::ForbiddenFile("Cargo.lock".to_string())),
        ],
    };
    assert_eq!(
        plan_text(&sel),
        "rollup 2 [#1 #3]\n\
         skip #2: shares a.rs with #1\n\
         skip #4: 201 lines > 200\n\
         skip #5: checks not all green\n\
         skip #6: not approved\n\
         skip #7: missing label rollup-ok\n\
         skip #8: touches Cargo.lock (CI, contract or lockfile)\n"
    );
}

#[test]
fn over_cap_is_named_in_the_plan() {
    let sel = Selection {
        heads: vec![],
        picked: vec![1],
        rejected: vec![(9, Reject::OverCap)],
    };
    assert!(plan_text(&sel).contains("skip #9: rollup already holds"));
}
