//! #3560 R2: the book walk over a planted tree, the page readers, and the repo's own book.

use super::*;
use std::fs;

fn write(root: &Path, rel: &str, text: &str) {
    let p = root.join(rel);
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(p, text).unwrap();
}

/// A book with a linked current page, a linked stale page, a pinned page, a dark page and a SUMMARY link to nothing.
fn planted() -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    let r = d.path();
    write(
        r,
        "book/src/SUMMARY.md",
        "# Summary\n\n[Intro](./intro.md)\n- [Old](ch/old.md#top)\n- [Pinned](ch/pinned.md)\n- [Gone](gone.md)\n",
    );
    write(r, "book/src/intro.md", "# Intro\n\nRun Qwen3.5-4B.\n");
    write(
        r,
        "book/src/ch/old.md",
        "# Old\n\n```\n# comment\n```\nQwen2.5-Coder\n",
    );
    write(
        r,
        "book/src/ch/pinned.md",
        "# Pinned\n<!-- ont:model-pinned: GGUF parity uses Qwen2-0.5B -->\nqwen2\n",
    );
    write(r, "book/src/dark.md", "no heading, TinyLlama\n");
    write(r, "book/src/img.png", "not a page");
    write(r, "docs/other.md", "llama\n");
    d
}

#[test]
fn a_planted_book_yields_every_page_and_its_links() {
    let d = planted();
    let (pages, stats) = walk(d.path());
    let files: Vec<&str> = pages.iter().map(|p| p.file.as_str()).collect();
    assert_eq!(
        files,
        [
            "book/src/SUMMARY.md",
            "book/src/ch/old.md",
            "book/src/ch/pinned.md",
            "book/src/dark.md",
            "book/src/intro.md"
        ]
    );
    assert_eq!(
        pages[1].title.as_deref(),
        Some("Old"),
        "a # inside a fence is not the title"
    );
    assert_eq!(pages[3].title, None);
    assert_eq!(
        pages.iter().map(|p| p.in_summary).collect::<Vec<_>>(),
        [true, true, true, false, true]
    );
    assert_eq!(
        (
            stats.pages,
            stats.summary_links,
            stats.summary_missing,
            stats.dark
        ),
        (5, 4, 1, 1)
    );
    // old names Qwen2.5 only (stale); dark names TinyLlama (stale); pinned names qwen2 but says why.
    assert_eq!((stats.naming_a_model, stats.naming_qwen35), (4, 1));
    assert_eq!((stats.stale, stats.pinned), (2, 1));
    assert!(stats.errors.is_empty(), "{:?}", stats.errors);
}

#[test]
fn a_book_that_read_nothing_is_an_error_and_a_tree_with_no_book_is_not() {
    let d = tempfile::tempdir().unwrap();
    let (_, none) = walk(d.path());
    assert!(!none.has_book && none.errors.is_empty(), "{none:?}");

    write(d.path(), "book/src/SUMMARY.md", "# Summary\n");
    let (_, empty) = walk(d.path());
    assert_eq!(
        empty.errors.len(),
        1,
        "SUMMARY alone read nothing: {empty:?}"
    );

    write(d.path(), "book/src/a.md", "# A\n");
    let (_, unlinked) = walk(d.path());
    assert_eq!(
        unlinked.errors.len(),
        1,
        "a SUMMARY linking nothing: {unlinked:?}"
    );

    write(d.path(), "book/src/SUMMARY.md", "- [A](a.md)\n");
    let (_, ok) = walk(d.path());
    assert!(ok.errors.is_empty(), "{ok:?}");
}

#[test]
fn the_page_readers_separate_pins_titles_and_links_from_look_alikes() {
    assert_eq!(
        md_pin_reason("  <!-- ont:model-pinned: tiny test model -->\n").as_deref(),
        Some("tiny test model")
    );
    assert_eq!(
        md_pin_reason("<!-- ont:model-pinned:   -->\n"),
        None,
        "no reason, no pin"
    );
    assert_eq!(
        md_pin_reason("ont:model-pinned: prose\n"),
        None,
        "not a comment"
    );
    assert_eq!(
        md_pin_reason("x <!-- ont:model-pinned: inline -->\n"),
        None,
        "a pin is a line of its own"
    );
    assert_eq!(
        title_of("~~~\n# no\n~~~\n#nospace\n# Yes\n").as_deref(),
        Some("Yes")
    );
    let links = summary_links("[a](./a.md) [b](b.md#s) [c](http://x/c.md) [d](d.png) [e]( e.md )");
    assert_eq!(
        links.into_iter().collect::<Vec<_>>(),
        ["a.md", "b.md", "e.md"]
    );
    assert!(positive_control());
}

/// The repo's own book: pages are read, SUMMARY links them, and the walk reports no error.
#[test]
fn the_repo_book_is_extracted() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (pages, stats) = walk(&root);
    assert!(stats.errors.is_empty(), "{:?}", stats.errors);
    assert!(stats.pages > 300, "only {} pages read", stats.pages);
    assert_eq!(pages.len(), stats.pages);
    assert!(
        stats.summary_links > 300,
        "{} SUMMARY links",
        stats.summary_links
    );
}

/// The drift gate: stale and dark pages are pinned shrink-only. A new page on an old model, or a page SUMMARY.md does
/// not link, raises a count and fails here; fixing one lowers it, and then the pin must be lowered with it.
#[test]
fn the_repo_stale_and_dark_pages_are_pinned_shrink_only() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (pages, stats) = walk(&root);
    let stale: Vec<&str> = pages
        .iter()
        .filter(|p| !p.model_current())
        .map(|p| p.file.as_str())
        .collect();
    let dark: Vec<&str> = pages
        .iter()
        .filter(|p| !p.in_summary)
        .map(|p| p.file.as_str())
        .collect();
    assert!(
        stats.stale <= STALE_BOOK_PAGES_PINNED,
        "{} book pages name a model other than {CURRENT_FAMILY} without `<!-- {PIN_MARKER} <reason> -->` \
         (pinned {STALE_BOOK_PAGES_PINNED}); migrate or pin the new one(s). Stale: {stale:#?}",
        stats.stale
    );
    assert!(
        stats.dark <= DARK_BOOK_PAGES_PINNED,
        "{} book pages are not linked from SUMMARY.md (pinned {DARK_BOOK_PAGES_PINNED}); link the new one(s). \
         Dark: {dark:#?}",
        stats.dark
    );
    assert_eq!(
        (stats.stale, stats.dark),
        (STALE_BOOK_PAGES_PINNED, DARK_BOOK_PAGES_PINNED),
        "a count fell: lower the pin to it (shrink-only)"
    );
}
