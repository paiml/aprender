use super::*;

// ========== PorterStemmer Tests ==========

#[test]
fn test_porter_basic_plurals() {
    let stemmer = PorterStemmer::new();
    assert_eq!(stemmer.stem("cats").expect("stem should succeed"), "cat");
    assert_eq!(stemmer.stem("ponies").expect("stem should succeed"), "poni");
    assert_eq!(
        stemmer.stem("caresses").expect("stem should succeed"),
        "caress"
    );
}

#[test]
fn test_porter_ed_ing() {
    let stemmer = PorterStemmer::new();
    assert_eq!(stemmer.stem("running").expect("stem should succeed"), "run");
    assert_eq!(stemmer.stem("jumped").expect("stem should succeed"), "jump");
    assert_eq!(
        stemmer.stem("skating").expect("stem should succeed"),
        "skate"
    );
}

#[test]
fn test_porter_y_suffix() {
    let stemmer = PorterStemmer::new();
    assert_eq!(stemmer.stem("happy").expect("stem should succeed"), "happi");
    assert_eq!(stemmer.stem("sky").expect("stem should succeed"), "sky");
}

#[test]
fn test_porter_common_words() {
    let stemmer = PorterStemmer::new();
    assert_eq!(
        stemmer.stem("studies").expect("stem should succeed"),
        "studi"
    );
    assert_eq!(
        stemmer.stem("studying").expect("stem should succeed"),
        "studi"
    );
    assert_eq!(stemmer.stem("flies").expect("stem should succeed"), "fli");
}

#[test]
fn test_porter_short_words() {
    let stemmer = PorterStemmer::new();
    assert_eq!(stemmer.stem("is").expect("stem should succeed"), "is");
    assert_eq!(stemmer.stem("as").expect("stem should succeed"), "as");
    assert_eq!(stemmer.stem("at").expect("stem should succeed"), "at");
}

#[test]
fn test_porter_empty_string() {
    let stemmer = PorterStemmer::new();
    assert_eq!(stemmer.stem("").expect("stem should succeed"), "");
}

#[test]
fn test_porter_uppercase() {
    let stemmer = PorterStemmer::new();
    // Porter stemmer converts to lowercase
    assert_eq!(stemmer.stem("RUNNING").expect("stem should succeed"), "run");
    assert_eq!(stemmer.stem("Cats").expect("stem should succeed"), "cat");
}

#[test]
fn test_porter_technical_words() {
    let stemmer = PorterStemmer::new();
    assert_eq!(
        stemmer.stem("computational").expect("stem should succeed"),
        "comput"
    );
    assert_eq!(
        stemmer.stem("relational").expect("stem should succeed"),
        "rel"
    );
}

#[test]
fn test_stem_tokens() {
    let stemmer = PorterStemmer::new();
    let words = vec!["running", "cats", "easily"];
    let stemmed = stemmer
        .stem_tokens(&words)
        .expect("stem_tokens should succeed");
    assert_eq!(stemmed, vec!["run", "cat", "easili"]);
}

#[test]
fn test_stem_tokens_empty() {
    let stemmer = PorterStemmer::new();
    let words: Vec<&str> = vec![];
    let stemmed = stemmer
        .stem_tokens(&words)
        .expect("stem_tokens should succeed");
    assert_eq!(stemmed, Vec::<String>::new());
}

#[test]
fn test_stem_tokens_mixed() {
    let stemmer = PorterStemmer::new();
    let words = vec!["machine", "learning", "algorithms", "are", "powerful"];
    let stemmed = stemmer
        .stem_tokens(&words)
        .expect("stem_tokens should succeed");
    assert_eq!(stemmed, vec!["machin", "learn", "algorithm", "ar", "pow"]);
}

#[test]
fn test_porter_default() {
    let stemmer = PorterStemmer;
    assert_eq!(stemmer.stem("running").expect("stem should succeed"), "run");
}

// ========== Helper Function Tests ==========

#[test]
fn test_is_vowel() {
    assert!(PorterStemmer::is_vowel('a'));
    assert!(PorterStemmer::is_vowel('e'));
    assert!(PorterStemmer::is_vowel('i'));
    assert!(PorterStemmer::is_vowel('o'));
    assert!(PorterStemmer::is_vowel('u'));
    assert!(!PorterStemmer::is_vowel('b'));
    assert!(!PorterStemmer::is_vowel('z'));
}

#[test]
fn test_measure() {
    assert_eq!(PorterStemmer::measure("tree"), 0);
    assert_eq!(PorterStemmer::measure("trees"), 1);
    assert_eq!(PorterStemmer::measure("trouble"), 1);
    assert_eq!(PorterStemmer::measure("troubles"), 2);
}

#[test]
fn test_ends_with_double_consonant() {
    assert!(PorterStemmer::ends_with_double_consonant("hopp"));
    assert!(PorterStemmer::ends_with_double_consonant("hiss"));
    assert!(!PorterStemmer::ends_with_double_consonant("hope"));
    assert!(!PorterStemmer::ends_with_double_consonant("hi"));
}

#[test]
fn test_ends_with_cvc() {
    assert!(PorterStemmer::ends_with_cvc("hop"));
    assert!(!PorterStemmer::ends_with_cvc("hoop"));
    assert!(!PorterStemmer::ends_with_cvc("hi"));
}

#[test]
fn test_porter_step4_suffixes() {
    let stemmer = PorterStemmer::new();
    // Test various -ance, -ence, -er, -ic, -able, -ible, -ant suffixes
    assert_eq!(stemmer.stem("reliance").expect("stem"), "reli");
    assert_eq!(stemmer.stem("reference").expect("stem"), "refer");
    assert_eq!(stemmer.stem("trainer").expect("stem"), "train");
    assert_eq!(stemmer.stem("electric").expect("stem"), "electr");
    assert_eq!(stemmer.stem("adjustable").expect("stem"), "adjust");
    assert_eq!(stemmer.stem("defensible").expect("stem"), "defens");
    assert_eq!(stemmer.stem("irritant").expect("stem"), "irrit");
}

#[test]
fn test_porter_step4_more_suffixes() {
    let stemmer = PorterStemmer::new();
    // Test -ement, -ment, -ent, -ion, -ism, -ate, -iti, -ous, -ive, -ize
    assert_eq!(stemmer.stem("replacement").expect("stem"), "replac");
    assert_eq!(stemmer.stem("adjustment").expect("stem"), "adjust");
    assert_eq!(stemmer.stem("dependent").expect("stem"), "depend");
    assert_eq!(stemmer.stem("adoption").expect("stem"), "adopt");
    assert_eq!(stemmer.stem("communism").expect("stem"), "commun");
    assert_eq!(stemmer.stem("activate").expect("stem"), "activ");
    assert_eq!(stemmer.stem("angulariti").expect("stem"), "angular");
    assert_eq!(stemmer.stem("homologous").expect("stem"), "homolog");
    assert_eq!(stemmer.stem("effective").expect("stem"), "effect");
    assert_eq!(stemmer.stem("bowdlerize").expect("stem"), "bowdler");
}
