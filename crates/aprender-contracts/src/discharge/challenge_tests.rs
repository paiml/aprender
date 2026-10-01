use super::*;

fn st(src: &str, line: usize, kw: &str) -> Option<(String, String)> {
    statement(&lex::blank(src), line, kw)
}

#[test]
fn the_statement_runs_from_the_name_to_the_first_top_level_colon_eq() {
    let src = "theorem foo (x : Nat) (h : x = 1 := by rfl)\n    {y : Nat} :\n  x + y = y + x := by\n  omega\n";
    assert_eq!(
        st(src, 1, "theorem"),
        Some((
            String::new(),
            "(x : Nat) (h : x = 1 := by rfl) {y : Nat} : x + y = y + x".into()
        ))
    );
}

#[test]
fn a_statement_keeps_universes_dotted_names_and_attributes_are_skipped() {
    let src = "@[simp] theorem Foo.bar.{u} {α : Type u} (a : α) : a = a := rfl\n";
    assert_eq!(
        st(src, 1, "theorem"),
        Some((".{u}".into(), "{α : Type u} (a : α) : a = a".into()))
    );
}

#[test]
fn comments_never_reach_the_statement() {
    let src = "theorem foo -- a := trap\n  : True /- := -/ ∧ True := ⟨trivial, trivial⟩\n";
    assert_eq!(
        st(src, 1, "theorem"),
        Some((String::new(), ": True ∧ True".into()))
    );
}

#[test]
fn an_equation_style_statement_ends_at_its_first_arm_and_a_line_opening_abs_does_not() {
    assert_eq!(
        st(
            "theorem f : ∀ n : Nat,\n    n = n\n  | 0 => rfl\n  | _ => rfl\n",
            1,
            "theorem"
        ),
        Some((String::new(), ": ∀ n : Nat, n = n".into()))
    );
    assert_eq!(
        st(
            "theorem r (x u : ℝ) (h : u > 0) :\n    |g u x - x| ≤ u / 2 := by\n  simp\n",
            1,
            "theorem"
        ),
        Some((
            String::new(),
            "(x u : ℝ) (h : u > 0) : |g u x - x| ≤ u / 2".into()
        ))
    );
}

#[test]
fn a_runaway_statement_is_unrestated() {
    assert_eq!(st("theorem f : True\ndef g := 1\n", 1, "theorem"), None);
}

#[test]
fn the_preamble_is_the_scope_still_open_in_frames() {
    let src = "import X\nopen A\nuniverse u\nnamespace N\nopen B\nsection S\nopen C\nvariable (x : Nat)\n  (y : Nat)\nend S\nnoncomputable section\nset_option maxRecDepth 4000\nopen D in\ntheorem t : True := trivial\n";
    let scope = preamble(&lex::blank(src), 14);
    assert_eq!(scope.top, vec!["open A".to_string(), "universe u".into()]);
    let frames: Vec<(&str, &str, Vec<String>)> = scope
        .frames
        .iter()
        .map(|f| (f.opener.as_str(), f.closer.as_str(), f.cmds.clone()))
        .collect();
    assert_eq!(
        frames,
        vec![
            ("namespace N", "end N", vec!["open B".to_string()]),
            (
                "noncomputable section",
                "end",
                vec!["set_option maxRecDepth 4000".to_string()]
            ),
        ],
        "the closed `section S` and the `open D in` leave nothing behind"
    );
}

/// A fixture tree: `gelu_bound` bound by exact name, `zz` bound by nothing.
fn fixture() -> tempfile::TempDir {
    let d = tempfile::tempdir().expect("tempdir");
    let w = |rel: &str, text: &str| {
        let p = d.path().join(rel);
        std::fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
        std::fs::write(p, text).expect("write");
    };
    w(
        "lean/ProvableContracts.lean",
        "import ProvableContracts.Theorems.Gelu.Bound\n",
    );
    w(
        "lean/ProvableContracts/Theorems/Gelu/Bound.lean",
        "open Real\nnamespace ProvableContracts.Gelu\n\
         theorem gelu_bound (x : Nat) : x ≤ x + 1 := Nat.le_succ x\n\
         theorem zz : True := trivial\n\
         end ProvableContracts.Gelu\n",
    );
    w(
        "contracts/gelu-v1.yaml",
        "equations:\n  e:\n    lean_theorem: ProvableContracts.Gelu.gelu_bound\n",
    );
    d
}

#[test]
fn render_restates_each_bound_root_under_pvl_challenge() {
    let d = fixture();
    let r = render(&d.path().join("lean"), &d.path().join("contracts")).expect("render");
    assert!(r.unrestated.is_empty(), "{:?}", r.unrestated);
    let text = &r.files["Challenge/gelu-v1.lean"];
    assert!(
        text.starts_with("import ProvableContracts.Theorems.Gelu.Bound\n"),
        "{text}"
    );
    assert!(
        text.contains(
            "section\n\
             open Real\n\
             namespace ProvableContracts.Gelu\n\
             theorem _root_.PvlChallenge.ProvableContracts.Gelu.gelu_bound (x : Nat) : x ≤ x + 1 := sorry\n\
             end ProvableContracts.Gelu\n\
             end\n"
        ),
        "{text}"
    );
    assert!(!text.contains("zz"), "an unbound theorem is not restated");
    assert_eq!(challenge_decl("A.b"), "PvlChallenge.A.b");
}

/// EV-7a's mutation: a hand-edited statement is a difference; so are extra and missing files. `write` makes the
/// directory fresh again and removes the extra.
#[test]
fn diff_names_edits_extras_and_missing_and_write_clears_them() {
    let d = fixture();
    let lean = d.path().join("lean");
    let r = render(&lean, &d.path().join("contracts")).expect("render");
    assert_eq!(
        diff(&lean, &r),
        vec!["missing  Challenge/gelu-v1.lean".to_string()]
    );
    write(&lean, &r).expect("write");
    assert!(diff(&lean, &r).is_empty());
    let p = lean.join("Challenge/gelu-v1.lean");
    let text = std::fs::read_to_string(&p).expect("read");
    std::fs::write(&p, text.replace("x ≤ x + 1", "x ≤ x + 2")).expect("edit");
    std::fs::write(lean.join("Challenge/stray.lean"), "").expect("stray");
    let lines = diff(&lean, &r);
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert!(lines[0].starts_with("extra    Challenge/stray.lean"));
    // The reported line is the first one that differs (kills `!=` -> `==` in the position search).
    let edited = 1 + text
        .lines()
        .position(|l| l.contains("x ≤ x + 1"))
        .expect("the fixture states x ≤ x + 1");
    assert!(edited > 1, "the edit must not be on the first line");
    assert!(
        lines[1].starts_with(&format!("differs  Challenge/gelu-v1.lean:{edited}:")),
        "{lines:?}"
    );
    write(&lean, &r).expect("rewrite");
    assert!(diff(&lean, &r).is_empty());
    assert!(!lean.join("Challenge/stray.lean").exists());
}

#[test]
fn a_root_whose_statement_cannot_be_lifted_is_unrestated_not_dropped() {
    let d = fixture();
    std::fs::write(
        d.path()
            .join("lean/ProvableContracts/Theorems/Gelu/Bound.lean"),
        "namespace ProvableContracts.Gelu\n\
         theorem gelu_bound : True\n\
         def g := 1\n\
         end ProvableContracts.Gelu\n",
    )
    .expect("write");
    let r = render(&d.path().join("lean"), &d.path().join("contracts")).expect("render");
    assert!(r.files.is_empty());
    assert_eq!(r.unrestated.len(), 1);
    assert_eq!(r.unrestated[0].1, "ProvableContracts.Gelu.gelu_bound");
}

/// #4244: `SiluAsymptotic` and `SiluLowerBound` both declare `ProvableContracts.Sigmoid.sigmoid_lt_exp` (Lean accepts
/// the identical duplicate on import), so the binding held two roots with one fqn and `silu-kernel-v1.lean` declared
/// the challenge twice -- Lean rejected the file. One restatement per fqn.
#[test]
fn a_theorem_declared_in_two_modules_is_restated_once() {
    let d = fixture();
    std::fs::write(
        d.path().join("lean/ProvableContracts.lean"),
        "import ProvableContracts.Theorems.Gelu.Bound\nimport ProvableContracts.Theorems.Gelu.Again\n",
    )
    .expect("write root");
    std::fs::write(
        d.path()
            .join("lean/ProvableContracts/Theorems/Gelu/Again.lean"),
        "namespace ProvableContracts.Gelu\n\
         theorem gelu_bound (x : Nat) : x ≤ x + 1 := Nat.le_succ x\n\
         end ProvableContracts.Gelu\n",
    )
    .expect("write twin");
    // silu-kernel-v1 binds by module prefix (`lean_theorem: Theorems.Sigmoid`), which reaches both declarations.
    std::fs::write(
        d.path().join("contracts/gelu-v1.yaml"),
        "equations:\n  e:\n    lean_theorem: Theorems.Gelu\n",
    )
    .expect("write contract");
    let r = render(&d.path().join("lean"), &d.path().join("contracts")).expect("render");
    let text = &r.files["Challenge/gelu-v1.lean"];
    assert_eq!(
        text.matches("theorem _root_.PvlChallenge.ProvableContracts.Gelu.gelu_bound")
            .count(),
        1,
        "{text}"
    );
}

/// #4502: attention-kernel-v1 and attention-scaling-v1 both bind `attention_weight_pos`, and each file declared
/// `PvlChallenge.…attention_weight_pos` -- the comparator saw 132 DUPLICATE rows across 87 names. One declaration
/// across the whole directory, in the first contract that binds the root.
#[test]
fn a_theorem_bound_by_two_contracts_is_restated_once_across_files() {
    let d = fixture();
    std::fs::write(
        d.path().join("contracts/gelu-v2.yaml"),
        "equations:\n  e:\n    lean_theorem: ProvableContracts.Gelu.gelu_bound\n",
    )
    .expect("write second contract");
    let r = render(&d.path().join("lean"), &d.path().join("contracts")).expect("render");
    let decl = "theorem _root_.PvlChallenge.ProvableContracts.Gelu.gelu_bound";
    let n: usize = r.files.values().map(|t| t.matches(decl).count()).sum();
    assert_eq!(n, 1, "{:#?}", r.files);
    assert!(
        r.files["Challenge/gelu-v1.lean"].contains(decl),
        "{:#?}",
        r.files
    );
    assert!(
        !r.files.contains_key("Challenge/gelu-v2.lean"),
        "a contract left with no root of its own writes no file: {:#?}",
        r.files
    );
}

/// #4502: a bound root outside the root's import cone (ORPHANED-ROOT) gets no challenge. Its module is never built,
/// so a challenge importing it cannot elaborate. Importing it again restores the challenge.
#[test]
fn an_orphaned_root_is_not_restated() {
    let d = fixture();
    let root = d.path().join("lean/ProvableContracts.lean");
    std::fs::write(&root, "-- nothing imported\n").expect("write");
    let r = render(&d.path().join("lean"), &d.path().join("contracts")).expect("render");
    assert!(r.files.is_empty(), "{:?}", r.files.keys());
    assert!(r.unrestated.is_empty(), "{:?}", r.unrestated);
    std::fs::write(&root, "import ProvableContracts.Theorems.Gelu.Bound\n").expect("write");
    let r = render(&d.path().join("lean"), &d.path().join("contracts")).expect("render");
    assert!(r.files.contains_key("Challenge/gelu-v1.lean"));
}

/// A hit with an identifier character on only ONE side is not the word; the scan resumes past it (#4587).
#[test]
fn find_word_needs_a_boundary_on_both_sides_and_resumes_past_a_miss() {
    // These two come first, so a scan that resumes at the wrong offset fails fast here rather than
    // looping forever on a rejected hit (at byte 0 below, or a resume that lands back on itself).
    // The first hit is rejected at byte 1, word len 2, so `at - len` = 1 - 2 underflows.
    assert_eq!(find_word("xab ab", "ab"), Some(4));
    // The first hit is rejected at byte 3, word len 3, so `at * len` = 9 resumes past the real hit at 7.
    assert_eq!(find_word("xxxabc abc", "abc"), Some(7));
    assert_eq!(find_word("xfoo bar", "foo"), None, "ident char before");
    assert_eq!(find_word("foox bar", "foo"), None, "ident char after");
}

/// An indented, non-blank line continues the command above it (#4587).
#[test]
fn preamble_joins_an_indented_continuation_line() {
    let scope = preamble("open Foo\n  Bar\ntheorem t : True := trivial\n", 3);
    assert_eq!(scope.top, vec!["open Foo Bar".to_string()]);
}
