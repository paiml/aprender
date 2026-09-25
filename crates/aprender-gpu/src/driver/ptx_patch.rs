//! GH-480: PTX backward branch patcher for Blackwell (sm_121+)
//!
//! The CUDA 13.0 JIT compiler on sm_121 has a bug where unconditional backward
//! branches (`bra label` where `label` is defined earlier in the code) cause
//! loop iterations to be silently dropped. This manifests as second+ loops in
//! a kernel executing fewer iterations than required, producing garbage results.
//!
//! Fix: Convert all unconditional backward branches to conditional branches
//! using an always-true predicate (`@%p_jw bra label`). This forces the JIT
//! to use a different code generation path that produces correct SASS.
//!
//! This module is NOT feature-gated so that its tests run without CUDA hardware.

use std::collections::{HashMap, HashSet};

/// Patch unconditional backward branches in PTX for Blackwell JIT workaround.
///
/// Returns `None` if no patches were needed (fast path for loop-free kernels).
pub(crate) fn patch_backward_branches_sm121(ptx: &str) -> Option<String> {
    let lines: Vec<&str> = ptx.lines().collect();
    let scopes = scan_scopes(&lines);
    let label_pos = collect_label_positions(&lines, &scopes);
    let patch_set = find_backward_branches(&lines, &scopes, &label_pos);
    if patch_set.is_empty() {
        return None;
    }
    let patch_count = patch_set.len();
    let out = emit_patched_ptx(ptx, &lines, &scopes, &patch_set);
    eprintln!("[GH-480] Patched {patch_count} backward branch(es) for sm_121 JIT workaround");
    Some(out)
}

/// Where one PTX line sits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Scope {
    /// The function (`.entry` / `.func`) whose body holds the line, numbered
    /// from 0 in file order; `None` at module scope, which includes a
    /// `.global` array initializer (#4096).
    func: Option<usize>,
    /// Brace depth before the line: 1 is the function body's own block.
    depth: usize,
    /// The line opens a function body.
    opens_body: bool,
}

/// Pass 0: find each line's function and brace depth.
///
/// A `{` at module scope opens a function body only after a `.entry` or
/// `.func` header. Any other module-scope `{` (a `.global ... = {` codebook,
/// #4096) is an initializer: the predicate must never be declared inside it.
fn scan_scopes(lines: &[&str]) -> Vec<Scope> {
    let mut scopes = Vec::with_capacity(lines.len());
    let mut depth = 0usize;
    let mut header_pending = false;
    let mut in_func = false;
    let mut funcs = 0usize;
    for line in lines {
        let code = line.split("//").next().unwrap_or("");
        let t = code.trim();
        let opens = code.matches('{').count();
        let closes = code.matches('}').count();
        let mut scope = Scope {
            func: in_func.then(|| funcs - 1),
            depth,
            opens_body: false,
        };
        if depth == 0 {
            if t.split_whitespace().any(|w| w == ".entry" || w == ".func") {
                header_pending = true;
            }
            if opens > 0 && header_pending {
                header_pending = false;
                in_func = true;
                funcs += 1;
                scope.func = Some(funcs - 1);
                scope.opens_body = true;
            } else if t.ends_with(';') {
                // A prototype (`.extern .func f(...);`) has no body.
                header_pending = false;
            }
        }
        depth = (depth + opens).saturating_sub(closes);
        if depth == 0 {
            in_func = false;
        }
        scopes.push(scope);
    }
    scopes
}

/// Pass 1: collect PTX label definition positions, keyed by function and
/// label name (PTX labels are function-scoped).
fn collect_label_positions<'a>(
    lines: &[&'a str],
    scopes: &[Scope],
) -> HashMap<(usize, &'a str), usize> {
    let mut label_pos = HashMap::new();
    for (i, line) in lines.iter().enumerate() {
        let Some(func) = scopes[i].func else {
            continue;
        };
        let Some(name) = line.trim().strip_suffix(':') else {
            continue;
        };
        if !name.is_empty() && !name.starts_with('.') && !name.contains(' ') {
            label_pos.insert((func, name), i);
        }
    }
    label_pos
}

/// Pass 2: identify line indices containing unconditional backward branches.
fn find_backward_branches(
    lines: &[&str],
    scopes: &[Scope],
    label_pos: &HashMap<(usize, &str), usize>,
) -> HashSet<usize> {
    let mut patch_set: HashSet<usize> = HashSet::new();
    for (i, line) in lines.iter().enumerate() {
        let Some(func) = scopes[i].func else {
            continue;
        };
        if is_backward_branch(line.trim(), i, func, label_pos) {
            patch_set.insert(i);
        }
    }
    patch_set
}

/// True when `t` is an unconditional `bra LABEL;` whose target was defined
/// earlier in the same function.
fn is_backward_branch(
    t: &str,
    i: usize,
    func: usize,
    label_pos: &HashMap<(usize, &str), usize>,
) -> bool {
    let Some(rest) = t.strip_prefix("bra ") else {
        return false;
    };
    let Some(target) = rest.strip_suffix(';') else {
        return false;
    };
    matches!(label_pos.get(&(func, target.trim())), Some(&def_line) if def_line < i)
}

/// Pass 3: emit the patched PTX string. Every function body gets its own
/// `%p_jw` declaration, before its first instruction.
fn emit_patched_ptx(
    ptx: &str,
    lines: &[&str],
    scopes: &[Scope],
    patch_set: &HashSet<usize>,
) -> String {
    let mut out = String::with_capacity(ptx.len() + 128);
    let mut decl_emitted = false;

    for (i, line) in lines.iter().enumerate() {
        let scope = scopes[i];
        if scope.opens_body {
            decl_emitted = false;
        } else if scope.func.is_some()
            && scope.depth == 1
            && !decl_emitted
            && !is_meta_line(line.trim())
        {
            out.push_str("    .reg .pred %p_jw;\n");
            out.push_str("    setp.ne.u32 %p_jw, 1, 0;\n");
            decl_emitted = true;
        }
        if patch_set.contains(&i) {
            emit_patched_branch(&mut out, line, line.trim());
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }

    if !ptx.ends_with('\n') && out.ends_with('\n') {
        out.pop();
    }
    out
}

/// True for PTX lines that should not trigger predicate declaration insertion.
fn is_meta_line(t: &str) -> bool {
    t.is_empty() || t.starts_with('.') || t.starts_with("//") || t == "{" || t == "}"
}

/// Emit a single patched backward branch with preserved indentation.
fn emit_patched_branch(out: &mut String, line: &str, t: &str) {
    let indent_len = line.len() - line.trim_start().len();
    let indent = &line[..indent_len];
    let target = t.get("bra ".len()..t.len() - 1).unwrap_or(t).trim();
    out.push_str(indent);
    out.push_str("@%p_jw bra ");
    out.push_str(target);
    out.push_str(";\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_backward_branches() {
        let ptx = ".version 8.0\n.target sm_90\n.address_size 64\n\
            .visible .entry test()\n{\n    .reg .u32 %r<2>;\n\
            mov.u32 %r0, 0;\n    bra exit;\nexit:\n    ret;\n}";
        // Forward-only branch — no patches needed
        assert!(patch_backward_branches_sm121(ptx).is_none());
    }

    #[test]
    fn test_single_backward_branch() {
        let ptx = ".version 8.0\n.target sm_90\n.address_size 64\n\
            .visible .entry test()\n{\n    .reg .u32 %r<2>;\n    .reg .pred %p<2>;\n\
            mov.u32 %r0, 0;\nloop:\n    add.u32 %r0, %r0, 1;\n\
            setp.lt.u32 %p0, %r0, 10;\n    @%p0 bra done;\n    bra loop;\ndone:\n    ret;\n}";
        let patched =
            patch_backward_branches_sm121(ptx).expect("single backward branch should be patched");
        assert!(patched.contains("@%p_jw bra loop;"));
        assert!(patched.contains("@%p0 bra done;"));
        assert!(patched.contains(".reg .pred %p_jw;"));
        assert!(patched.contains("setp.ne.u32 %p_jw, 1, 0;"));
    }

    #[test]
    fn test_multiple_backward_branches() {
        let ptx = ".version 8.0\n.target sm_90\n.address_size 64\n\
            .visible .entry test()\n{\n    .reg .u32 %r<4>;\n\
            loop1:\n    add.u32 %r0, %r0, 1;\n    bra loop1;\n\
            loop2:\n    add.u32 %r1, %r1, 1;\n    bra loop2;\n    ret;\n}";
        let patched = patch_backward_branches_sm121(ptx)
            .expect("multiple backward branches should be patched");
        assert!(patched.contains("@%p_jw bra loop1;"));
        assert!(patched.contains("@%p_jw bra loop2;"));
        assert_eq!(patched.matches(".reg .pred %p_jw;").count(), 1);
        assert_eq!(patched.matches("setp.ne.u32 %p_jw, 1, 0;").count(), 1);
    }

    #[test]
    fn test_preserves_conditional_backward_branches() {
        let ptx = ".version 8.0\n.target sm_90\n.address_size 64\n\
            .visible .entry test()\n{\n    .reg .u32 %r<2>;\n    .reg .pred %p<2>;\n\
            loop:\n    add.u32 %r0, %r0, 1;\n    setp.lt.u32 %p0, %r0, 10;\n\
            @%p0 bra loop;\n    ret;\n}";
        // Already conditional — no patches needed
        assert!(patch_backward_branches_sm121(ptx).is_none());
    }

    #[test]
    fn test_preserves_indentation() {
        let ptx = ".version 8.0\n.target sm_90\n.address_size 64\n\
            .visible .entry test()\n{\n    .reg .u32 %r<2>;\n\
            loop:\n        add.u32 %r0, %r0, 1;\n        bra loop;\n    ret;\n}";
        let patched =
            patch_backward_branches_sm121(ptx).expect("indented backward branch should be patched");
        assert!(patched.contains("        @%p_jw bra loop;"));
    }

    #[test]
    fn test_nested_loops() {
        let ptx = ".version 8.0\n.target sm_90\n.address_size 64\n\
            .visible .entry test()\n{\n    .reg .u32 %r<4>;\n    .reg .pred %p<4>;\n\
            outer:\n    setp.ge.u32 %p0, %r0, 10;\n    @%p0 bra exit;\n\
            inner:\n    add.u32 %r1, %r1, 1;\n    setp.lt.u32 %p1, %r1, 32;\n\
            @%p1 bra skip;\n    bra inner;\nskip:\n    add.u32 %r0, %r0, 1;\n\
            bra outer;\nexit:\n    ret;\n}";
        let patched =
            patch_backward_branches_sm121(ptx).expect("nested backward branches should be patched");
        assert!(patched.contains("@%p_jw bra inner;"));
        assert!(patched.contains("@%p_jw bra outer;"));
        // Forward branches preserved
        assert!(patched.contains("@%p0 bra exit;"));
        assert!(patched.contains("@%p1 bra skip;"));
    }

    #[test]
    fn test_no_loops_fast_path() {
        // ROPE-style linear kernel (no loops)
        let ptx = ".version 8.0\n.target sm_90\n.address_size 64\n\
            .visible .entry rope()\n{\n    .reg .u32 %r<2>;\n    .reg .f32 %f<4>;\n\
            mov.u32 %r0, %tid.x;\n    mul.f32 %f0, %f1, %f2;\n    ret;\n}";
        assert!(patch_backward_branches_sm121(ptx).is_none());
    }

    #[test]
    fn test_decl_inserted_before_first_instruction() {
        let ptx = ".version 8.0\n.target sm_90\n.address_size 64\n\
            .visible .entry test()\n{\n    .reg .u32 %r<2>;\n    .reg .f32 %f<2>;\n\
            mov.u32 %r0, 0;\nloop:\n    add.u32 %r0, %r0, 1;\n    bra loop;\n    ret;\n}";
        let patched =
            patch_backward_branches_sm121(ptx).expect("backward branch should be patched");
        // The .reg .pred and setp should appear AFTER .reg .f32 but BEFORE mov
        let pred_pos = patched
            .find(".reg .pred %p_jw;")
            .expect("patched PTX must contain pred decl");
        let setp_pos = patched
            .find("setp.ne.u32 %p_jw, 1, 0;")
            .expect("patched PTX must contain setp init");
        let first_mov = patched
            .find("mov.u32 %r0, 0;")
            .expect("patched PTX must contain mov instruction");
        let last_reg = patched
            .rfind(".reg .f32")
            .expect("patched PTX must contain .reg .f32 decl");
        assert!(pred_pos > last_reg, "pred decl must come after last .reg");
        assert!(setp_pos > pred_pos, "setp must come after pred decl");
        assert!(
            setp_pos < first_mov,
            "setp must come before first instruction"
        );
    }
    /// The IQ3_S GEMV module's shape (`aprender-serve` `cuda/layout.rs`): a
    /// multi-line `.global` codebook initializer BEFORE the `.entry`, and a
    /// loop in the body.
    const IQ3S_SHAPED: &str = "
.version 7.5
.target sm_70
.address_size 64

// IQ3S_GRID: 512 packed 4-byte codebook entries.
.global .align 4 .u32 iq3s_grid_g[512] = {
    16843009, 16843011, 16843013, 16843019,
    252248839, 252249345, 252250881, 252641537
};

.visible .entry iq3_s_gemv_warp_reduce(
    .param .u64 y_ptr,
    .param .u32 k_dim
)
{
    .reg .u32 %r<8>;
    .reg .pred %p<4>;

    mov.u32 %r0, %tid.x;
    mov.u64 %rd10, iq3s_grid_g;
$L_s3_blk:
    setp.ge.u32 %p1, %r0, 8;
    @%p1 bra $L_s3_blk_end;
    add.u32 %r0, %r0, 1;
    bra $L_s3_blk;
$L_s3_blk_end:
    ret;
}
";

    /// The lines between `= {` and `};` of the first `.global` initializer.
    fn initializer_body(ptx: &str) -> Vec<&str> {
        ptx.lines()
            .skip_while(|l| !l.trim_end().ends_with("= {"))
            .skip(1)
            .take_while(|l| l.trim() != "};")
            .collect()
    }

    /// #4096: the predicate was declared INSIDE the codebook initializer, the
    /// first line ending with `{`, and ptxas rejected the module on sm_121
    /// ("Parsing error near '.reg'"), so IQ3_S/IQ2_S had no CUDA path on cc>=12.
    #[test]
    fn global_initializer_is_not_a_function_body_4096() {
        let patched = patch_backward_branches_sm121(IQ3S_SHAPED)
            .expect("the IQ3_S-shaped loop must be patched");
        assert_eq!(
            initializer_body(&patched),
            initializer_body(IQ3S_SHAPED),
            "the codebook initializer must come through byte for byte"
        );
        let decl = patched.find(".reg .pred %p_jw;").expect("decl");
        let entry = patched.find(".visible .entry").expect("entry");
        let first_mov = patched.find("mov.u32 %r0, %tid.x;").expect("mov");
        assert!(decl > entry, "the predicate belongs to the entry's body");
        assert!(decl < first_mov, "declared before the first instruction");
        assert!(patched.contains("@%p_jw bra $L_s3_blk;"));
        assert_eq!(patched.matches(".reg .pred %p_jw;").count(), 1);
    }

    /// A one-line initializer (`kvalues_iq4nl[16] = {...};`) never opened a
    /// body, and must still not.
    #[test]
    fn one_line_initializer_before_entry_is_module_scope() {
        let ptx = IQ3S_SHAPED.replace(
            ".global .align 4 .u32 iq3s_grid_g[512] = {\n    16843009, 16843011, 16843013, 16843019,\n    252248839, 252249345, 252250881, 252641537\n};",
            ".global .align 4 .s32 kvalues_iq4nl[4] = {-127, -104, -83, -65};",
        );
        assert!(ptx.contains("kvalues_iq4nl[4] = {-127, -104, -83, -65};"));
        let patched = patch_backward_branches_sm121(&ptx).expect("patched");
        let decl = patched.find(".reg .pred %p_jw;").expect("decl");
        assert!(decl > patched.find(".visible .entry").expect("entry"));
    }

    /// Two functions in one module: each body declares its own predicate
    /// (a `.reg` is function-scoped), and a label name reused by the second
    /// function does not hide the first function's backward branch.
    #[test]
    fn every_function_declares_its_own_predicate_and_labels_are_function_scoped() {
        let ptx = ".version 8.0\n.target sm_90\n.address_size 64\n\
            .func helper()\n{\n    .reg .u32 %r<2>;\n    mov.u32 %r0, 0;\n\
            loop:\n    add.u32 %r0, %r0, 1;\n    bra loop;\n    ret;\n}\n\
            .visible .entry main() {\n    .reg .u32 %r<2>;\n    mov.u32 %r1, 0;\n\
            bra loop;\nloop:\n    add.u32 %r1, %r1, 1;\n    ret;\n}\n";
        let patched = patch_backward_branches_sm121(ptx).expect("helper's loop is backward");
        assert_eq!(
            patched.matches(".reg .pred %p_jw;").count(),
            2,
            "one declaration per function body:\n{patched}"
        );
        assert_eq!(
            patched.matches("@%p_jw bra loop;").count(),
            1,
            "helper's `bra loop` is backward; main's is forward:\n{patched}"
        );
        let main_at = patched.find(".visible .entry main()").expect("main");
        assert!(
            patched[..main_at].contains("@%p_jw bra loop;"),
            "the patched branch is helper's"
        );
    }

    /// A prototype has no body: the next module-scope `{` (an initializer)
    /// must not be taken for one.
    #[test]
    fn a_prototype_does_not_arm_the_next_brace() {
        let ptx = ".version 8.0\n.target sm_90\n.address_size 64\n\
            .extern .func ext(.param .u32 a);\n\
            .global .align 4 .u32 tbl[2] = {\n    1, 2\n};\n\
            .visible .entry k()\n{\n    .reg .u32 %r<2>;\n\
            l:\n    add.u32 %r0, %r0, 1;\n    bra l;\n}\n";
        let patched = patch_backward_branches_sm121(ptx).expect("patched");
        assert_eq!(initializer_body(&patched), vec!["    1, 2"]);
    }
}
