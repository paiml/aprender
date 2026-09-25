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
//! This module is NOT feature-gated so that its tests run without CUDA hardware:
//! it sits at the crate root, not under the cuda-gated `driver` (#4096).

use std::collections::{HashMap, HashSet};

/// Patch unconditional backward branches in PTX for Blackwell JIT workaround.
///
/// Returns `None` if no patches were needed (fast path for loop-free kernels).
pub(crate) fn patch_backward_branches_sm121(ptx: &str) -> Option<String> {
    let lines: Vec<&str> = ptx.lines().collect();
    let scopes = function_scopes(&lines);
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

/// Pass 0: the function body (by ordinal) each line belongs to, or `None` at
/// module scope. A `.global` table's `= { … };` initializer is module scope:
/// only a `{` after an `.entry`/`.func` header opens a body (#4096). Labels
/// and `.reg` are function-scoped in PTX, so every later pass keys on this.
fn function_scopes(lines: &[&str]) -> Vec<Option<usize>> {
    let mut scopes = Vec::with_capacity(lines.len());
    let (mut depth, mut header, mut current, mut count) = (0usize, false, None, 0usize);
    for line in lines {
        let code = line.split("//").next().unwrap_or("");
        header |= depth == 0 && (code.contains(".entry") || code.contains(".func"));
        if depth == 0 && header && code.contains('{') {
            current = Some(count);
            count += 1;
            header = false;
        }
        scopes.push(current);
        depth = brace_depth_after(depth, code);
        if depth == 0 {
            current = None;
            header &= !code.trim_end().ends_with(';');
        }
    }
    scopes
}

/// Brace depth after `code`, starting from `depth`.
fn brace_depth_after(depth: usize, code: &str) -> usize {
    code.chars().fold(depth, |d, c| match c {
        '{' => d + 1,
        '}' => d.saturating_sub(1),
        _ => d,
    })
}

/// Pass 1: label definition positions keyed by (function, label name).
fn collect_label_positions<'a>(
    lines: &[&'a str],
    scopes: &[Option<usize>],
) -> HashMap<(usize, &'a str), usize> {
    let mut label_pos = HashMap::new();
    for (i, line) in lines.iter().enumerate() {
        let (Some(f), Some(name)) = (scopes[i], line.trim().strip_suffix(':')) else {
            continue;
        };
        if !name.is_empty() && !name.starts_with('.') && !name.contains(' ') {
            label_pos.insert((f, name), i);
        }
    }
    label_pos
}

/// Pass 2: identify line indices containing unconditional backward branches.
fn find_backward_branches(
    lines: &[&str],
    scopes: &[Option<usize>],
    label_pos: &HashMap<(usize, &str), usize>,
) -> HashSet<usize> {
    let mut patch_set: HashSet<usize> = HashSet::new();
    for (i, line) in lines.iter().enumerate() {
        let Some(f) = scopes[i] else { continue };
        if is_backward_branch(line.trim(), i, f, label_pos) {
            patch_set.insert(i);
        }
    }
    patch_set
}

/// True when `t` is an unconditional `bra LABEL;` whose target was defined
/// earlier in the same function `f`.
fn is_backward_branch(
    t: &str,
    i: usize,
    f: usize,
    label_pos: &HashMap<(usize, &str), usize>,
) -> bool {
    let Some(rest) = t.strip_prefix("bra ") else {
        return false;
    };
    let Some(target) = rest.strip_suffix(';') else {
        return false;
    };
    matches!(label_pos.get(&(f, target.trim())), Some(&def_line) if def_line < i)
}

/// Pass 3: emit the patched PTX string. Each function holding a patched
/// branch gets its own `%p_jw` before its first instruction.
fn emit_patched_ptx(
    ptx: &str,
    lines: &[&str],
    scopes: &[Option<usize>],
    patch_set: &HashSet<usize>,
) -> String {
    let mut out = String::with_capacity(ptx.len() + 128);
    let mut need_decl: HashSet<usize> = patch_set.iter().filter_map(|&i| scopes[i]).collect();

    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if let Some(f) = scopes[i] {
            // Never on the line that opens the body: for a multi-line header
            // that line is `) {`, and a `.reg` before it lands in the
            // parameter list — ptxas rejects the module on sm_121.
            let opener = i == 0 || scopes[i - 1] != Some(f);
            if !opener && !is_meta_line(t) && need_decl.remove(&f) {
                out.push_str("    .reg .pred %p_jw;\n");
                out.push_str("    setp.ne.u32 %p_jw, 1, 0;\n");
            }
        }
        if patch_set.contains(&i) {
            emit_patched_branch(&mut out, line, t);
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

    /// #4096: IQ3_S/IQ2_S open with a `.global … = {` codebook ahead of the
    /// `.entry`. The pass took that `{` for the body and wrote `.reg` into the
    /// initializer: ptxas "Parsing error near '.reg'" on sm_121, no CUDA path.
    #[test]
    fn test_decl_never_lands_in_a_global_initializer_4096() {
        let ptx = ".version 8.0\n.target sm_121\n.address_size 64\n\
            .global .align 4 .u32 t[2] = {\n    1, 2\n};\n\
            .visible .entry k()\n{\n    .reg .u32 %r<2>;\n\
            loop:\n    add.u32 %r0, %r0, 1;\n    bra loop;\n    ret;\n}";
        let patched = patch_backward_branches_sm121(ptx).expect("the loop must be patched");
        let table = patched.find("= {").expect("table kept");
        let table_end = patched.find("};").expect("table close kept");
        let decl = patched.find(".reg .pred %p_jw;").expect("decl emitted");
        let entry_open = patched.find(".visible .entry k()").expect("entry kept");
        assert!(
            !patched[table..table_end].contains("%p_jw"),
            "the initializer must hold only its values:\n{patched}"
        );
        assert!(
            decl > entry_open,
            "decl must sit in the kernel body:\n{patched}"
        );
        assert!(patched.contains("    1, 2\n};\n"), "initializer untouched");
    }

    /// #4096 sibling: `.reg` is function-scoped, so a second kernel with a loop
    /// needs its own `%p_jw` — one decl per module left it undeclared.
    #[test]
    fn test_each_patched_function_gets_its_own_decl_4096() {
        let ptx = ".version 8.0\n.target sm_121\n.address_size 64\n\
            .visible .entry a()\n{\n    .reg .u32 %r<2>;\n\
            la:\n    add.u32 %r0, %r0, 1;\n    bra la;\n    ret;\n}\n\
            .visible .entry b()\n{\n    ret;\n}\n\
            .visible .entry c()\n{\n    .reg .u32 %r<2>;\n\
            lc:\n    add.u32 %r0, %r0, 1;\n    bra lc;\n    ret;\n}";
        let patched = patch_backward_branches_sm121(ptx).expect("both loops patched");
        assert_eq!(patched.matches(".reg .pred %p_jw;").count(), 2, "{patched}");
        let b = patched.find(".visible .entry b()").expect("b kept");
        let c = patched.find(".visible .entry c()").expect("c kept");
        assert!(
            !patched[b..c].contains("%p_jw"),
            "b has no loop, needs no decl"
        );
        assert!(
            patched[c..].contains(".reg .pred %p_jw;"),
            "c declares its own"
        );
    }

    /// #4096 sibling: labels are function-scoped. Two kernels both using
    /// `loop:` keyed one map entry, so the first kernel's backward branch
    /// looked forward (to the second kernel's label) and went unpatched.
    #[test]
    fn test_same_label_in_two_kernels_patches_both_4096() {
        let ptx = ".version 8.0\n.target sm_121\n.address_size 64\n\
            .visible .entry a()\n{\n    .reg .u32 %r<2>;\n\
            loop:\n    add.u32 %r0, %r0, 1;\n    bra loop;\n    ret;\n}\n\
            .visible .entry c()\n{\n    .reg .u32 %r<2>;\n\
            loop:\n    add.u32 %r0, %r0, 1;\n    bra loop;\n    ret;\n}";
        let patched = patch_backward_branches_sm121(ptx).expect("both loops patched");
        assert_eq!(patched.matches("@%p_jw bra loop;").count(), 2, "{patched}");
        assert!(
            !patched.contains("    bra loop;"),
            "no unpatched branch left"
        );
    }

    /// A forward branch to a label in an EARLIER kernel is not a backward
    /// branch: it names this kernel's label, defined later.
    #[test]
    fn test_label_in_earlier_kernel_is_not_a_backward_target_4096() {
        let ptx = ".version 8.0\n.target sm_121\n.address_size 64\n\
            .visible .entry a()\n{\nexit:\n    ret;\n}\n\
            .visible .entry c()\n{\n    bra exit;\nexit:\n    ret;\n}";
        assert!(patch_backward_branches_sm121(ptx).is_none());
    }

    /// Single-line body opener and a `.func` prototype (no body) before it.
    #[test]
    fn test_prototype_and_inline_brace_scope_4096() {
        let ptx = ".version 8.0\n.target sm_121\n.address_size 64\n\
            .extern .func (.param .b32 r) f (.param .b32 x);\n\
            .global .u32 g[1] = { 7 };\n\
            .visible .entry k() {\n    .reg .u32 %r<2>;\n\
            loop: // {\n    add.u32 %r0, %r0, 1;\n    bra loop;\n    ret;\n}";
        let patched = patch_backward_branches_sm121(ptx);
        // `loop: // {` is not a bare label line; the branch has no backward
        // target, so nothing to patch — the comment brace must not open scope.
        assert!(patched.is_none());
        let ptx2 = ptx.replace("loop: // {", "loop:");
        let patched = patch_backward_branches_sm121(&ptx2).expect("patched");
        assert!(
            patched.contains(".global .u32 g[1] = { 7 };\n.visible"),
            "{patched}"
        );
        assert!(patched.find(".reg .pred %p_jw;") > patched.find(".visible .entry k()"));
    }

    /// A multi-line parameter list closed by `) {`: the declaration belongs
    /// after that line, never inside the parameter list. This is the shape
    /// trueno-gpu emits for `batched_rmsnorm_vectorized`, which took the
    /// qwen35 CUDA path down on GB10 (ptxas: syntax error near `.reg`).
    #[test]
    fn test_decl_never_lands_in_a_multiline_param_list() {
        let ptx = ".version 8.0\n.target sm_90\n.address_size 64\n\
            .visible .entry k(\n    .param .u64 a,\n    .param .u64 b\n) {\n\
            .reg .u32 %r<2>;\nloop:\n    add.u32 %r0, %r0, 1;\n    bra loop;\n    ret;\n}";
        let patched = patch_backward_branches_sm121(ptx).expect("patched");
        let decl = patched.find(".reg .pred %p_jw;").expect("decl");
        let body = patched.find(") {").expect("opener");
        assert!(decl > body, "decl inside the parameter list:\n{patched}");
        assert!(patched.contains(".param .u64 b\n) {\n"), "{patched}");
    }
}
