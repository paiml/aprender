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

    // Pass 1: Collect label definition positions
    let mut label_pos: HashMap<&str, usize> = HashMap::new();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if let Some(name) = t.strip_suffix(':') {
            if !name.is_empty() && !name.starts_with('.') && !name.contains(' ') {
                label_pos.insert(name, i);
            }
        }
    }

    // Pass 2: Find unconditional backward branches
    let mut patch_set: HashSet<usize> = HashSet::new();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        // Match "bra LABEL;" — unconditional (no @%p prefix)
        if let Some(rest) = t.strip_prefix("bra ") {
            if let Some(target) = rest.strip_suffix(';') {
                let target = target.trim();
                if let Some(&def_line) = label_pos.get(target) {
                    if def_line < i {
                        patch_set.insert(i);
                    }
                }
            }
        }
    }

    if patch_set.is_empty() {
        return None;
    }

    let patch_count = patch_set.len();

    // Pass 3: Build patched PTX
    let mut out = String::with_capacity(ptx.len() + 128);
    let mut in_body = false;
    let mut decl_emitted = false;

    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();

        // Detect kernel body entry
        if !in_body && (t == "{" || t.ends_with('{')) {
            in_body = true;
            out.push_str(line);
            out.push('\n');
            continue;
        }

        // Insert predicate declaration + init before first executable line
        if in_body && !decl_emitted {
            let is_meta =
                t.is_empty() || t.starts_with('.') || t.starts_with("//") || t == "{" || t == "}";
            if !is_meta {
                out.push_str("    .reg .pred %p_jw;\n");
                out.push_str("    setp.ne.u32 %p_jw, 1, 0;\n");
                decl_emitted = true;
            }
        }

        // Patch backward branches
        if patch_set.contains(&i) {
            let indent_len = line.len() - line.trim_start().len();
            let indent = &line[..indent_len];
            // Safety: patch_set only contains lines where strip_prefix/suffix succeeded
            let target = t.get("bra ".len()..t.len() - 1).unwrap_or(t).trim();
            out.push_str(indent);
            out.push_str("@%p_jw bra ");
            out.push_str(target);
            out.push_str(";\n");
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }

    // Match trailing newline behavior of input
    if !ptx.ends_with('\n') && out.ends_with('\n') {
        out.pop();
    }

    eprintln!("[GH-480] Patched {patch_count} backward branch(es) for sm_121 JIT workaround");

    Some(out)
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
        let pred_pos =
            patched.find(".reg .pred %p_jw;").expect("patched PTX must contain pred decl");
        let setp_pos =
            patched.find("setp.ne.u32 %p_jw, 1, 0;").expect("patched PTX must contain setp init");
        let first_mov =
            patched.find("mov.u32 %r0, 0;").expect("patched PTX must contain mov instruction");
        let last_reg = patched.rfind(".reg .f32").expect("patched PTX must contain .reg .f32 decl");
        assert!(pred_pos > last_reg, "pred decl must come after last .reg");
        assert!(setp_pos > pred_pos, "setp must come after pred decl");
        assert!(setp_pos < first_mov, "setp must come before first instruction");
    }
}
