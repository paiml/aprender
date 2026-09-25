#!/usr/bin/env python3
"""classify_bare_int_qtype.py — every bare-integer match arm / compare in the
workspace is classified `qtype | not-qtype | unknown`, and the tree may not grow
a new `qtype` or `unknown` line that the committed baseline does not list.

WHY (#3431, PP-QUANT-001 M2; operator "First task", 2026-09-17). The PMAT-3427
receipt measured a regex upper bound of 558 non-test lines where an integer
literal might be a ggml tensor-type id, and resolved 2 of them. A bare `12 =>`
is exactly the site that cannot see upstream add NVFP4 (40), Q1_0 (41), Q2_0
(42): the named enum `GgmlType` is reconciled to ggml by
`crates/aprender-quant/tests/ggml_traits_fixture.rs`; a literal is reconciled to
nothing. The receipt never recorded its regex, so the pattern is re-derived
here and ships its case table (this repo's guard regexes were wrong five times;
every one was caught by a table, none by review).

THE UNIT IS A LINE, THE EVIDENCE IS ITS `match` SCRUTINEE.
  candidate  a non-test line that is an integer-literal match arm
             (`12 =>`, `0 | 1 =>`, `8 if x =>`) or compares a qtype-named
             value to an integer (`qtype == 12`, `dtype != 0`).
  qtype      the compare names a qtype-ish identifier, or the arm's enclosing
             `match` scrutinee does.
  unknown    the scrutinee is not qtype-named, but the arm's line names a ggml
             type (`Q4_K`, `GGML_TYPE_`, `GgmlType::`, `F16` …) — a literal id
             under an innocuous name.
  not-qtype  everything else.

GATE. qtype and unknown lines are keyed (path, stripped text) and must each be
in the baseline with at least that multiplicity. A new one is RED. A baseline
row the tree no longer has is reported (shrink the baseline) but is not RED.

  python3 scripts/classify_bare_int_qtype.py              # gate
  python3 scripts/classify_bare_int_qtype.py --self-test  # case table
  python3 scripts/classify_bare_int_qtype.py --summary    # counts by class/crate
  python3 scripts/classify_bare_int_qtype.py --write-baseline
"""
from __future__ import annotations

import collections
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(ROOT, "docs", "audits", "bare-int-qtype-baseline.tsv")

QTYPE_WORD = r"(?:qtype|q_type|ggml_type|ggml_dtype|quant_type|tensor_type|weight_type|dtype|type_id|GgmlType)"
QTYPE_NAME_RE = re.compile(r"(?i)" + QTYPE_WORD)
INT_ARM_RE = re.compile(r"^\s*(?:\d+|0x[0-9a-fA-F]+)(?:_?[ui](?:8|16|32|64|size))?"
                        r"(?:\s*(?:\||\.\.=?)\s*(?:\d+|0x[0-9a-fA-F]+)(?:_?[ui](?:8|16|32|64|size))?)*"
                        r"\s*(?:if\b[^=]*)?=>")
INT_CMP_RE = re.compile(r"(?i)\b[\w.]*" + QTYPE_WORD + r"[\w.]*(?:\s+as\s+\w+)?\s*(?:==|!=)\s*\d+\b")
MATCH_RE = re.compile(r"\bmatch\s+(.+?)\s*\{\s*$")
FN_RE = re.compile(r"\bfn\s+(\w+)")

# REVIEWED SITES. The name-based rule is a heuristic; each row below is a site a
# person read and resolved, keyed (path, enclosing fn) -> (class, why). A row
# that no longer matches any candidate is reported by the gate, so this table
# cannot rot silently.
OVERRIDES = {
    ("crates/apr-cli/src/commands/debug.rs", "format_model_type"):
        ("not-qtype", "APR model-type id (0x0001 LinearRegression …), not a tensor type"),
    ("crates/aprender-serve/src/model_loader.rs", "read_apr_v1_model_type"):
        ("not-qtype", "APR model-type id (0x0001 LinearRegression …), not a tensor type"),
    ("crates/aprender-compute/src/inference/gguf.rs", "read_metadata_value"):
        ("not-qtype", "GGUF metadata VALUE type (0 u8 … 12 f64), not a tensor type"),
    ("crates/aprender-gpu/fuzz/fuzz_targets/fuzz_ptx_builder.rs", "param_type"):
        ("not-qtype", "fuzzer choice index into PtxType"),
    ("crates/aprender-core/src/format/quantize.rs", "from_u8"):
        ("not-qtype", "APR's own QuantType byte (0x01 Q8_0 …), a separate id space"),
    ("crates/aprender-serve/src/cuda/executor/graph_dispatch.rs", "qtype_from_ggml"):
        ("qtype", "ggml ids by definition — the fn is named for it"),
    ("crates/aprender-present-yaml/src/formats.rs", "from_u32"):
        ("not-qtype", "present-yaml's own DType (1 = F64), unrelated to ggml"),
    ("crates/aprender-serve/src/apr_transformer/loader.rs", "from_byte"):
        ("not-qtype", "quantized-APR header byte (1 = Q4_K), its own 3-id space"),
    ("crates/apr-format/src/v2/tensor_index_impl.rs", "from_u8"):
        ("qtype", "APR TensorDType shares ggml ids 12/14/30 and shadows 8/9 (GH-438)"),
    ("crates/aprender-quant/src/ggml_type.rs", "from_id_inner"):
        ("qtype", "THE sanctioned id table, reconciled to upstream by ggml_traits_fixture"),
}
GGML_NAME_RE = re.compile(r"\b(?:GGML_TYPE_\w+|GgmlType::\w+|Q[1-8]_(?:K|0|1)\w*|IQ[1-4]_\w+|"
                          r"TQ[12]_0|BF16|F16|F32|MXFP4|NVFP4)\b")
CFG_TEST_RE = re.compile(r"#\[cfg\((?:all\()?test")


used_overrides: set = set()


def is_test_path(rel: str) -> bool:
    parts = rel.split("/")
    base = parts[-1]
    return ("tests" in parts or "benches" in parts or "examples" in parts
            or base in ("tests.rs", "test.rs") or base.endswith("_tests.rs")
            or base.endswith("_test.rs") or base.startswith("test_")
            or base.startswith("tests_"))


def strip_code(line: str) -> str:
    """Line with string/char literals blanked and `//` comments cut, so braces
    and patterns inside them do not count."""
    out, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if c == "/" and i + 1 < n and line[i + 1] == "/":
            break
        if c == '"':
            j = i + 1
            while j < n and line[j] != '"':
                j += 2 if line[j] == "\\" else 1
            out.append('""')
            i = j + 1
            continue
        if c == "'" and i + 2 < n and (line[i + 2] == "'" or line[i + 1] == "\\"):
            j = line.find("'", i + 2)
            if j != -1:
                out.append("' '")
                i = j + 1
                continue
        out.append(c)
        i += 1
    return "".join(out)


def classify_text(rel: str, text: str):
    """Yield (lineno, class, stripped_line) for every candidate in one file."""
    lines = text.splitlines()
    depth = 0
    match_stack = []          # (depth_inside, scrutinee)
    fn_stack = []             # (depth_inside, name)
    pending_fn = None
    skip_until = None         # depth at which a #[cfg(test)] item closes
    pending_cfg_test = False
    in_block_comment = False
    for no, raw in enumerate(lines, 1):
        line = raw
        if in_block_comment:
            end = line.find("*/")
            if end == -1:
                continue
            line = line[end + 2:]
            in_block_comment = False
        code = strip_code(line)
        if "/*" in code and "*/" not in code[code.find("/*"):]:
            code = code[:code.find("/*")]
            in_block_comment = True
        stripped = code.strip()
        if CFG_TEST_RE.search(stripped):
            pending_cfg_test = True
        opens, closes = code.count("{"), code.count("}")
        if skip_until is None and pending_cfg_test and opens and not stripped.startswith("#"):
            skip_until = depth
            pending_cfg_test = False
        elif pending_cfg_test and stripped.endswith(";"):
            pending_cfg_test = False
        fm = FN_RE.search(code)
        if fm:
            pending_fn = fm.group(1)
        if pending_fn and opens:
            fn_stack.append((depth + 1, pending_fn))
            pending_fn = None
        elif pending_fn and stripped.endswith(";"):
            pending_fn = None
        fn_name = fn_stack[-1][1] if fn_stack else ""
        if skip_until is None and not stripped.startswith("//"):
            m = MATCH_RE.search(code)
            cmp_hit = INT_CMP_RE.search(code)
            arm_hit = INT_ARM_RE.match(code)
            cls = None
            if cmp_hit:
                cls = "qtype"
            elif arm_hit and match_stack and match_stack[-1][0] == depth:
                scrut = match_stack[-1][1]
                if QTYPE_NAME_RE.search(scrut):
                    cls = "qtype"
                elif GGML_NAME_RE.search(code[arm_hit.end():]):
                    cls = "unknown"
                else:
                    cls = "not-qtype"
            if cls:
                ov = OVERRIDES.get((rel, fn_name))
                if ov:
                    cls = ov[0]
                    used_overrides.add((rel, fn_name))
                yield no, cls, raw.strip()
            if m:
                match_stack.append((depth + 1, m.group(1)))
        depth += opens - closes
        while match_stack and depth < match_stack[-1][0]:
            match_stack.pop()
        while fn_stack and depth < fn_stack[-1][0]:
            fn_stack.pop()
        if skip_until is not None and depth <= skip_until:
            skip_until = None


def scan(crates_dir: str):
    for dirpath, dirnames, filenames in os.walk(crates_dir):
        dirnames[:] = sorted(d for d in dirnames if d not in ("target", ".git"))
        for f in sorted(filenames):
            if not f.endswith(".rs"):
                continue
            path = os.path.join(dirpath, f)
            rel = os.path.relpath(path, ROOT)
            if is_test_path(rel):
                continue
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            for no, cls, s in classify_text(rel, text):
                yield rel, no, cls, s


def load_baseline():
    rows = collections.Counter()
    if not os.path.exists(BASELINE):
        return rows
    with open(BASELINE, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            cls, rel, text = line.rstrip("\n").split("\t", 2)
            rows[(cls, rel, text)] += 1
    return rows


def self_test() -> int:
    # (label, file body, wanted [(class, text)] in order)
    Q, N, U = "qtype", "not-qtype", "unknown"
    cases = [
        # MUST CLASSIFY qtype
        ("arm under `match qtype`", "match qtype {\n    12 => a(),\n    _ => b(),\n}",
         [(Q, "12 => a(),")]),
        ("arm under `match t.qtype`", "match t.qtype {\n    0 | 1 => a(),\n}", [(Q, "0 | 1 => a(),")]),
        ("arm under `match ggml_type as u32`", "match ggml_type as u32 {\n    14 => 210,\n}", [(Q, "14 => 210,")]),
        ("guarded arm under `match dtype`", "match dtype {\n    8 if big => x,\n}", [(Q, "8 if big => x,")]),
        ("suffixed literal arm", "match qtype {\n    12u32 => x,\n}", [(Q, "12u32 => x,")]),
        ("range arm", "match qtype {\n    10..=14 => k(),\n}", [(Q, "10..=14 => k(),")]),
        ("compare `qtype == 12`", "if qtype == 12 { k() }", [(Q, "if qtype == 12 { k() }")]),
        ("compare `t.tensor_type != 0`", "let x = t.tensor_type != 0;", [(Q, "let x = t.tensor_type != 0;")]),
        ("compare through a cast", "if w.qtype as u32 == 14 {}", [(Q, "if w.qtype as u32 == 14 {}")]),
        ("arm in a nested match keeps the INNER scrutinee",
         "match n {\n    1 => match qtype {\n        12 => a(),\n    },\n}",
         [(N, "1 => match qtype {"), (Q, "12 => a(),")]),
        # MUST CLASSIFY unknown
        ("innocuous scrutinee, arm names a ggml type", "match t {\n    12 => \"x\",\n    14 => GgmlType::Q6K,\n}",
         [(N, "12 => \"x\","), (U, "14 => GgmlType::Q6K,")]),
        ("innocuous scrutinee, arm names Q4_K", "match id {\n    12 => Q4_K_BLOCK,\n}", [(U, "12 => Q4_K_BLOCK,")]),
        # MUST CLASSIFY not-qtype
        ("arm under `match n_dims`", "match n_dims {\n    2 => a(),\n}", [(N, "2 => a(),")]),
        ("arm under `match version`", "match version {\n    3 => ok(),\n}", [(N, "3 => ok(),")]),
        # MUST NOT MATCH
        ("a non-literal arm", "match qtype {\n    Q4_K => a(),\n}", []),
        ("a commented-out arm", "match qtype {\n    // 12 => a(),\n}", []),
        ("an arm in a block comment", "match qtype {\n    /*\n    12 => a(),\n    */\n}", []),
        ("a qtype word inside a string", "let s = \"qtype == 12\";", []),
        ("compare of a non-qtype name", "if n_layers == 12 {}", []),
        ("a literal `=>` inside a string", "let s = \"12 => x\";", []),
        ("a #[cfg(test)] module", "#[cfg(test)]\nmod tests {\n    fn f(qtype: u32) { if qtype == 12 {} }\n}\nfn g() {}", []),
        ("code AFTER a #[cfg(test)] module still counts",
         "#[cfg(test)]\nmod tests {\n    fn f() {}\n}\nfn g(qtype: u32) -> bool { qtype == 1 }",
         [(Q, "fn g(qtype: u32) -> bool { qtype == 1 }")]),
        ("a #[cfg(test)] use line does not swallow the file",
         "#[cfg(test)]\nuse x::y;\nfn g(qtype: u32) -> bool { qtype == 1 }",
         [(Q, "fn g(qtype: u32) -> bool { qtype == 1 }")]),
        ("a closure body `=>` in a non-match block", "let f = |x| {\n    12 => 1,\n};", []),
    ]
    fails = 0
    print("classify_bare_int_qtype self-test")
    # A reviewed override re-classes by (path, enclosing fn), and ONLY there.
    OVERRIDES[("crates/x/src/ov.rs", "model_kind")] = ("not-qtype", "self-test row")
    for label, rel, want in [
        ("override applies in its (path, fn)", "crates/x/src/ov.rs", "not-qtype"),
        ("override does not leak to another path", "crates/x/src/other.rs", "qtype"),
    ]:
        got = [c for _, c, _ in classify_text(rel, "fn model_kind(dtype: u32) {\n    match dtype {\n        3 => a(),\n    }\n}")]
        ok = got == [want]
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {label:<56} {'' if ok else f'got {got}, wanted {[want]}'}")
    del OVERRIDES[("crates/x/src/ov.rs", "model_kind")]
    for label, body, want in cases:
        got = [(c, s) for _, c, s in classify_text("crates/x/src/case.rs", body)]
        ok = got == want
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {label:<56} {'' if ok else f'got {got}, wanted {want}'}")
    for label, rel, want in [
        ("path: a tests/ dir is test", "crates/a/tests/x.rs", True),
        ("path: a *_tests.rs file is test", "crates/a/src/foo_tests.rs", True),
        ("path: a tests_*.rs file is test", "crates/a/src/q/tests_part_02.rs", True),
        ("path: a normal source file is not", "crates/a/src/quantize/mod.rs", False),
        ("path: a name merely containing `test` is not", "crates/a/src/attestation.rs", False),
    ]:
        ok = is_test_path(rel) == want
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    total = len(cases) + 7
    print(f"\n{total} case(s), {fails} failure(s)")
    return 1 if fails else 0


def main(argv) -> int:
    if "--self-test" in argv:
        return self_test()
    hits = list(scan(os.path.join(ROOT, "crates")))
    by_class = collections.Counter(c for _, _, c, _ in hits)
    if "--summary" in argv:
        per = collections.Counter((r.split("/")[1], c) for r, _, c, _ in hits)
        print(f"candidates {len(hits)}: " + ", ".join(f"{k} {v}" for k, v in sorted(by_class.items())))
        for (crate, c), v in sorted(per.items()):
            print(f"  {crate:<32} {c:<10} {v}")
        return 0
    tracked = [(c, r, s) for r, _, c, s in hits if c != "not-qtype"]
    if "--write-baseline" in argv:
        with open(BASELINE, "w", encoding="utf-8") as fh:
            fh.write("# class\tpath\tline text — generated by scripts/classify_bare_int_qtype.py "
                     "--write-baseline (#3431). May only SHRINK.\n")
            for row in sorted(tracked):
                fh.write("\t".join(row) + "\n")
        print(f"wrote {len(tracked)} rows to {os.path.relpath(BASELINE, ROOT)}")
        return 0
    print("== bare-integer qtype lines (#3431) ==")
    print(f"candidates {len(hits)}: " + ", ".join(f"{k} {v}" for k, v in sorted(by_class.items())))
    if not hits:
        print("FAIL  zero candidates: the tree has integer match arms, so this is a BROKEN PATTERN. Run --self-test.")
        return 1
    stale = sorted(set(OVERRIDES) - used_overrides)
    if stale:
        print(f"FAIL  {len(stale)} OVERRIDES row(s) match no candidate — the site moved or was renamed; "
              "re-review it:")
        for r, fn in stale:
            print(f"        {r} fn {fn}")
        return 1
    base = load_baseline()
    if not base:
        print(f"FAIL  no baseline at {os.path.relpath(BASELINE, ROOT)}")
        return 1
    have = collections.Counter(tracked)
    new = have - base
    gone = base - have
    if gone:
        print(f"note  {sum(gone.values())} baseline row(s) no longer in the tree — shrink the baseline:")
        for (c, r, s), k in sorted(gone.items())[:20]:
            print(f"        {c}\t{r}\t{s}" + (f"  (x{k})" if k > 1 else ""))
    if new:
        print(f"FAIL  {sum(new.values())} new bare-integer {'/'.join(sorted({c for c, _, _ in new}))} line(s) "
              "not in the baseline. Name the id through trueno_quant::GgmlType instead:")
        lines = {(c, r, s): no for r, no, c, s in hits}
        for (c, r, s) in sorted(new):
            print(f"        {c:<8} {r}:{lines[(c, r, s)]}: {s}")
        return 1
    print(f"ok    {sum(have.values())} tracked line(s), all in the baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
