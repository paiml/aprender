#!/usr/bin/env python3
"""PP-ARCH-001 Phase 0 (#3422): classify every RMSNorm / RoPE / softmax /
attention implementation in aprender-serve, and ratchet the duplicates.

Spec: docs/specifications/PP-ARCH-001-MASTER.md §2.1, §3 row 0, §5.

§2.1's exit criterion is "zero new reimplementations of attention, softmax,
RMSNorm or RoPE". This script is the oracle that makes that re-derivable:

  shared     defined in one of SHARED_HOMES (the layer architectures should call)
  duplicate  outside SHARED_HOMES, name in a family AND body carries that
             family's math (the reimplementation §2.1 forbids)
  wrapper    name in a family, body has none of the math (kernel launch,
             dispatch, getter, buffer plumbing): not a reimplementation
  novel      a reviewed OVERRIDE: genuinely different operator (§2.2 slot)

Test code (`#[cfg(test)]` items, *_tests.rs, tests/) is excluded. String
literals are blanked, so PTX/WGSL source inside strings never matches.

Modes:
  --self-test       case table for the classifier
  --summary         counts per family x class, plus adoption per forward file
  --write-baseline  rewrite docs/audits/shared-block-duplicates-baseline.tsv
  (default)         gate: RED on a duplicate not in the baseline, on a baseline
                    row that no longer exists (the baseline must shrink with
                    each migration), and on a stale OVERRIDE
"""
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "crates", "aprender-serve", "src")
BASELINE = os.path.join(ROOT, "docs", "audits", "shared-block-duplicates-baseline.tsv")

# The shared layer (§1, §4). Phase 1 may move homes; that is a reviewed edit here.
SHARED_HOMES = (
    "gguf/ops.rs",
    "gguf/inference/forward/attention.rs",
    "gguf/inference/forward/ffn_block.rs",
)

# family -> (name regex, body-math regex)
FAMILIES = {
    "rmsnorm": (re.compile(r"rms_?norm"), re.compile(r"\bsqrt\b|\.sqrt\(|\brsqrt\b|\.rsqrt\(")),
    "rope": (re.compile(r"rope|rotary|rotate_half"),
             re.compile(r"\.(?:sin|cos|sin_cos)\(|\bpowf\(|\.powf\(")),
    "softmax": (re.compile(r"softmax"), re.compile(r"\.exp\(|\bexp\(")),
    # attention math = an inline exp (hand softmax), or the full QK-scale-softmax
    # chain: a softmax call plus a sqrt scale. A sqrt alone is a kernel launcher
    # computing its scale argument, not a reimplementation.
    "attention": (re.compile(r"attention|attn|sdpa|scaled_dot"),
                  re.compile(r"(?s)\.exp\(|\bexp\(|^(?=.*\bsoftmax\w*\s*\()(?=.*\bsqrt\b)")),
}
FAMILY_ORDER = ("rmsnorm", "rope", "softmax", "attention")

# (path, fn) -> (class, reason). A reviewed judgement the name/body rule gets wrong.
_FUSED_Q8 = "fused RMSNorm + Q8_0 activation quantize: no shared fused equivalent (§2.2)"
OVERRIDES = {
    ("gguf/config.rs", "attn_scale"): ("wrapper", "config getter returning 1/sqrt(head_dim)"),
    ("quantize/activation.rs", "quantize_rmsnorm_q8_0_scalar"): ("novel", _FUSED_Q8),
    ("quantize/activation.rs", "quantize_rmsnorm_q8_0_avx2"): ("novel", _FUSED_Q8),
    ("quantize/quantize_rmsnorm_into.rs", "quantize_rmsnorm_q8_0_into"): ("novel", _FUSED_Q8),
}

FN_RE = re.compile(r"\bfn\s+(\w+)")
CFG_TEST_RE = re.compile(r"#\[cfg\((?:all\()?test")
TEST_ATTR_RE = re.compile(r"#\[(?:\w+::)*test\b")

FORWARD_DIR = "gguf/inference/forward/"


def is_test_path(rel: str) -> bool:
    base = os.path.basename(rel)
    parts = rel.split("/")[:-1]
    if any(p in ("tests", "test") or p.endswith("_tests") for p in parts):
        return True
    return ("/tests/" in "/" + rel or base.endswith("_tests.rs") or base == "tests.rs"
            or base.startswith("tests_") or "_tests_" in base or base.startswith("test_"))


def strip_code(line: str) -> str:
    """Line with string/char literals blanked and `//` comments cut."""
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


def clean_lines(text: str):
    """Stripped lines; block comments and raw strings (r#"..."#) blanked."""
    out, in_block, in_raw = [], False, None
    for raw in text.splitlines():
        line = raw
        if in_raw is not None:
            end = line.find(in_raw)
            if end == -1:
                out.append("")
                continue
            line = '""' + line[end + len(in_raw):]
            in_raw = None
        if in_block:
            end = line.find("*/")
            if end == -1:
                out.append("")
                continue
            line = line[end + 2:]
            in_block = False
        m = re.search(r'\br(#+)"', line)
        if m and '"' + m.group(1) not in line[m.end():]:
            in_raw = '"' + m.group(1)
            line = line[:m.start()] + '""'
        line = strip_code(line)
        while "/*" in line:
            s = line.find("/*")
            e = line.find("*/", s + 2)
            if e == -1:
                line = line[:s]
                in_block = True
                break
            line = line[:s] + line[e + 2:]
        out.append(line)
    return out


def functions(text: str):
    """Yield (name, lineno, body) for every non-test fn with a body."""
    lines = clean_lines(text)
    depth = 0
    skip_until = None
    pending_cfg_test = False
    open_fns = []            # [name, lineno, depth_inside, body_lines]
    pending = None           # (name, lineno) waiting for its `{`
    nest = 0                 # () / [] depth, so `[f32; 4]` is not a `;`
    for no, line in enumerate(lines, 1):
        if skip_until is None and (CFG_TEST_RE.search(line) or TEST_ATTR_RE.search(line)):
            pending_cfg_test = True
        fn_at = {m.start(): m.group(1) for m in FN_RE.finditer(line)}
        for fn in open_fns:
            fn[3].append(line)
        for col, ch in enumerate(line):
            if col in fn_at and skip_until is None:
                pending = (fn_at[col], no)
                nest = 0
            if ch == "{":
                depth += 1
                if pending_cfg_test and skip_until is None:
                    skip_until = depth - 1
                    pending_cfg_test = False
                    pending = None
                elif pending is not None and skip_until is None:
                    open_fns.append([pending[0], pending[1], depth, [line]])
                    pending = None
            elif ch == "}":
                if open_fns and open_fns[-1][2] == depth:
                    name, lno, _, body = open_fns.pop()
                    yield name, lno, "\n".join(body)
                depth -= 1
                if skip_until is not None and depth == skip_until:
                    skip_until = None
            elif ch in "([":
                nest += 1
            elif ch in ")]":
                nest -= 1
            elif ch == ";" and nest == 0:
                pending = None     # trait method / extern declaration: no body
        if pending_cfg_test and skip_until is None and line.strip().endswith(";"):
            pending_cfg_test = False   # #[cfg(test)] on a `use`/`mod x;` line


def family_of(name: str):
    for fam in FAMILY_ORDER:
        if FAMILIES[fam][0].search(name):
            return fam
    return None


def classify_fn(rel: str, name: str, body: str):
    fam = family_of(name)
    if fam is None:
        return None, None
    if (rel, name) in OVERRIDES:
        return fam, OVERRIDES[(rel, name)][0]
    if rel in SHARED_HOMES:
        return fam, "shared"
    return fam, "duplicate" if FAMILIES[fam][1].search(body) else "wrapper"


def classify_text(rel: str, text: str):
    for name, lno, body in functions(text):
        fam, cls = classify_fn(rel, name, body)
        if fam is not None:
            yield fam, cls, name, lno


def shared_names(src: str):
    """{home: set of fn names defined there (non-test)}."""
    out = {}
    for rel in SHARED_HOMES:
        with open(os.path.join(src, rel), encoding="utf-8", errors="replace") as fh:
            out[rel] = {name for name, _, _ in functions(fh.read())}
    return out


def iter_sources(src: str):
    for dirpath, _, files in os.walk(src):
        for f in sorted(files):
            if not f.endswith(".rs"):
                continue
            path = os.path.join(dirpath, f)
            rel = os.path.relpath(path, src)
            if is_test_path(rel):
                continue
            with open(path, encoding="utf-8", errors="replace") as fh:
                yield rel, fh.read()


def scan(src: str):
    rows, seen_overrides = [], set()
    for rel, text in iter_sources(src):
        for fam, cls, name, lno in classify_text(rel, text):
            rows.append((fam, cls, rel, name, lno))
            if (rel, name) in OVERRIDES:
                seen_overrides.add((rel, name))
    return sorted(rows), seen_overrides


def adoption(src: str):
    """Per forward file: free/path calls (not `.method(`) into each shared home,
    by callee name, excluding names the file defines itself."""
    homes = shared_names(src)
    out = {}
    for rel, text in iter_sources(src):
        if not rel.startswith(FORWARD_DIR) or rel in SHARED_HOMES:
            continue
        own = {n for n, _, _ in functions(text)}
        body = "\n".join(clean_lines(text))
        row = {}
        for home, names in homes.items():
            names = sorted(names - own)
            if not names:
                row[home] = 0
                continue
            pat = re.compile(r"(?:(?<=self\.)|(?<![.\w]))(?:\w+::)*(" + "|".join(map(re.escape, names)) + r")\s*\(")
            row[home] = sum(1 for m in pat.finditer(body)
                            if not re.search(r"\bfn\s*$", body[max(0, m.start() - 8):m.start()]))
        out[rel] = row
    return out


def dup_keys(rows):
    return Counter((fam, rel, name) for fam, cls, rel, name, _ in rows if cls == "duplicate")


def load_baseline():
    c = Counter()
    if not os.path.exists(BASELINE):
        return c
    with open(BASELINE, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            fam, rel, name = line.rstrip("\n").split("\t")
            c[(fam, rel, name)] += 1
    return c


def self_test() -> int:
    cases = [
        # (rel, source, expected [(family, class, name)])
        ("x/norm.rs", "fn rms_norm(x: &[f32]) -> f32 { let s: f32 = x.iter().map(|v| v*v).sum(); s.sqrt() }",
         [("rmsnorm", "duplicate", "rms_norm")]),
        ("x/norm.rs", "pub fn rmsnorm_gpu(&mut self) -> Result<()> { self.launch(k) }",
         [("rmsnorm", "wrapper", "rmsnorm_gpu")]),
        ("gguf/ops.rs", "pub fn rms_norm(x: &[f32]) -> f32 { x[0].sqrt() }",
         [("rmsnorm", "shared", "rms_norm")]),
        ("x/r.rs", "fn apply_rope(q: &mut [f32], pos: usize) {\n let (s, c) = t.sin_cos();\n}",
         [("rope", "duplicate", "apply_rope")]),
        ("x/r.rs", "fn rope_theta(&self) -> f32 { self.theta }", [("rope", "wrapper", "rope_theta")]),
        ("x/r.rs", "fn rotate_half(x: &mut [f32]) { let f = 10000f32.powf(-1.0); }",
         [("rope", "duplicate", "rotate_half")]),
        ("x/s.rs", "fn softmax(v: &mut [f32]) { for x in v { *x = x.exp(); } }",
         [("softmax", "duplicate", "softmax")]),
        ("x/s.rs", "fn softmax(v: &mut [f32]) { ops::softmax(v) }", [("softmax", "wrapper", "softmax")]),
        ("x/a.rs", "fn attention(q: &[f32]) -> f32 { let s = 1.0 / (d as f32).sqrt(); (q[0] * s).exp() }",
         [("attention", "duplicate", "attention")]),
        ("x/a.rs", "fn attn_output_buffer(&self) -> &Buf { &self.b }",
         [("attention", "wrapper", "attn_output_buffer")]),
        # math inside a string (PTX) is not math
        ("x/k.rs", 'fn rmsnorm_ptx() -> String { "sqrt.approx.f32 %f1".to_string() }',
         [("rmsnorm", "wrapper", "rmsnorm_ptx")]),
        # math inside a raw string spanning lines is not math
        ("x/k.rs", 'fn rope_ptx() -> &\'static str {\n r#"\n cos.approx.f32 x.cos()\n "#\n}',
         [("rope", "wrapper", "rope_ptx")]),
        # math inside a comment is not math
        ("x/k.rs", "fn softmax_launch(&self) {\n // x.exp() on device\n self.go()\n}",
         [("softmax", "wrapper", "softmax_launch")]),
        # a #[cfg(test)] module is excluded
        ("x/t.rs", "#[cfg(test)]\nmod tests {\n fn rms_norm_ref(x: f32) -> f32 { x.sqrt() }\n}", []),
        # a test fn after a cfg(test) item closes is excluded, the next real fn is not
        ("x/t.rs", "#[cfg(test)]\nmod tests { fn softmax(x: f32) -> f32 { x.exp() } }\n"
                   "fn softmax_real(x: f32) -> f32 { x.exp() }",
         [("softmax", "duplicate", "softmax_real")]),
        # #[cfg(test)] on a `mod x;` line does not swallow the next fn
        ("x/t.rs", "#[cfg(test)]\nmod tests;\nfn softmax(x: f32) -> f32 { x.exp() }",
         [("softmax", "duplicate", "softmax")]),
        # nested fn: both are classified by their own body
        ("x/n.rs", "fn attention(q: f32) -> f32 {\n fn softmax(x: f32) -> f32 { x.exp() }\n softmax(q)\n}",
         [("softmax", "duplicate", "softmax"), ("attention", "duplicate", "attention")]),
        # trait declaration without body is not a fn with a body
        ("x/tr.rs", "trait T { fn rms_norm(&self, x: &[f32]);\n fn other(&self) { let y = 1; } }", []),
        # a #[test] fn is excluded
        ("x/m.rs", "#[test]\nfn test_flash_attention(q: f32) -> f32 { q.exp() }", []),
        ("x/m.rs", "#[tokio::test]\nasync fn softmax_case() { let y = x.exp(); }", []),
        # a kernel launcher computing its scale is not attention math
        ("x/a.rs", "fn prefill_attention_cublas(&mut self) { for l in 0..n { let s = (d as f32).sqrt(); self.go(s); } }",
         [("attention", "wrapper", "prefill_attention_cublas")]),
        # QK-scale-softmax chain calling a softmax helper is a reimplementation
        ("x/a.rs", "fn attention_with_cache(q: &[f32]) {\n let s = (d as f32).sqrt();\n softmax_simd(&mut sc);\n}",
         [("attention", "duplicate", "attention_with_cache")]),
        # an override wins over the rule
        ("gguf/config.rs", "fn attn_scale(&self) -> f32 { 1.0 / (self.d as f32).sqrt() }",
         [("attention", "wrapper", "attn_scale")]),
        # unrelated names never enter the table
        ("x/u.rs", "fn layer_norm(x: &[f32]) -> f32 { x[0].sqrt() }", []),
        # rmsnorm wins over attention when both appear (family order)
        ("x/o.rs", "fn attn_rms_norm(x: f32) -> f32 { x.sqrt() }", [("rmsnorm", "duplicate", "attn_rms_norm")]),
    ]
    fails = 0
    for rel, src, want in cases:
        got = [(fam, cls, name) for fam, cls, name, _ in classify_text(rel, src)]
        if sorted(got) != sorted(want):
            fails += 1
            print(f"FAIL {rel}: {src[:60]!r}\n  want {want}\n  got  {got}")
    for p, want in [("a/b_tests.rs", True), ("a/tests/x.rs", True), ("a/tests.rs", True),
                    ("a/single_tests_q8k.rs", True), ("a/attention.rs", False),
                    ("inference/coverage_tests/rms_norm.rs", True)]:
        if is_test_path(p) != want:
            fails += 1
            print(f"FAIL is_test_path({p}) != {want}")
    total = len(cases) + 6
    print(f"self-test: {total - fails}/{total} pass")
    return 1 if fails else 0


def summary(rows, src):
    counts = Counter((fam, cls) for fam, cls, *_ in rows)
    classes = ("shared", "duplicate", "wrapper", "novel")
    print("family      " + "".join(f"{c:>10}" for c in classes))
    for fam in FAMILY_ORDER:
        print(f"{fam:<12}" + "".join(f"{counts[(fam, c)]:>10}" for c in classes))
    dup_files = {rel for fam, cls, rel, *_ in rows if cls == "duplicate"}
    print(f"duplicate files: {len(dup_files)}")
    ad = adoption(src)
    blocks = SHARED_HOMES[1:]
    print(f"forward files calling the shared layer: "
          f"ops {sum(1 for r in ad.values() if r[SHARED_HOMES[0]])}/{len(ad)}, "
          f"attention/ffn blocks {sum(1 for r in ad.values() if any(r[b] for b in blocks))}/{len(ad)}")
    print("   ops  blocks  file")
    for rel in sorted(ad, key=lambda r: (sum(ad[r].values()), r)):
        r = ad[rel]
        print(f"  {r[SHARED_HOMES[0]]:>4}  {sum(r[b] for b in blocks):>6}  {rel}")


def main(argv) -> int:
    if "--self-test" in argv:
        return self_test()
    rows, seen = scan(SRC)
    if "--summary" in argv:
        summary(rows, SRC)
        return 0
    cur = dup_keys(rows)
    if "--write-baseline" in argv:
        with open(BASELINE, "w", encoding="utf-8") as fh:
            fh.write("# PP-ARCH-001 Phase 0 (#3422): duplicate RMSNorm/RoPE/softmax/attention\n"
                     "# implementations outside the shared layer. Generated by\n"
                     "# scripts/classify_shared_block_adoption.py --write-baseline. This file may\n"
                     "# only SHRINK: a migration deletes rows; a new row is a new duplicate.\n"
                     "# family\tpath (under crates/aprender-serve/src)\tfn\n")
            for (fam, rel, name), n in sorted(cur.items()):
                for _ in range(n):
                    fh.write(f"{fam}\t{rel}\t{name}\n")
        print(f"wrote {sum(cur.values())} rows to {os.path.relpath(BASELINE, ROOT)}")
        return 0
    base = load_baseline()
    red = []
    for k, n in sorted(cur.items()):
        if n > base[k]:
            red.append(f"NEW duplicate {k[0]} {k[1]}::{k[2]} (x{n - base[k]}): call the shared layer "
                       f"({', '.join(SHARED_HOMES)}) instead, or register a novel operator in OVERRIDES "
                       f"with a reason (PP-ARCH-001 §2.2)")
    for k, n in sorted(base.items()):
        if n > cur[k]:
            red.append(f"STALE baseline row {k[0]} {k[1]}::{k[2]} (x{n - cur[k]}): it is gone, delete it "
                       f"from {os.path.relpath(BASELINE, ROOT)} so the ratchet keeps the gain")
    for k in sorted(set(OVERRIDES) - seen):
        red.append(f"STALE override {k[0]}::{k[1]}: no such fn, delete it from OVERRIDES")
    for r in red:
        print("RED  " + r)
    print(f"shared-block ratchet: {sum(cur.values())} duplicates (baseline {sum(base.values())}), "
          f"{len(red)} violation(s)")
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
