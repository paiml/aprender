#!/usr/bin/env python3
"""SIMD Property Checker - Pre-commit validation for SIMD code quality."""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

_TGT = re.compile(r'#\[target_feature\(enable\s*=\s*"([^"]+)"\)\]')
_FN = re.compile(r"^\s*unsafe\s+fn\s+(\w+)")
_SAFE = re.compile(r"//\s*SAFETY:", re.IGNORECASE)
_INL = re.compile(r"#\[inline")
_FMA = {"_mm256_fmadd_ps", "_mm256_fmsub_ps", "_mm_fmadd_ps", "_mm_fmsub_ps"}


@dataclass(frozen=True)
class Backend:
    """SIMD backend config."""
    name: str
    pat: str
    feat: str
    path: str


@dataclass(frozen=True)
class Issue:
    """Detected issue."""
    crit: bool
    path: str
    line: int
    fn: str
    msg: str


BACKENDS = (
    Backend("SSE2", r"_mm_\w+", "sse2", "src/backends/sse2.rs"),
    Backend("AVX2", r"_mm256_\w+", "avx2", "src/backends/avx2.rs"),
    Backend("AVX512", r"_mm512_\w+", "avx512f", "src/backends/avx512.rs"),
    Backend("NEON", r"v(?:ld|st|add|sub|mul).*q_f32", "neon", "src/backends/neon.rs"),
)


def _above(lines: list[str], end: int, pat: re.Pattern, n: int = 5) -> bool:
    return any(pat.search(lines[i]) for i in range(max(0, end - n), end))


def _feat(lines: list[str], ln: int) -> str | None:
    for i in range(max(0, ln - 5), ln):
        if m := _TGT.search(lines[i]):
            return m.group(1)
    return None


def _intrin(lines: list[str], start: int, pat: re.Pattern) -> set[str]:
    res, d = set(), 0
    for i in range(start, len(lines)):
        d += lines[i].count("{") - lines[i].count("}")
        res.update(pat.findall(lines[i]))
        if d == 0 and i > start:
            break
    return res


def _feat_mismatch(ins: set[str], ft: str | None) -> str | None:
    """Return feature-mismatch message, or None if features are correct."""
    if not ft:
        return "no target_feature"
    if any(i.startswith("_mm512_") for i in ins) and "avx512" not in ft:
        return "need avx512f"
    if any(i.startswith("_mm256_") for i in ins) and ft == "sse2":
        return "need avx2"
    if ins & _FMA and "fma" not in ft:
        return "need fma"
    return None


def _annotation_warnings(lines: list[str], ln: int, path: str, nm: str) -> list[Issue]:
    """Check for missing SAFETY comment and inline attribute."""
    warns = []
    if not _above(lines, ln, _SAFE, 10):
        warns.append(Issue(False, path, ln + 1, nm, "no SAFETY"))
    if not _above(lines, ln, _INL):
        warns.append(Issue(False, path, ln + 1, nm, "no inline"))
    return warns


def _chk(lines: list[str], ln: int, nm: str, be: Backend, pat: re.Pattern) -> list[Issue]:
    ins = _intrin(lines, ln, pat)
    if not ins:
        return []
    iss = []
    mismatch = _feat_mismatch(ins, _feat(lines, ln))
    if mismatch:
        iss.append(Issue(True, be.path, ln + 1, nm, mismatch))
    iss.extend(_annotation_warnings(lines, ln, be.path, nm))
    return iss


def _file(be: Backend) -> list[Issue]:
    p = Path(be.path)
    if not p.exists():
        return []
    lines, pat, iss = p.read_text().splitlines(), re.compile(be.pat), []
    for i, ln in enumerate(lines):
        if m := _FN.match(ln):
            iss.extend(_chk(lines, i, m.group(1), be, pat))
    return iss


def _print_section(label: str, items: list[Issue], limit: int = 0) -> None:
    """Print a labeled section of issues, optionally capping displayed count."""
    if not items:
        return
    print(f"\n{label} ({len(items)}):")
    shown = items[:limit] if limit else items
    for i in shown:
        print(f"  {i.path}:{i.line} {i.fn}() - {i.msg}")
    remaining = len(items) - len(shown)
    if remaining > 0:
        print(f"  +{remaining} more")


def _out(iss: list[Issue]) -> int:
    c, w = [i for i in iss if i.crit], [i for i in iss if not i.crit]
    _print_section("CRITICAL", c)
    _print_section("WARN", w, limit=5)
    print(f"\n{len(c)} critical, {len(w)} warn")
    return 1 if c else 0


def main() -> int:
    print("SIMD Checker\n" + "=" * 40)
    iss = [i for b in BACKENDS for i in _file(b)]
    return _out(iss) if iss else (print("PASS"), 0)[1]


if __name__ == "__main__":
    sys.exit(main())
