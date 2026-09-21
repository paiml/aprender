"""The Q4_K universe's definition -- ONE place, read by the model ladder's producer and judge (#3763, #3712
row A2) and by any other gate that asks "is this file Q4_K?" (#3742).

Issue author, #3712 done_when 1: "filename is not evidence of quantization". A file's quantization is read
from its TENSOR HEADER (`apr tensors --json`): its dominant dtypes are the ones at the maximum COUNT among its
tensors of 2 or more dimensions, and it is a member of the DTYPE universe iff DTYPE is one of them. By count,
not bytes: a Q4_K_M file's large Q6_K embedding outweighs its Q4_K matrices in bytes (measured,
Qwen3.5-0.8B-Q4_K_M: Q6_K by bytes, Q4_K 98 of 205 by count). A tie at the top is a member -- the strict side:
an included file must prove itself, an excluded one is never looked at. 1-D tensors (norms, biases) do not
vote. The file's NAME never reaches this module.

    histogram(tensors) -> {DTYPE: count}      dominant(counts) -> [DTYPE, ...]
    is_member(counts, dtype) -> bool          row(file, rc, header_json, dtype, size) -> receipt row

CLI (the shell wrapper scripts/lib/tensor_universe.sh calls these):
    tensor_universe.py row <file> <rc> <header.json> <DTYPE> <bytes>   one candidate row (JSON) for a receipt
    tensor_universe.py dominant <header.json>        the dominant dtypes, space-separated; exit 2 if unreadable
    tensor_universe.py member <header.json> <DTYPE>  exit 0 member, 1 not a member, 2 unreadable
    tensor_universe.py --self-test                   the definition's own case table
"""
import collections
import json
import sys


def histogram(tensors):
    """Dtype -> count over the tensors of 2 or more dimensions, dtype names upper-cased (APR says q4_k)."""
    return dict(collections.Counter(str(t["dtype"]).upper() for t in tensors if len(t.get("shape") or []) >= 2))


def dominant(counts):
    top = max(counts.values(), default=0)
    return sorted(k for k, v in counts.items() if v == top) if top > 0 else []


def is_member(counts, dtype):
    """The universe rule: DTYPE is a most-frequent >= 2-D tensor dtype (ties are members)."""
    return str(dtype).upper() in dominant({str(k).upper(): int(v) for k, v in counts.items()})


def read_header(path):
    counts = histogram(json.load(open(path))["tensors"])
    if not counts:
        raise ValueError("no tensor of 2 or more dimensions")
    return counts


def row(file, rc, header_json, dtype, size):
    r = {"file": file, "bytes": int(size)}
    try:
        if int(rc) != 0:
            raise ValueError(f"apr tensors exited {rc}")
        counts = read_header(header_json)
        r.update(dtype_counts=dict(sorted(counts.items())), dominant=dominant(counts), member=is_member(counts, dtype))
    except Exception as e:  # an unread header is recorded as such -- the judge names it, never excludes it silently
        r.update(error=str(e)[:160], member=False)
    return r


def _self_test():
    def t(dtype, *shape):
        return {"dtype": dtype, "shape": list(shape)}
    q4km = [t("Q4_K", 8, 8)] * 98 + [t("Q5_K", 8, 8)] * 36 + [t("Q8_0", 8, 8)] * 36 + [t("F32", 8, 8)] * 18 \
        + [t("Q6_K", 248320, 1024)] * 17 + [t("F32", 1024)] * 133   # measured Qwen3.5-0.8B-Q4_K_M shape
    cases = [
        ("a Q4_K_M file is a member by count although Q6_K dominates its bytes", q4km, "Q4_K", True),
        ("a tie at the top is a member (the strict side)", [t("Q4_K", 2, 2)] * 10 + [t("Q6_K", 2, 2)] * 10, "Q4_K", True),
        ("1-D tensors do not vote", [t("Q4_K", 2, 2)] * 3 + [t("F32", 64)] * 500, "Q4_K", True),
        ("a Q5_0-dominant file is not a member, whatever it is named", [t("Q5_0", 2, 2)] * 50 + [t("Q4_K", 2, 2)] * 3, "Q4_K", False),
        ("APR's lower-case dtype names count", [t("q4_k", 2, 2)] * 169 + [t("q6_k", 2, 2)] * 29, "Q4_K", True),
        ("an f16 file is not a member", [t("F16", 2, 2)] * 200, "Q4_K", False),
    ]
    bad = 0
    for what, tensors, dtype, want in cases:
        got = is_member(histogram(tensors), dtype)
        print(f"{'ok  ' if got == want else 'FAIL'}  tensor_universe: {what}")
        bad += got != want
    if histogram([t("F32", 64)]) == {} and dominant({}) == []:
        print("ok    tensor_universe: a header with no >= 2-D tensor has no dominant dtype (row() records it unreadable)")
    else:
        print("FAIL  tensor_universe: a 1-D-only header produced a dominant dtype")
        bad += 1
    print(f"tensor_universe self-test: {len(cases) + 1} case(s), {bad} bad")
    return 1 if bad else 0


def main(argv):
    if argv[:1] == ["--self-test"]:
        return _self_test()
    if argv[:1] == ["row"] and len(argv) == 6:
        print(json.dumps(row(argv[1], argv[2], argv[3], argv[4], argv[5])))
        return 0
    if argv[:1] in (["dominant"], ["member"]) and len(argv) >= 2:
        try:
            counts = read_header(argv[1])
        except Exception as e:
            print(f"unreadable: {e}", file=sys.stderr)
            return 2
        if argv[0] == "dominant":
            print(" ".join(dominant(counts)))
            return 0
        return 0 if len(argv) == 3 and is_member(counts, argv[2]) else 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
