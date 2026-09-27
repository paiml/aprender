#!/usr/bin/env python3
"""distill_vocab_identity.py - measure whether a distillation teacher and student share a vocab.

apr-distill-teacher-vocab-alignment-v1 truncates teacher logits to the student's vocab and
ASSUMES the first N tokens of both models are the same tokens (the "shared tokenizer prefix"
invariant, left to the operator). This script turns that assumption into a measurement: it
reads tokenizer.ggml.{tokens,merges,token_type} from two GGUFs and reports lengths, a sha256
over each list, the first differing index, and every other tokenizer.* key that differs.

Usage:
  distill_vocab_identity.py TEACHER.gguf STUDENT.gguf [--out FILE]
Exit:
  0  the student's token list is a prefix of the teacher's (identical when lengths match)
  1  it is not: the truncation in the contract would align logits of DIFFERENT tokens
  2  usage or read error
"""
import argparse
import hashlib
import json
import os
import sys

LISTS = ("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_type")


def sha(xs):
    m = hashlib.sha256()
    for x in xs:
        m.update(x if isinstance(x, bytes) else str(x).encode())
        m.update(b"\0")
    return m.hexdigest()


def read(path):
    from gguf import GGUFReader  # pip: gguf (llama.cpp's reader)

    r = GGUFReader(path, "r")
    lists, scalars = {}, {}
    for k, f in r.fields.items():
        if not k.startswith("tokenizer."):
            continue
        if k in LISTS:
            if k == "tokenizer.ggml.token_type":
                lists[k] = [int(f.parts[i][0]) for i in f.data]
            else:
                lists[k] = [bytes(f.parts[i]) for i in f.data]
        elif f.data:
            v = f.parts[f.data[0]]
            is_str = f.types and f.types[0].name == "STRING"
            scalars[k] = bytes(v).decode() if is_str else v.tolist()
    arch_f = r.fields.get("general.architecture")
    arch = bytes(arch_f.parts[arch_f.data[0]]).decode() if arch_f else None
    width_f = r.fields.get(f"{arch}.embedding_length") if arch else None
    width = int(width_f.parts[width_f.data[0]][0]) if width_f else None
    tensors = {t.name for t in r.tensors}
    return lists, scalars, {
        "architecture": arch,
        "embedding_length": width,
        "tied_embeddings": "output.weight" not in tensors,
    }


def compare(teacher, student):
    tl, ts, tm = read(teacher)
    sl, ss, sm = read(student)
    rep = {
        "teacher": {"file": os.path.basename(teacher), **tm},
        "student": {"file": os.path.basename(student), **sm},
        "lists": {},
    }
    for k in LISTS:
        a, b = tl.get(k), sl.get(k)
        n = min(len(a or []), len(b or []))
        rep["lists"][k] = {
            "teacher_len": None if a is None else len(a),
            "student_len": None if b is None else len(b),
            "teacher_sha256": None if a is None else sha(a),
            "student_sha256": None if b is None else sha(b),
            "identical": a == b,
            "first_diff": next((i for i in range(n) if a[i] != b[i]), None),
        }
    rep["other_tokenizer_keys_differing"] = sorted(
        k for k in set(ts) | set(ss) if ts.get(k) != ss.get(k)
    )
    tok = rep["lists"]["tokenizer.ggml.tokens"]
    rep["student_vocab_is_teacher_prefix"] = (
        tok["teacher_len"] is not None
        and tok["student_len"] is not None
        and tok["student_len"] <= tok["teacher_len"]
        and tok["first_diff"] is None
    )
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("teacher")
    ap.add_argument("student")
    ap.add_argument("--out")
    a = ap.parse_args()
    try:
        rep = compare(a.teacher, a.student)
    except (OSError, ImportError, KeyError, ValueError) as e:
        print(f"distill_vocab_identity: cannot read: {e}", file=sys.stderr)
        return 2
    text = json.dumps(rep, indent=2, sort_keys=True) + "\n"
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 0 if rep["student_vocab_is_teacher_prefix"] else 1


if __name__ == "__main__":
    sys.exit(main())
