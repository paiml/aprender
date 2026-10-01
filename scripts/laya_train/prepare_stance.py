"""TweetEval stance_abortion -> a Laya data dir (D-19 as amended by A2, laya-finetune-gate-v1 1.4.0).

    uv run --project scripts/laya_train --frozen python scripts/laya_train/prepare_stance.py [--cell s64|s16] [--out DIR]
    (or: just laya-prepare-stance [s64|s16])

Source: the gitignored local dataset `apr data tweet-eval-stance --output data/tweet-eval-stance`
(train.jsonl / validation.jsonl / test.jsonl rows {id, input, label, label_text, source_split}).

--cell s64 (the default, the contract's `demo_s64`) -> data/decide/tweet-stance-64:
    train.jsonl  the 192 shots of demo_s64.selection (s64-seed13), each VERIFIED (below);
    eval.jsonl   BY RULE, `eval_set.demo_rule`: data.in_distribution_heldout over the validation split then
                 the train split, minus every shot, every exclusions.excluded_train_ids id and every
                 exclusions.groups[*].members pair; shot text overlap refused; duplicates dropped keeping the
                 first. ASSERTED to be demo_s64.eval_rows rows with demo_s64.eval_class_counts per class;
    shift.jsonl  the test split (SemEval-2016 test) in file order, ASSERTED to be demo_s64.shift_rows rows --
                 the shift probe, reported and never a gate clause.
--cell s16 (the 1.2.0 `demo` record) -> data/decide/tweet-stance-16: the s16-seed13 shots and the test split
    as eval.jsonl, byte-identical to what this script has always written.

Shot verification: every `ordered_examples[].id` is looked up in the source train split and REFUSED unless
sha256 of its input equals the manifest `exact_hash`, its nfc-trim-ws-v1 hash equals `normalized_hash`, and
its label index equals the manifest label; `label_names` must equal the contract's criteria_order.

Never overwrites: an existing non-empty --out is refused unless every file already holds exactly the bytes
this run would write (re-running is then a no-op). Output lands under the root-anchored, gitignored /data/:
tweet text is never committed and never printed (counts and sha256s only).
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import contract  # noqa: E402
from common import jsonl_bytes  # noqa: E402
from data import (DataError, decode_jsonl, exact_sha256, in_distribution_heldout, normalized_sha256,  # noqa: E402
                  refuse_unencodable, sha256_bytes)

REPO = contract.REPO
SRC = REPO / "data" / "tweet-eval-stance"
OUT = REPO / "data" / "decide" / "tweet-stance-16"          # the s16 default (kept for spike 027's reuse)
CELLS = {"s16": ("demo", "tweet-stance-16"), "s64": ("demo_s64", "tweet-stance-64")}

# The stance-abortion task exactly as spike 024 asked it (tools/tasks.py), criteria in label order.
INSTRUCTIONS = "What stance does the author of this tweet take on abortion?"
DESCRIPTIONS = {
    "none": "No stance on abortion, or the tweet is not about abortion",
    "against": "Opposes abortion (pro-life)",
    "favor": "Supports abortion rights (pro-choice)",
}


def fail(msg):
    """A consistency check on the pinned public dataset or the committed manifest failed (exit 1)."""
    print("PREPARE FAILED: " + msg, file=sys.stderr)
    sys.exit(1)


def read_jsonl(path):
    """{id: row} and the id order of a source split, rows split exactly as Rust's str::lines (data.jsonl_lines).
    An invalid UTF-8 byte or a lone-surrogate input is REFUSED source-row-encoding and a line that is not a
    JSON object with an `id` REFUSED source-row-schema (DataError -> `REFUSED <rule>`, exit 2 in main)."""
    rows = {}
    order = []
    for n, line in enumerate(decode_jsonl(path.read_bytes(), "source", path.name), 1):
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            raise DataError("source-row-schema", "%s line %d is not JSON (%s)" % (path.name, n, e))
        if not isinstance(r, dict) or "id" not in r:
            raise DataError("source-row-schema", "%s line %d is not a JSON object with an id" % (path.name, n))
        refuse_unencodable(r.get("input"), "source", path.name, n, "input")
        if r["id"] in rows:
            fail("%s line %d repeats id %s" % (path.name, n, r["id"]))
        rows[r["id"]] = r
        order.append(r["id"])
    return rows, order


def verified_shots(manifest_path, order, shots_per_class, train):
    """[(text, label name)] of a selection manifest, every shot verified against the local train split."""
    payload = json.loads((REPO / manifest_path).read_text())["payload"]
    if payload["label_names"] != order:
        fail("selection label_names %s != contract criteria_order %s" % (payload["label_names"], order))
    if int(payload["shots_per_class"]) != int(shots_per_class):
        fail("selection shots_per_class %s != contract %s" % (payload["shots_per_class"], shots_per_class))
    shots = []
    for ex in payload["ordered_examples"]:
        r = train.get(ex["id"])
        if r is None:
            fail("selection id %s is not in %s" % (ex["id"], SRC / "train.jsonl"))
        if exact_sha256(r["input"]) != ex["exact_hash"]:
            fail("selection id %s: sha256(input) != manifest exact_hash (the local dataset differs)" % ex["id"])
        if normalized_sha256(r["input"]) != ex["normalized_hash"]:
            fail("selection id %s: nfc-trim-ws-v1 hash != manifest normalized_hash" % ex["id"])
        if int(r["label"]) != int(ex["label"]) or r["label_text"] != order[int(ex["label"])]:
            fail("selection id %s: label %s/%s disagrees with manifest label %s"
                 % (ex["id"], r["label"], r["label_text"], ex["label"]))
        shots.append((r["input"], order[int(ex["label"])]))
    return payload, shots


def labelled(rows, ids, order, split):
    out = []
    for rid in ids:
        r = rows[rid]
        if r["label_text"] != order[int(r["label"])]:
            fail("%s id %s: label %s/%s is not in the contract order" % (split, rid, r["label"], r["label_text"]))
        out.append((rid, r["input"], r["label_text"]))
    return out


def build(cell, src=None):
    """{file name: bytes} for a cell, plus a printable summary (counts and hashes only). `src` is the
    source dataset dir (default data/tweet-eval-stance)."""
    src = SRC if src is None else Path(src)
    block_name, _ = CELLS[cell]
    decl = contract.gate_contract()[block_name]
    order = list(decl["criteria_order"])
    if list(DESCRIPTIONS) != order:
        fail("criteria descriptions %s are not in the contract order %s" % (list(DESCRIPTIONS), order))
    need = ["train.jsonl", "test.jsonl"] + (["validation.jsonl"] if cell == "s64" else [])
    for name in need:
        if not (src / name).is_file():
            fail("%s is missing; run: apr data tweet-eval-stance --output data/tweet-eval-stance" % (src / name))
    train, train_order = read_jsonl(src / "train.jsonl")
    test, test_order = read_jsonl(src / "test.jsonl")
    payload, shots = verified_shots(decl["selection"], order, decl["shots_per_class"], train)
    test_rows = [(t, lab) for _, t, lab in labelled(test, test_order, order, "test")]
    task = {"type": "choice", "instructions": INSTRUCTIONS, "criteria": DESCRIPTIONS}
    files = {"task.json": (json.dumps(task, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
             "train.jsonl": jsonl_bytes(shots)}
    counts = lambda rows: [sum(1 for _, x in rows if x == lab) for lab in order]  # noqa: E731
    if cell == "s16":
        if len(test_rows) != int(decl["eval_rows"]):
            fail("test split has %d rows, the contract demo declares %s" % (len(test_rows), decl["eval_rows"]))
        files["eval.jsonl"] = jsonl_bytes(test_rows)
        return files, {"train": counts(shots), "eval": counts(test_rows)}
    validation, val_order = read_jsonl(src / "validation.jsonl")
    ex = payload["exclusions"]
    members = [tuple(m) for g in ex["groups"] for m in g["members"]]
    heldout = in_distribution_heldout(         # DataError heldout-shot-overlap -> REFUSED, exit 2 (main)
        labelled(validation, val_order, order, "validation"), labelled(train, train_order, order, "train"),
        [e["id"] for e in payload["ordered_examples"]], ex["excluded_train_ids"], members, shots)
    got_counts = counts(heldout)
    if len(heldout) != int(decl["eval_rows"]) or got_counts != [int(x) for x in decl["eval_class_counts"]]:
        fail("eval_set.demo_rule built %d rows %s; demo_s64 declares %s rows %s (criteria order %s)"
             % (len(heldout), got_counts, decl["eval_rows"], list(decl["eval_class_counts"]), order))
    if len(test_rows) != int(decl["shift_rows"]):
        fail("test split has %d rows, demo_s64 declares shift_rows %s" % (len(test_rows), decl["shift_rows"]))
    shot_norm = {normalized_sha256(t) for t, _ in shots}
    if any(normalized_sha256(t) in shot_norm for t, _ in test_rows):
        fail("a shift (test) row equals a shot after nfc-trim-ws-v1; train.py would refuse shift.jsonl")
    files["eval.jsonl"] = jsonl_bytes(heldout)
    files["shift.jsonl"] = jsonl_bytes(test_rows)
    return files, {"train": counts(shots), "eval": got_counts, "shift": counts(test_rows),
                   "exclusions": {"excluded_train_ids": len(ex["excluded_train_ids"]), "group_members": len(members)}}


def write_once(out, files):
    """Write `files` into `out`; an existing non-empty `out` must already hold exactly these bytes.

    Only the files this run writes are compared, and dotfiles already in `out` (a Finder .DS_Store) are
    ignored (D3-5). Any OTHER extra file is still refused: a stale shift.jsonl beside a cell that writes
    none would be read by train.py as a shift probe. REFUSED prepare-out-dir (a DataError; exit 2 in main)."""
    present = sorted(q.name for q in out.iterdir() if not q.name.startswith(".")) if out.exists() else []
    if present:
        extra = sorted(set(present) - set(files))
        differ = sorted(n for n, b in files.items() if not (out / n).is_file() or (out / n).read_bytes() != b)
        if extra or differ:
            raise DataError("prepare-out-dir", "%s exists and differs from what this run would write (differing or "
                            "missing %s, foreign %s); refusing to overwrite" % (out, differ, extra))
        return "unchanged (byte-identical)"
    out.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (out / name).write_bytes(body)
    return "written"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cell", choices=sorted(CELLS), default="s64",
                    help="s64 (default, demo_s64: in-distribution eval by rule + shift probe) or s16 (the 1.2.0 demo)")
    ap.add_argument("--out", default=None, help="data dir (default data/decide/tweet-stance-<16|64>)")
    ap.add_argument("--src", default=None, help="source dataset dir (default data/tweet-eval-stance)")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else REPO / "data" / "decide" / CELLS[args.cell][1]
    try:
        files, summary = build(args.cell, args.src)
        status = write_once(out, files)
    except DataError as e:                     # every input refusal: one REFUSED <rule> line, exit 2, no traceback
        print(str(e), file=sys.stderr)
        return 2
    try:
        where = out.resolve().relative_to(REPO)
    except ValueError:
        where = out
    print("prepared %s (cell %s, %s): %s" % (where, args.cell, status, json.dumps(summary, sort_keys=True)))
    for name in sorted(files):
        print("  %-12s %4d lines  sha256 %s" % (name, files[name].count(b"\n"), sha256_bytes(files[name])))
    print("PREPARE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
