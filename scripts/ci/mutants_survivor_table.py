#!/usr/bin/env python3
"""mutants_survivor_table.py -- the gate for a PR too big for the one-job mutants section (#4587 A').

Operator ruling (2026-09-29 06:36Z, via the cop): for PR-1 and PR-2 of 0.70, CI itself
generates the survivor table. The split runs are CI matrix jobs (`mutants-shard`), each
bound to the PR head sha and the PR diff hash; a hand-made table does not count. This
script verifies the table BEFORE trusting it, and any gap is RED:

  1. every shard 1..N delivered meta.json + list.txt + outcomes.json, once each;
  2. every shard's head_sha == the PR head sha, diff_sha256 == sha256 of the PR diff,
     and shards == N;
  3. every shard listed the SAME universe (the full --list of the diff), and it is not empty;
  4. the union of the shards' tested mutants == that universe, each id exactly once
     (a multiset compare: nothing missing, nothing extra, nothing tested twice);
  5. each mutant is killed (CaughtMutant, or Unviable: the compiler rejects it),
     equivalent (MissedMutant/Timeout with a row in the equivalents file), or survived;
  6. an equivalents row carries a one-line proof and names a quorum receipt in the repo --
     evidence/pr-review/<pr>/<sha>/receipt.intoto.jsonl, an in-toto pr-review statement --
     that itself contains the mutant id (any other file naming the id is RED), whose
     reviewer_actor is not its author_actor, and whose <receipt>.minisig verifies with
     minisign against --pubkey. CI passes the BASE branch's .github/pr-review.pub, so a
     PR cannot sign its own proof: only the CI signer holds the secret half. No --pubkey,
     or no minisign, with an equivalents row is RED, never a skip; a row for an id outside
     the universe is RED;
  7. survived <= --max-missed (the existing limit, MUTANTS_MAX_MISSED, default 0).

  mutants_survivor_table.py check --dir D --shards N --head-sha SHA --diff pr.diff
                                  [--equiv F] [--max-missed M] [--repo R] [--table OUT.tsv]
  mutants_survivor_table.py --self-test
exit 0 GREEN . 1 RED . 2 usage
"""
import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

KILLED = {"CaughtMutant": "killed", "Unviable": "killed"}
UNKILLED = {"MissedMutant", "Timeout"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def read_list(path):
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


RECEIPT_PATH = re.compile(r"evidence/pr-review/[0-9]+/[0-9a-f]{40}/receipt\.intoto\.jsonl")


def load_equiv(path, repo, pubkey=None):
    """-> ({id: proof}, [errors]). TSV: id <TAB> one-line proof <TAB> receipt path (repo-relative)."""
    eq, errs = {}, []
    if not path:
        return eq, errs
    if not os.path.isfile(path):
        return eq, [f"equivalents file {path} does not exist"]
    with open(path, encoding="utf-8") as f:
        for n, ln in enumerate(f, 1):
            ln = ln.rstrip("\n")
            if not ln.strip() or ln.startswith("#"):
                continue
            cols = ln.split("\t")
            if len(cols) != 3 or not all(c.strip() for c in cols):
                errs.append(f"equivalents line {n}: want 3 non-empty tab-separated columns (id, proof, receipt)")
                continue
            mid, proof, receipt = cols
            if mid in eq:
                errs.append(f"equivalents line {n}: {mid} listed twice")
                continue
            if not RECEIPT_PATH.fullmatch(receipt):
                errs.append(f"equivalents line {n}: {receipt} is not a pr-review receipt path "
                            "(evidence/pr-review/<pr>/<sha>/receipt.intoto.jsonl)")
                continue
            rp = os.path.join(repo, receipt)
            if not os.path.isfile(rp):
                errs.append(f"equivalents line {n}: quorum receipt {receipt} does not exist")
                continue
            with open(rp, encoding="utf-8", errors="replace") as r:
                text = r.read()
            try:
                st = json.loads(text.splitlines()[0]) if text.strip() else {}
            except ValueError:
                st = {}
            if not (isinstance(st, dict) and st.get("_type") == "https://in-toto.io/Statement/v1"
                    and "pr-review" in str(st.get("predicateType", ""))):
                errs.append(f"equivalents line {n}: {receipt} is not an in-toto pr-review statement")
                continue
            if mid not in text:
                errs.append(f"equivalents line {n}: quorum receipt {receipt} does not name {mid}")
                continue
            pred = st.get("predicate") if isinstance(st.get("predicate"), dict) else {}
            author = str((pred.get("author_actor") or {}).get("id", ""))
            reviewer = str((pred.get("reviewer_actor") or {}).get("id", ""))
            if not author or not reviewer or author == reviewer:
                errs.append(f"equivalents line {n}: {receipt} is a self-review or names no actors "
                            f"(author '{author}', reviewer '{reviewer}')")
                continue
            if not pubkey or not os.path.isfile(pubkey):
                errs.append(f"equivalents line {n}: no --pubkey to verify {receipt}: an unverified proof is not one")
                continue
            if not shutil.which("minisign"):
                errs.append(f"equivalents line {n}: minisign is not on PATH: {receipt} cannot be verified")
                continue
            if not os.path.isfile(rp + ".minisig"):
                errs.append(f"equivalents line {n}: {receipt}.minisig is missing: the receipt is unsigned")
                continue
            v = subprocess.run(["minisign", "-V", "-q", "-m", rp, "-x", rp + ".minisig", "-p", pubkey],
                               capture_output=True, text=True)
            if v.returncode != 0:
                errs.append(f"equivalents line {n}: {receipt} signature does not verify against {pubkey}")
                continue
            eq[mid] = proof
    return eq, errs


def check(d, shards, head_sha, diff, equiv, max_missed, repo, table_out=None, pubkey=None):
    errs = []
    if not os.path.isfile(diff):
        return [f"PR diff {diff} does not exist"], None
    want_diff = sha256_file(diff)
    universe = None
    tested = collections.Counter()
    status = {}
    runs = []
    for k in range(1, shards + 1):
        sd = os.path.join(d, f"mutants-shard-{k}")
        paths = {n: os.path.join(sd, n) for n in ("meta.json", "list.txt", "outcomes.json")}
        missing = [n for n, p in paths.items() if not os.path.isfile(p)]
        if missing:
            errs.append(f"shard {k}/{shards}: no {', '.join(missing)} (a shard that delivered nothing is not a pass)")
            continue
        try:
            meta = json.load(open(paths["meta.json"], encoding="utf-8"))
            oc = json.load(open(paths["outcomes.json"], encoding="utf-8"))
        except (OSError, ValueError) as e:
            errs.append(f"shard {k}: unparseable meta/outcomes: {e}")
            continue
        if meta.get("shard") != k or meta.get("shards") != shards:
            errs.append(f"shard {k}: meta says shard {meta.get('shard')}/{meta.get('shards')}, want {k}/{shards}")
        if meta.get("head_sha") != head_sha:
            errs.append(f"shard {k}: ran on {meta.get('head_sha')}, not the PR head {head_sha}")
        if meta.get("diff_sha256") != want_diff:
            errs.append(f"shard {k}: diff sha256 {meta.get('diff_sha256')} != the PR diff's {want_diff}")
        runs.append(f"{meta.get('run_id')}/{meta.get('run_attempt')}")
        lst = read_list(paths["list.txt"])
        if universe is None:
            universe = lst
        elif collections.Counter(lst) != collections.Counter(universe):
            errs.append(f"shard {k}: listed a different universe ({len(lst)}) than shard 1 ({len(universe)})")
        for o in oc.get("outcomes", []):
            sc = o.get("scenario")
            if not isinstance(sc, dict) or "Mutant" not in sc:
                continue
            mid = sc["Mutant"].get("name")
            if not mid:
                errs.append(f"shard {k}: an outcome with no mutant name")
                continue
            tested[mid] += 1
            status[mid] = o.get("summary")
    if universe is not None and not universe:
        errs.append("the listed universe is empty: a table of nothing is not a judged diff")
    if errs:
        return errs, None
    want = collections.Counter(universe)
    for mid in sorted((want - tested).keys()):
        errs.append(f"MISSING  {mid} (listed, never tested)")
    for mid in sorted((tested - want).keys()):
        extra = tested[mid] - want.get(mid, 0)
        errs.append(f"EXTRA    {mid} (tested {extra} more time(s) than listed)")
    eq, eerrs = load_equiv(equiv, repo, pubkey)
    errs += eerrs
    for mid in sorted(set(eq) - set(want)):
        errs.append(f"equivalent row for {mid}, which is not in the listed universe (stale proof)")
    counts = collections.Counter()
    rows, survived = [], []
    for mid in sorted(set(universe)):
        s = status.get(mid)
        if s in KILLED:
            v = "killed"
        elif s in UNKILLED and mid in eq:
            v = "equivalent"
        elif s in UNKILLED:
            v = "survived"
            survived.append(mid)
        else:
            errs.append(f"UNJUDGED {mid}: outcome {s!r} is neither killed, missed nor timeout")
            continue
        counts[v] += want[mid]
        rows.append(f"{mid}\t{v}\t{s}\t{eq.get(mid, '')}")
    if table_out:
        with open(table_out, "w", encoding="utf-8") as f:
            f.write(f"# head {head_sha} diff-sha256 {want_diff} runs {' '.join(sorted(set(runs)))}\n")
            f.write("\n".join(rows) + "\n")
    print(f"listed {len(universe)} . killed {counts['killed']} . equivalent {counts['equivalent']} . "
          f"survived {counts['survived']} (limit {max_missed}) . runs {' '.join(sorted(set(runs)))}")
    if len(survived) > max_missed:
        errs.append(f"{len(survived)} survivor(s) > limit {max_missed}:")
        errs += [f"  SURVIVED {m}" for m in survived]
    return errs, counts


def self_test():
    """The case table (G3): every planted defect is RED, the complete table is GREEN."""
    ids = [f"crates/x/src/a.rs:{n}:5: replace f{n} -> bool with true" for n in range(1, 8)]
    head = "a" * 40

    def build(root, mutate=None, summaries=None, shards=2):
        os.makedirs(root)
        diff = os.path.join(root, "pr.diff")
        open(diff, "w").write("diff --git a/x b/x\n+fn f() {}\n")
        dsha = sha256_file(diff)
        parts = [ids[k::shards] for k in range(shards)]
        spec = {"head": head, "diff": dsha, "parts": parts, "list": list(ids), "skip": None, "summ": summaries or {}}
        if mutate:
            mutate(spec)
        for k in range(1, shards + 1):
            if spec["skip"] == k:
                continue
            sd = os.path.join(root, "art", f"mutants-shard-{k}")
            os.makedirs(sd)
            json.dump({"shard": k, "shards": shards, "head_sha": spec["head"] if k == 1 else head,
                       "diff_sha256": spec["diff"] if k == 2 else dsha, "run_id": 1, "run_attempt": 1},
                      open(os.path.join(sd, "meta.json"), "w"))
            # a planted universe goes to shard 2 only (empty: to every shard)
            lst = spec["list"] if (k == 2 or not spec["list"]) else ids
            open(os.path.join(sd, "list.txt"), "w").write("\n".join(lst) + "\n")
            outs = [{"scenario": "Baseline", "summary": "Success"}]
            outs += [{"scenario": {"Mutant": {"name": m}}, "summary": spec["summ"].get(m, "CaughtMutant")}
                     for m in spec["parts"][k - 1]]
            json.dump({"outcomes": outs}, open(os.path.join(sd, "outcomes.json"), "w"))
        return diff

    RP = "evidence/pr-review/1/" + "c" * 40 + "/receipt.intoto.jsonl"

    here = os.path.dirname(os.path.abspath(__file__))
    keys = os.path.join(here, "..", "..", "tests", "fixtures", "pr-review", "keys")
    key, pub = os.path.join(keys, "pr-review-test-TEST-ONLY.key"), os.path.join(keys, "pr-review-test.pub")
    if not shutil.which("minisign") or not os.path.isfile(key):
        print(f"ENV   minisign or the test key {key} is missing: the signature rows cannot be measured")
        return 2
    kt = tempfile.mkdtemp()
    import atexit
    atexit.register(shutil.rmtree, kt, True)
    okey, opub = os.path.join(kt, "o.key"), os.path.join(kt, "o.pub")
    if subprocess.run(["minisign", "-G", "-W", "-f", "-p", opub, "-s", okey], capture_output=True).returncode:
        print("ENV   minisign -G failed: cannot mint the other-key fixture")
        return 2

    def receipt(root, text):
        # a CI-shaped receipt naming `text`, signed with the test key. Prefixes plant one defect each:
        # STRAY: a non-receipt file . RAW: right path, not a statement . SELF: reviewer = author
        # UNSIGNED: no .minisig . OTHERKEY: signed with a key that is not --pubkey
        mode, _, body = text.partition(":") if text.split(":")[0] in ("STRAY", "RAW", "SELF", "UNSIGNED", "OTHERKEY") else ("", "", text)
        if mode == "STRAY":
            open(os.path.join(root, "receipt.txt"), "w").write(body)
            return
        rp = os.path.join(root, RP)
        os.makedirs(os.path.dirname(rp), exist_ok=True)
        if mode == "RAW":
            open(rp, "w").write(body + "\n")
            return
        rv = "agent:claude-opus-5-5/a" if mode == "SELF" else "agent:claude-sonnet-5/b"
        open(rp, "w").write(json.dumps({"_type": "https://in-toto.io/Statement/v1",
            "predicateType": "https://paiml.dev/attestations/pr-review/v2",
            "predicate": {"author_actor": {"id": "agent:claude-opus-5-5/a"}, "reviewer_actor": {"id": rv},
                          "note": body}}) + "\n")
        if mode != "UNSIGNED":
            subprocess.run(["minisign", "-S", "-s", okey if mode == "OTHERKEY" else key, "-m", rp,
                            "-x", rp + ".minisig"], capture_output=True, input=b"\n", check=True)

    cases = []

    def case(name, want_green, mutate=None, summaries=None, equiv_rows=None, max_missed=0, receipt_text=None, why="",
             nokey=False):
        # why: the RED must be for THIS reason, not another check that happens to fire too
        cases.append((name, want_green, mutate, summaries, equiv_rows, max_missed, receipt_text, why, nokey))

    def drop_one(s):
        s["parts"][0] = s["parts"][0][1:]
    def dup_one(s):
        s["parts"][1] = s["parts"][1] + [s["parts"][0][0]]
    def wrong_sha(s):
        s["head"] = "b" * 40
    def wrong_diff(s):
        s["diff"] = "0" * 64
    def no_shard(s):
        s["skip"] = 2
    def other_universe(s):
        s["list"] = s["list"][:-1]
    def extra(s):
        s["parts"][0] = s["parts"][0] + ["crates/x/src/a.rs:99:1: not listed"]
    def empty(s):
        s["list"], s["parts"] = [], [[], []]

    miss = {ids[3]: "MissedMutant"}
    case("complete table, all killed -> GREEN", True)
    case("a table missing 1 mutant -> RED", False, drop_one, why="MISSING")
    case("a mutant tested twice -> RED", False, dup_one, why="EXTRA")
    case("a shard run on the wrong head sha -> RED", False, wrong_sha, why="not the PR head")
    case("a shard bound to another diff -> RED", False, wrong_diff, why="diff sha256")
    case("a shard that delivered nothing -> RED", False, no_shard, why="delivered nothing")
    case("shards that listed different universes -> RED", False, other_universe, why="different universe")
    case("a tested mutant that was never listed -> RED", False, extra, why="EXTRA")
    case("an empty universe -> RED", False, empty, why="universe is empty")
    case("one survivor over limit 0 -> RED", False, summaries=miss, why="SURVIVED")
    case("one timeout over limit 0 -> RED", False, summaries={ids[2]: "Timeout"}, why="SURVIVED")
    case("an unviable mutant counts as killed -> GREEN", True, summaries={ids[1]: "Unviable"})
    case("an unknown outcome -> RED", False, summaries={ids[1]: "Failure"}, why="UNJUDGED")
    case("survivor within --max-missed 1 -> GREEN", True, summaries=miss, max_missed=1)
    case("equivalent with proof + receipt naming it -> GREEN", True, summaries=miss,
         equiv_rows=[f"{ids[3]}\tf4 is only called with x>0, so the mutant is dead\t" + RP], receipt_text=ids[3])
    case("equivalent whose receipt does not name it -> RED", False, summaries=miss,
         equiv_rows=[f"{ids[3]}\tproof\t" + RP], receipt_text="some other mutant", why="does not name")
    case("equivalent citing a stray file that names it -> RED", False, summaries=miss,
         equiv_rows=[f"{ids[3]}\tproof\treceipt.txt"], receipt_text="STRAY:" + ids[3], why="not a pr-review receipt path")
    case("equivalent citing a receipt path that is not an in-toto statement -> RED", False, summaries=miss,
         equiv_rows=[f"{ids[3]}\tproof\t" + RP], receipt_text="RAW:" + ids[3], why="not an in-toto")
    case("equivalent whose receipt is a self-review -> RED", False, summaries=miss,
         equiv_rows=[f"{ids[3]}\tproof\t" + RP], receipt_text="SELF:" + ids[3], why="self-review")
    case("equivalent whose receipt is unsigned -> RED", False, summaries=miss,
         equiv_rows=[f"{ids[3]}\tproof\t" + RP], receipt_text="UNSIGNED:" + ids[3], why="unsigned")
    case("equivalent signed by a key that is not --pubkey -> RED", False, summaries=miss,
         equiv_rows=[f"{ids[3]}\tproof\t" + RP], receipt_text="OTHERKEY:" + ids[3], why="does not verify")
    case("equivalent row checked with no --pubkey -> RED", False, summaries=miss,
         equiv_rows=[f"{ids[3]}\tproof\t" + RP], receipt_text=ids[3], why="no --pubkey", nokey=True)
    case("equivalent with no proof column -> RED", False, summaries=miss,
         equiv_rows=[f"{ids[3]}\t\t" + RP], receipt_text=ids[3], why="3 non-empty")
    case("equivalent row for an id outside the universe -> RED", False,
         equiv_rows=["crates/x/src/a.rs:99:1: gone\tproof\t" + RP], receipt_text="crates/x/src/a.rs:99:1: gone", why="stale proof")

    bad = 0
    for name, want_green, mutate, summ, eq_rows, maxm, rtext, why, nokey in cases:
        with tempfile.TemporaryDirectory() as t:
            root = os.path.join(t, "r")
            diff = build(root, mutate, summ)
            equiv = None
            if eq_rows is not None:
                receipt(root, rtext or "")
                equiv = os.path.join(root, "equiv.tsv")
                open(equiv, "w").write("\n".join(eq_rows) + "\n")
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                errs, _ = check(os.path.join(root, "art"), 2, head, diff, equiv, maxm, root,
                                pubkey=None if nokey else pub)
            green = not errs
            ok = green == want_green and (green or why in "\n".join(errs))
            bad += not ok
            print(f"{'ok  ' if ok else 'FAIL'}  {name}" + ("" if ok else f" -- got {'GREEN' if green else 'RED'}: {errs}"))
    print("PASS" if not bad else f"FAIL: {bad} row(s)")
    return 1 if bad else 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        return self_test()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--dir", required=True)
    c.add_argument("--shards", type=int, required=True)
    c.add_argument("--head-sha", required=True)
    c.add_argument("--diff", required=True)
    c.add_argument("--equiv")
    c.add_argument("--pubkey", help="minisign public key the equivalents receipts must verify against (the BASE branch copy)")
    c.add_argument("--max-missed", type=int, default=0)
    c.add_argument("--repo", default=".")
    c.add_argument("--table")
    a = ap.parse_args()
    if a.shards < 1 or len(a.head_sha) != 40:
        print("usage: --shards >= 1 and a 40-hex --head-sha", file=sys.stderr)
        return 2
    errs, _ = check(a.dir, a.shards, a.head_sha, a.diff, a.equiv, a.max_missed, a.repo, a.table, a.pubkey)
    for e in errs:
        print(f"RED   {e}")
    if errs:
        return 1
    print("GREEN every listed mutant judged once, on the PR head and diff, survivors within the limit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
