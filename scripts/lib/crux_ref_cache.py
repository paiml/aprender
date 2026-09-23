"""crux_ref_cache.py: the CRUX reference cache (#4036, part of #4033 lever c).

A reference answer (hf, vLLM, llamafile) to a prompt is a property of the ORACLE and the QUESTION, not of the
host that asked it: the same model file, prompt, sampling and engine build give the same truth on lambda and on gx10.
So a host that already measured the references can hand them to the next host, and that host runs only the apr legs.
Measured on the 0.69.1 lambda sweep (aprender-36, #4033): the reference engines are 84% of CRUX wall time.

llama.cpp is NOT cached, and asking to cache it is refused: its llama-server is also apr's REFERENCE RENDERER
on the serve verb (the raw-prompt routes, POST /generate and the like, get their prompt rendered by it). A
cache hit that switched llama.cpp off refused 14 of apr's own serve cells on gx10 (2026-09-23), RED rather
than a false GREEN, and no saving. Only pure comparators are cached: engines no apr cell depends on.

KEY. One entry per (model sha256, thinking mode, backend, engine, row verb, prompt id), bound to everything that could
change the answer. The key holds:
  - the prompt object's sha256 (messages, max_tokens, verbs);
  - the mode's max_tokens for that prompt, and the protocol's temperature, seed and context;
  - the oracle's version: a plugin's probe reduced to its package versions;
  - the HF source (repo, revision, dtype) for the source-weight engines;
  - a digest of every harness file that PRODUCES a reference row: the dogfood, its cell libs and the engine's driver.
The host is NOT in the key; that is the point. The backend IS: a CPU-lane reference does not vouch for a GPU lane.

THREAT MODEL. The integrity checks catch a cache that is CORRUPTED, EDITED IN PART or LEFT FROM ANOTHER HARNESS:
a byte of an artifact, any field of an entry (added, removed or changed), a pointer, an unreferenced file. They do
NOT stop a deliberate forger who can write the cache dir and re-seal every hash — nothing stored beside the entry
can, and that writer can equally edit the dogfood, the judge or the receipt. Forgery resistance would need a key
held outside the cache; it is out of scope for #4036 (quorum round 5, lane 2), and stated here rather than claimed.

THREE OUTCOMES, never a fourth:
  hit    every expected reference entry is present and intact: the rows are injected, and the engines do not run.
  miss   any entry is absent: the mode runs every engine as before, and its clean rows are stored afterwards.
         Absence is the only thing a changed oracle or harness can produce, since it changes the digest.
  stale  an entry is present at its digest but no longer matches it: an edited key, a tampered row, an artifact
         whose sha256 moved, or a refused row that should never have been stored. It is RED and NEVER reused. Every
         reference row of that mode is injected REFUSED and names the entry, so the judge has no oracle and the
         cell goes RED. Recovery is deleting the entry by hand, after reading why it went stale.

Only rows that answered (rc 0, not refused) are stored. A refusal is not truth, so it is recomputed every run.

  crux_ref_cache.py lookup  --cache D --work W --manifest M <key args> --out-rows F   exit 0 hit · 10 miss · 11 stale
Every other exit (1 a keying refusal, 2 a crash) is NONE of the three, and the dogfood declines the run on it: a
crash that shared the miss code would turn a corrupted cache into a silent recompute (quorum round 3, lane 2).
  crux_ref_cache.py store   --cache D --work W --manifest M <key args>                exit 0 · prints "stored N"
Key args: --model-sha --thinking --backend --host --engines e1,e2 --verbs v1,v2 --oracle eng=version (repeat) --source JSON
          --temperature --seed --context --max-tokens <the mode's global cap> --root <repo root>
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import tempfile

SCHEMA = "crux-ref-cache/v1"
HIT, MISS, STALE, CRASH = 0, 10, 11, 2
CACHED_ENGINES = ("hf", "vllm", "llamafile")
NOT_CACHEABLE = {"llama.cpp": "its llama-server is apr's reference renderer on the serve routes, so it must run "
                              "wherever apr runs (#4036, measured on gx10)"}
SOURCE_ENGINES = ("hf", "vllm")

# The harness files that PRODUCE a reference row. The judge's files (crux_inference_judge.py, crux_oracles.py, the
# certifier, the smoke scope) are not here: they read rows and never change one.
COMMON_FILES = (
    "scripts/lib/crux_ref_cache.py",  # its injection and provenance shape the rows the judge reads
    "scripts/crux_inference_dogfood.sh",
    "scripts/lib/crux_cells_serve_code.sh",
    "scripts/lib/crux_cell_teardown.sh",
    "scripts/lib/crux_openai_client.py",
    "scripts/lib/crux_proc.py",
    "scripts/lib/crux_pty_chat.py",
    "scripts/lib/crux_serve_routes.py",
    "scripts/lib/crux_sse.py",
    "scripts/llama_pin.toml",
    "scripts/llama_bin.sh",
)
ENGINE_FILES = {
    "hf": ("scripts/crux_engine_hf.sh", "scripts/lib/crux_hf_verify.py", "scripts/crux_hf/engine.py",
           "scripts/crux_hf/pyproject.toml", "scripts/crux_hf/uv.lock"),
    "vllm": ("scripts/crux_engine_vllm.sh", "scripts/lib/crux_hf_verify.py", "scripts/crux_vllm/engine.py",
             "scripts/crux_vllm/pyproject.toml", "scripts/crux_vllm/uv.lock"),
    "llamafile": ("scripts/crux_engine_llamafile.sh",),
}
# A probe line carries the device, the capability and the lock path of THIS host. Only the package versions say
# which oracle answered, so only they enter the key.
PROBE_KEYS = ("vllm", "transformers", "torch", "tokenizers", "jinja2", "llamafile")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def oracle_id(engine, version):
    if engine in ("hf", "vllm"):
        toks = [t for t in version.split() if "=" in t and t.split("=", 1)[0] in PROBE_KEYS]
        return " ".join(sorted(toks)) or version
    return version


def harness_digest(root, engine):
    h = hashlib.sha256()
    for rel in COMMON_FILES + ENGINE_FILES.get(engine, ()):
        p = os.path.join(root, rel)
        h.update(rel.encode() + b"\0")
        h.update((sha256_file(p) if os.path.isfile(p) else "absent").encode() + b"\n")
    return h.hexdigest()


def expected(work, verbs):
    """The (row verb, prompt id) pairs this run asks every engine for, from the dogfood's own prompt files, limited
    to the verbs the run was given (`serve` makes the `serve run` and `serve stream` rows)."""
    return [(v, pid) for v, pid in _all_pairs(work) if (v.split()[0] if v.startswith("serve") else v) in verbs]


def _all_pairs(work):
    pairs = []
    with open(os.path.join(work, "pids-all.txt")) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    sp = os.path.join(work, "serve-prompts.jsonl")
    if os.path.exists(sp):
        for line in open(sp):
            if line.strip():
                pid, verbs = json.loads(line)
                pairs.extend((v, pid) for v in verbs)
    cp = os.path.join(work, "code-prompts.txt")
    if os.path.exists(cp):
        pairs.extend(("code", pid.strip()) for pid in open(cp) if pid.strip())
    return pairs


def entry_key(a, engine, verb, pid, oracles, harness):
    # The cap the engine is GIVEN (quorum round 1, PMAT-4036): run and chat cells get the mode's global cap (the
    # largest budget in the whole prompt set, so another prompt's edit moves it), serve and code the prompt's own.
    # A cap that cannot be read is a refusal, never a key with max_tokens None.
    if verb in ("run", "chat"):
        maxtok = str(a.max_tokens)
    else:
        maxtok_f = os.path.join(a.work, "maxtok-%s-%s.txt" % (pid, a.thinking))
        if not os.path.isfile(maxtok_f):
            sys.exit("crux_ref_cache: %s has no per-mode cap file %s: an entry without its max_tokens cannot be keyed"
                     % (pid, maxtok_f))
        maxtok = open(maxtok_f).read().strip()
    return {
        "schema": SCHEMA,
        "model_sha256": a.model_sha,
        "thinking": a.thinking,
        "backend": a.backend,
        "engine": engine,
        "verb": verb,
        "prompt_id": pid,
        "prompt_sha256": sha256_file(os.path.join(a.work, "prompt-%s.json" % pid)),
        "sampling": {"temperature": a.temperature, "seed": a.seed, "context": a.context,
                     "max_tokens": maxtok},
        "oracle": oracle_id(engine, oracles[engine]),
        "source": json.loads(a.source or "null") if engine in SOURCE_ENGINES else None,
        "harness_sha256": harness[engine],
    }


def digest(key):
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def content_sha256(ent):
    """The WHOLE entry — every field but this seal itself: key, rows, files, origin — as one digest, recorded at store
    and checked at lookup. A rule per field only sees the fields it names: deleting a row's `stdout` passed every one
    of them (quorum round 4, lane 1), and the origin was left outside a {key, rows, files} seal (quorum round 6,
    lane 2, measured). Any field added, removed or changed anywhere in the entry moves this digest."""
    return hashlib.sha256(json.dumps({k: v for k, v in ent.items() if k != "content_sha256"},
                                     sort_keys=True).encode()).hexdigest()


def entry_dir(cache, d):
    return os.path.join(cache, d[:2], d)


def plan(a):
    oracles = dict(o.split("=", 1) for o in a.oracle)
    engines = [e for e in a.engines.split(",") if e]
    for e in engines:
        if e in NOT_CACHEABLE:
            sys.exit("crux_ref_cache: %s cannot be cached: %s" % (e, NOT_CACHEABLE[e]))
        if e not in CACHED_ENGINES:
            sys.exit("crux_ref_cache: %s is not a reference engine (cached: %s)" % (e, ", ".join(CACHED_ENGINES)))
        if e not in oracles:
            sys.exit("crux_ref_cache: no --oracle version for %s: an unversioned oracle cannot be keyed" % e)
    harness = {e: harness_digest(a.root, e) for e in engines}
    out = []
    for e in engines:
        for verb, pid in expected(a.work, a.verbs.split(",")):
            k = entry_key(a, e, verb, pid, oracles, harness)
            out.append((k, digest(k)))
    return out


def row_paths(row):
    """(container, field) for every string field of the row that names a file: top level and one level down."""
    for k, v in row.items():
        if isinstance(v, str) and v.startswith("/"):
            yield row, k
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, str) and v2.startswith("/"):
                    yield v, k2


def why_stale(key, d, edir):
    """None when the entry at edir is intact for key; otherwise why it may not be reused."""
    try:
        ent = json.load(open(os.path.join(edir, "entry.json")))
    except (OSError, ValueError) as e:
        return "entry.json unreadable (%s)" % e.__class__.__name__
    if not ent.get("content_sha256") or content_sha256(ent) != ent["content_sha256"]:
        return "its content changed since it was stored (a key, row or files field was edited, added or removed)"
    if ent.get("key") != key:
        diff = sorted(f for f in set(key) | set(ent.get("key") or {}) if key.get(f) != (ent.get("key") or {}).get(f))
        return "its key no longer matches its digest (fields: %s)" % ", ".join(diff)
    if digest(ent["key"]) != d:
        return "its key hashes to a different digest"
    rows = ent.get("rows") or []
    if not rows:
        return "it holds no row"
    for r in rows:
        ident = (r.get("model_sha256"), r.get("thinking"), r.get("engine"), r.get("verb"), r.get("prompt_id"))
        if ident != (key["model_sha256"], key["thinking"], key["engine"], key["verb"], key["prompt_id"]):
            return "a row's identity %r contradicts the key" % (ident,)
        if r.get("refused") or r.get("rc") != 0:
            return "it holds a refused or failed row, which is never truth"
        for box, f in row_paths_rel(r):
            rel = box[f][len("refcache:"):]
            # A pointer is a plain name the entry's own files{} holds — never a path: `../` would read or write outside
            # the cache and the work dir, and a name files{} does not hold is a row no sha256 vouches for.
            if not rel or rel != os.path.basename(rel) or rel in (".", "..") or rel not in (ent.get("files") or {}):
                return "a row's artifact pointer %r is not one of the entry's own hashed files" % box[f]
        for box, f in row_paths(r):
            return "a row still names an absolute path %r: its artifact was never stored" % box[f]
    used = {box[f][len("refcache:"):] for r in rows for box, f in row_paths_rel(r)}
    extra = sorted(set(ent.get("files") or {}) - used)
    if extra:
        return "files{} holds %s, which no row points at: an entry carries only the files its rows need" % extra[0]
    for rel, want in (ent.get("files") or {}).items():
        if not rel or rel != os.path.basename(rel) or rel in (".", ".."):
            return "files{} names %r, which is not a plain file name" % rel
        p = os.path.join(edir, "files", rel)
        if not os.path.isfile(p):
            return "artifact %s is missing" % rel
        if sha256_file(p) != want:
            return "artifact %s changed since it was stored (sha256)" % rel
    return None


def cmd_lookup(a):
    entries = plan(a)
    hits, misses, stale = [], [], []
    for k, d in entries:
        edir = entry_dir(a.cache, d)
        if not os.path.exists(edir):
            misses.append((k, d))
            continue
        w = why_stale(k, d, edir)
        (stale if w else hits).append((k, d, w))
    if not entries:
        print("reference cache: nothing to look up")
        return MISS
    if not stale and misses:
        print("reference cache MISS: %d of %d entries absent (e.g. %s %s %s): every engine runs"
              % (len(misses), len(entries), misses[0][0]["engine"], misses[0][0]["verb"], misses[0][0]["prompt_id"]))
        return MISS
    out = open(a.out_rows, "w")
    for k, d, _ in hits:
        if stale:
            continue
        edir = entry_dir(a.cache, d)
        ent = json.load(open(os.path.join(edir, "entry.json")))
        dest = os.path.join(a.work, "refcache", d)
        os.makedirs(dest, exist_ok=True)
        for r in ent["rows"]:
            for box, f in list(row_paths_rel(r)):
                rel = box[f][len("refcache:"):]
                shutil.copyfile(os.path.join(edir, "files", rel), os.path.join(dest, rel))
                box[f] = os.path.join(dest, rel)
            r["host"] = a.host
            r["reference_cache"] = {"digest": d, "origin": ent.get("origin")}
            out.write(json.dumps(r) + "\n")
    if stale:
        k0, d0, w0 = stale[0]
        why = ("reference cache entry %s (%s %s %s) is STALE: %s; a stale entry is RED and never reused (#4036), "
               "delete it by hand after reading why" % (d0[:16], k0["engine"], k0["verb"], k0["prompt_id"], w0))
        for k, d in entries:
            out.write(json.dumps({"kind": "gen", "engine": k["engine"], "prompt_id": k["prompt_id"], "rc": None,
                                  "stdout": None, "stderr": None, "refused": why, "model_sha256": k["model_sha256"],
                                  "host": a.host, "verb": k["verb"], "thinking": k["thinking"], "backend": a.backend,
                                  "reference_cache": {"digest": d, "stale": d in {s[1] for s in stale}}}) + "\n")
        out.close()
        print("reference cache STALE: %d of %d entries (%s); every reference row of this mode is refused"
              % (len(stale), len(entries), w0))
        return STALE
    out.close()
    print("reference cache HIT: %d entries, %d engines skipped this mode" % (len(hits), len({k["engine"] for k, _, _ in hits})))
    return HIT


def row_paths_rel(row):
    for k, v in row.items():
        if isinstance(v, str) and v.startswith("refcache:"):
            yield row, k
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, str) and v2.startswith("refcache:"):
                    yield v, k2


def cmd_store(a):
    rows = [json.loads(line) for line in open(a.manifest) if line.strip()]
    stored = skipped = 0
    for k, d in plan(a):
        edir = entry_dir(a.cache, d)
        if os.path.exists(edir):
            continue
        mine = [r for r in rows if r.get("kind") == "gen" and r.get("model_sha256") == k["model_sha256"]
                and r.get("thinking") == k["thinking"] and r.get("engine") == k["engine"]
                and r.get("verb") == k["verb"] and r.get("prompt_id") == k["prompt_id"]
                and "reference_cache" not in r]
        if not mine or any(r.get("refused") or r.get("rc") != 0 for r in mine):
            skipped += 1
            continue
        os.makedirs(os.path.dirname(edir), exist_ok=True)
        tmp = tempfile.mkdtemp(prefix=".tmp-", dir=os.path.dirname(edir))
        os.makedirs(os.path.join(tmp, "files"))
        files, ok = {}, True
        for i, r in enumerate(mine):
            for box, f in list(row_paths(r)):
                p = box[f]
                if not p.startswith(a.work.rstrip("/") + "/"):
                    continue
                if not os.path.isfile(p):
                    ok = False
                    break
                rel = "%d-%s-%s" % (i, f, os.path.basename(p))
                shutil.copyfile(p, os.path.join(tmp, "files", rel))
                files[rel] = sha256_file(os.path.join(tmp, "files", rel))
                box[f] = "refcache:" + rel
        if not ok:
            shutil.rmtree(tmp)
            skipped += 1
            continue
        ent = {"key": k, "rows": mine, "files": files,
               "origin": {"host": a.host, "backend": a.backend, "harness_git_sha": a.harness_git_sha or None,
                          "stored_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}}
        ent["content_sha256"] = content_sha256(ent)
        json.dump(ent, open(os.path.join(tmp, "entry.json"), "w"), indent=1, sort_keys=True)
        try:
            os.rename(tmp, edir)
            stored += 1
        except OSError:
            shutil.rmtree(tmp)  # another writer stored the same digest first
    print("reference cache: stored %d entries, %d not stored (refused, failed or absent rows are never truth)"
          % (stored, skipped))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=("lookup", "store"))
    p.add_argument("--cache", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--model-sha", required=True)
    p.add_argument("--thinking", required=True, choices=("off", "on"))
    p.add_argument("--backend", required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--engines", required=True)
    p.add_argument("--verbs", required=True)
    p.add_argument("--oracle", action="append", default=[])
    p.add_argument("--source", default="")
    p.add_argument("--temperature", required=True)
    p.add_argument("--seed", required=True)
    p.add_argument("--context", required=True)
    p.add_argument("--max-tokens", type=int, required=True, help="the mode's global cap, which run/chat cells are given")
    p.add_argument("--root", required=True)
    p.add_argument("--harness-git-sha", default="")
    p.add_argument("--out-rows")
    a = p.parse_args()
    if a.cmd == "lookup":
        if not a.out_rows:
            p.error("lookup needs --out-rows")
        return cmd_lookup(a)
    return cmd_store(a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 — any crash is its own outcome, never a miss
        sys.stderr.write("crux_ref_cache: CRASH %s: %s\n" % (e.__class__.__name__, e))
        sys.exit(CRASH)
