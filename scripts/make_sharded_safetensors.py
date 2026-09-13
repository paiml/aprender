#!/usr/bin/env python3
"""make_sharded_safetensors.py — split ONE `model.safetensors` into N shards plus a
HuggingFace `model.safetensors.index.json`, with no network and no dependencies.

WHY THIS EXISTS (#3024). The nightly story's only SafeTensors models are the 0.5B and
1.5B single files — small enough that `apr pull` never shards them — so the sharded
index path (`ShardedIndex`, `run_sharded_safetensors_inference`) had ZERO exercise from
any gate, on any command, and #3022 shipped: `apr chat` silently ran the toy Demo model
for a `model.safetensors.index.json` while printing the real model's path.

Downloading a real 3B+ sharded release to close that gap would put a multi-gigabyte
network fetch on the critical path of a nightly. A shard boundary is a property of the
*index and the file split*, not of the parameter count: two shards of a 0.5B exercise
`weight_map`, the per-shard header rewrite, and the cross-shard tensor lookup exactly as
two shards of a 7B do. So the fixture is DERIVED from a model the story already holds.

The output is byte-deterministic for a given input and `--shards`: tensors are assigned
to shards in the input header's own order, and every shard's JSON header is written with
sorted keys and no whitespace padding beyond the 8-byte alignment safetensors requires.
Running it twice gives identical files, so a fixture built on one host and a fixture
built on a runner are the same bytes.

    python3 scripts/make_sharded_safetensors.py <model.safetensors> <out-dir> [--shards N]
    python3 scripts/make_sharded_safetensors.py --self-test        # round-trip case table

SAFETENSORS ON DISK: <8-byte LE header length><JSON header><tensor bytes>. The header maps
name -> {dtype, shape, data_offsets: [start, end]} with offsets relative to the start of
the byte buffer, plus an optional "__metadata__" string map. Splitting means re-basing
those offsets per shard; nothing about the tensor bytes changes.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile

ALIGN = 8


def read_header(path: str) -> tuple[dict, int]:
    """-> (header dict, byte offset where the tensor buffer starts)."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: shorter than a safetensors header length")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > 100_000_000:
            raise ValueError(f"{path}: implausible header length {n}")
        head = json.loads(f.read(n).decode("utf-8"))
    return head, 8 + n


def tensor_entries(header: dict) -> list[tuple[str, dict]]:
    """Every real tensor, in the header's own order (`__metadata__` is not a tensor)."""
    return [(k, v) for k, v in header.items() if k != "__metadata__"]


def write_shard(src: str, buf_start: int, entries: list[tuple[str, dict]], out: str) -> int:
    """Write one shard holding `entries`, re-basing offsets from 0. -> bytes written."""
    head: dict = {}
    cursor = 0
    for name, meta in entries:
        start, end = meta["data_offsets"]
        size = end - start
        head[name] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    blob = json.dumps(head, sort_keys=True, separators=(",", ":")).encode("utf-8")
    pad = (-len(blob)) % ALIGN
    blob += b" " * pad
    with open(src, "rb") as fin, open(out, "wb") as fout:
        fout.write(struct.pack("<Q", len(blob)))
        fout.write(blob)
        for _, meta in entries:
            start, end = meta["data_offsets"]
            fin.seek(buf_start + start)
            remaining = end - start
            while remaining:
                chunk = fin.read(min(remaining, 8 << 20))
                if not chunk:
                    raise ValueError(f"{src}: truncated at tensor offset {start}")
                fout.write(chunk)
                remaining -= len(chunk)
    return os.path.getsize(out)


def assign_groups(entries: list[tuple[str, dict]], shards: int) -> list[list[tuple[str, dict]]]:
    """Partition `entries` into `shards` groups balanced by BYTES.

    Not by tensor count: an embedding matrix is most of a small model, so a count-balanced
    split leaves one shard nearly empty. Advance when the current shard has passed its byte
    target AND enough tensors are left to give every shard after the next one at least one.
    `>=` and not `>`: the tensors still to place INCLUDE the current one, so `shards - idx - 1`
    shards need exactly that many. Off by one here silently produced a one-shard "sharded"
    model — an index whose weight_map names a single file, which is not the thing under test.
    """
    total = sum(m["data_offsets"][1] - m["data_offsets"][0] for _, m in entries)
    target = total / shards
    groups: list[list[tuple[str, dict]]] = [[] for _ in range(shards)]
    idx, acc = 0, 0.0
    for k, (name, meta) in enumerate(entries):
        advance = (
            idx < shards - 1
            and groups[idx]
            and acc >= target * (idx + 1)
            and (len(entries) - k) >= (shards - idx - 1)
        )
        if advance:
            idx += 1
        groups[idx].append((name, meta))
        acc += meta["data_offsets"][1] - meta["data_offsets"][0]
    return groups


def _check_splittable(src: str, entries: list, shards: int) -> None:
    """Refuse a request that cannot produce a sharded model, before any byte is written."""
    if shards < 2:
        raise ValueError("--shards must be >= 2; a one-shard index is not a sharded model")
    if len(entries) < shards:
        raise ValueError(f"{src}: {len(entries)} tensors cannot fill {shards} shards")


def write_shards(src: str, buf_start: int, groups: list, out_dir: str) -> dict[str, str]:
    """Write one file per group; -> the tensor -> filename map the index carries."""
    shards = len(groups)
    weight_map: dict[str, str] = {}
    for i, group in enumerate(groups, start=1):
        name = f"model-{i:05d}-of-{shards:05d}.safetensors"
        write_shard(src, buf_start, group, os.path.join(out_dir, name))
        for tensor, _ in group:
            weight_map[tensor] = name
    return weight_map


def write_index(out_dir: str, weight_map: dict[str, str], total: int) -> str:
    """Write `model.safetensors.index.json`; -> its path."""
    index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}
    index_path = os.path.join(out_dir, "model.safetensors.index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")
    return index_path


def split(src: str, out_dir: str, shards: int) -> str:
    """Write `shards` shard files plus the index; -> the index path."""
    header, buf_start = read_header(src)
    entries = tensor_entries(header)
    _check_splittable(src, entries, shards)
    os.makedirs(out_dir, exist_ok=True)
    groups = assign_groups(entries, shards)
    if any(not g for g in groups):
        raise ValueError(
            f"{src}: byte-balanced split left an empty shard "
            f"({[len(g) for g in groups]}) — an index naming fewer than {shards} files "
            "is not a sharded model"
        )
    weight_map = write_shards(src, buf_start, groups, out_dir)
    total = sum(m["data_offsets"][1] - m["data_offsets"][0] for _, m in entries)
    return write_index(out_dir, weight_map, total)


def _write_toy(src: str) -> dict[str, bytes]:
    """A minimal but real safetensors file; -> the payload of each tensor."""
    names = [f"blk.{i}.weight" for i in range(6)]
    payloads = {n: bytes([(i * 7 + j) % 251 for j in range(64 * (i + 1))]) for i, n in enumerate(names)}
    head, cursor = {}, 0
    for n in names:
        b = payloads[n]
        head[n] = {"dtype": "F32", "shape": [len(b) // 4], "data_offsets": [cursor, cursor + len(b)]}
        cursor += len(b)
    head["__metadata__"] = {"format": "pt"}
    blob = json.dumps(head, sort_keys=True, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((-len(blob)) % ALIGN)
    with open(src, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for n in names:
            f.write(payloads[n])
    return payloads


def _tensors_survived(out: str, index: dict, payloads: dict[str, bytes]) -> list[str]:
    """-> the names whose bytes did NOT come back unchanged from their shard."""
    bad = []
    for name, shard in index["weight_map"].items():
        h, bs = read_header(os.path.join(out, shard))
        s0, e0 = h[name]["data_offsets"]
        with open(os.path.join(out, shard), "rb") as f:
            f.seek(bs + s0)
            got = f.read(e0 - s0)
        if got != payloads[name]:
            bad.append(name)
    return bad


def _refuses(src: str, out_dir: str, shards: int) -> bool:
    """-> True when `split` refused the call instead of writing something unusable."""
    try:
        split(src, out_dir, shards)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return True
    return False


def _self_test() -> int:
    """Case table: build a tiny safetensors, split it, and prove every tensor's BYTES
    survive. A splitter that silently dropped or mis-offset a tensor would otherwise
    produce a fixture that makes the gate fail for the wrong reason."""
    state = {"n": 0, "red": 0}

    def row(label: str, ok: bool, detail: str = "") -> None:
        state["n"] += 1
        if ok:
            print(f"ok    row {state['n']}  {label}")
        else:
            print(f"FAIL  row {state['n']}  {label}  {detail}")
            state["red"] = 1

    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "model.safetensors")
        payloads = _write_toy(src)
        out = os.path.join(td, "sharded")
        index = json.load(open(split(src, out, 2), encoding="utf-8"))

        row("the index names every tensor",
            sorted(index["weight_map"]) == sorted(payloads), str(sorted(index["weight_map"])))
        row("the index names exactly 2 shards",
            len(set(index["weight_map"].values())) == 2, str(set(index["weight_map"].values())))
        row("total_size is the sum of the tensor bytes",
            index["metadata"]["total_size"] == sum(len(v) for v in payloads.values()))
        bad = _tensors_survived(out, index, payloads)
        row("every tensor's bytes survive the split", not bad, str(bad))

        out2 = os.path.join(td, "sharded2")
        split(src, out2, 2)
        row("a second run is byte-identical", all(
            open(os.path.join(out, f), "rb").read() == open(os.path.join(out2, f), "rb").read()
            for f in sorted(os.listdir(out))
        ))

        row("--shards 1 is refused", _refuses(src, os.path.join(td, "one"), 1))
        junk = os.path.join(td, "junk.bin")
        with open(junk, "wb") as f:
            f.write(b"not a safetensors file at all")
        row("a non-safetensors input is refused", _refuses(junk, os.path.join(td, "junk-out"), 2))

    print(f"{state['n'] - state['red']}/{state['n']} rows")
    return state["red"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", help="path to a single-file model.safetensors")
    ap.add_argument("out_dir", nargs="?", help="directory to write the shards and the index into")
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--self-test", action="store_true", help="run the round-trip case table")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if not args.src or not args.out_dir:
        ap.error("src and out_dir are required unless --self-test")
    print(split(args.src, args.out_dir, args.shards))
    return 0


if __name__ == "__main__":
    sys.exit(main())
