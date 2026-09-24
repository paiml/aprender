# PMAT-4131 receipt: sha256 once, cached by file identity (branch feat/4131-sha-identity-cache)

Stacked on fix/4126-timeout-policy (both edit scripts/model_ladder.sh).

## Why (measured)
The #4034 lambda A/B found a ~420-480 s no-lock tail in every arm. That tail is `sha256sum` over the
whole inventory (26 files, 109.1 GB) at 0.23 GB/s (a 5.68 GB file took 24.6 s), on every run,
`--only` included, with rung files hashed a second time in the inventory loop.

## The change
- ladder_sha256 <path> prints "<sha> <hashed|cache|memo>".
  - Key: `stat -Lc '%d %i %s %.9Y %.9Z'` (dev, inode, size, mtime_ns, ctime_ns; follows symlinks).
  - Cache: per host, at $LADDER_SHA_CACHE (default ~/.cache/apr-ladder/sha256-identity.tsv), appended
    under flock. The last matching line wins, and only a 64-hex value is accepted.
  - A file that changed while it was being hashed (the key re-read after hashing differs) is not
    cached.
  - An in-run memo stops rung files being re-hashed in the inventory loop.
- The rung and inventory sites call it. It writes to a $WORK file, not $( ), so the memo survives.
  Inventory rows gain `sha_source`.

## Evidence
- NEW scripts/check_ladder_sha_cache.sh (guard_tree-dispatched), run against real files. Cases:
  miss-then-hit (hashed / cache / memo), rewrite-same-size, mtime-restored (touch -d back to the old
  mtime; ctime moves, so it re-hashes), inode-swap, symlink, cache-consulted (a planted entry for the
  current identity IS returned, so the re-hash cases are not vacuous).
  --self-test: path-keyed cache (killed by rewrite-same-size, mtime-restored and inode-swap) and a
  key without ctime (killed by mtime-restored).
- scripts/check_ladder_provenance.sh was updated rather than weakened. It still evaluates the ONE
  shipped collection line, now with the shipped ladder_sha256 lifted alongside it. Its mutant B (the
  wrong fix: hash the link text) now mutates ladder_sha256 and is still RED on the sha assertion.
  Mutant A (drop -L on bytes) is still RED.
- Measured on lambda's REAL inventory (26 files), with a private cache: cold pass **294.3 s**,
  cached pass **0.2 s**. Some of the cold files were already page-cached; the 0.23 GB/s estimate
  was 473 s.
- check_model_ladder.sh --self-test 302/0. write_errors, only_selection, dogfood_no_defer,
  serve_teardown, serve_probe_timeout, apr_bin_pinned and guards_are_wired: all rc 0. The bashrs
  gate shows the base's 8 findings, none in this diff.

## Deliberately not done
The issue floated skipping inventory rows `--only` does not select. That would change the
receipt's inventory shape (sha256 null), and the cache already removes the repeat cost, so it is
not done here.
