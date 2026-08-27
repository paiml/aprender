# D-04: detect filesystem/network/path reach that arrives through a GROUPED std import.
#
# The Makefile's other two detectors match literal `std::fs` / `std::net` / `std::path`
# and the whole words `Path` / `PathBuf`. A grouped import contains none of them:
#
#     use std::{fs, net::TcpStream};        <- text is `std::{fs`, matches nothing
#     use std::{                            <- and the item can sit on its own line
#         collections::BTreeMap,
#         fs,
#     };
#
# Both bind `fs` into scope exactly as `use std::fs;` does, and both are what rustfmt
# emits under `imports_granularity = "Crate"` -- which rustfmt.toml asks for -- so this is
# the DEFAULT spelling in this repo, not a corner case.
#
# Accumulate each `use` statement to its terminating `;` so a multi-line group is tested
# as one string, then require BOTH a `std::` prefix and an fs/net/path path segment. The
# segment test is anchored on non-identifier characters so `collections`, `fmt`,
# `pathological_case` and `Cow` cannot trip it.
#
# Emits `<line>:<statement>` so the Makefile's existing `sed` can prepend the filename and
# the reported line number still points at the real `use`.

/^[[:space:]]*(pub[[:space:]]+)?use[[:space:]]/ {
  start = FNR
  stmt = $0
  while (stmt !~ /;/ && (getline nxt) > 0) {
    stmt = stmt " " nxt
  }
  if (stmt ~ /(^|[^A-Za-z0-9_])std[[:space:]]*::/ &&
      stmt ~ /(^|[^A-Za-z0-9_])(fs|net|path)([^A-Za-z0-9_]|$)/) {
    gsub(/[[:space:]]+/, " ", stmt)
    printf "%d:%s\n", start, stmt
  }
}
