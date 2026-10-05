"""Report include!() targets that `cargo package --list` does not contain.

Separate file, not a heredoc: an inline heredoc collides with the shell
redirection used to feed the include list, and the collision is silent -- python
takes the data as its script and prints nothing, so the caller sees "no
missing files" for every input.

argv: <listing-file> <includes-file>
      listing-file  : one packaged path per line
      includes-file : "<target>\t<including-file>" per line
stdout: the subset of includes-file whose target is absent from listing-file
"""
import sys

with open(sys.argv[1], encoding="utf-8", errors="replace") as fh:
    packaged = {line.rstrip("\n") for line in fh if line.strip()}

with open(sys.argv[2], encoding="utf-8", errors="replace") as fh:
    for line in fh:
        line = line.rstrip("\n")
        if not line:
            continue
        target, _, source = line.partition("\t")
        if target not in packaged:
            print(f"{target}\t{source}")
