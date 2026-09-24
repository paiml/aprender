/-
PVL-001 EV-7b (#4201) — the comparator: does each solution close EXACTLY its pinned Challenge statement?

  lake env lean --run scripts/Comparator.lean Challenge/<stem>.lean …

Each argument is one EV-7a Challenge file. It is elaborated here, against the BUILT tree (`build.sh` first;
`lake env` builds nothing), and every declaration it adds under `PvlChallenge` is one row. A Challenge file
imports the solution modules it pins, so its solution is in the same environment. One NDJSON line per row:

  {"name": F, "challenge_type_hash": H|null, "solution_type_hash": H|null, "axioms": [..]|null}

- `name` is the SOLUTION's full name F; the challenge is `PvlChallenge.F` (`solutionOf?`, the one naming rule;
  `pv`'s `discharge::comparator::challenge_decl` is its inverse and the two are tested against each other).
- A hash is the sha256 of the type's canonical form (`canon`: binder names dropped, binder info kept),
  after the statement was elaborated in its own file. Equal hashes mean the solution proves the pinned statement; a weakened one
  (a strengthened hypothesis) proves a different type. `pv discharge check --comparator` judges, not this file.
- `solution_type_hash: null` — no such constant: the challenge pins a theorem that does not exist.
- `axioms` are the SOLUTION's (`collectAxioms`), sorted; `sorryAx` there means it closes nothing.

Exit: 0 every file elaborated (rows judged by pv) · 1 a file failed to parse or elaborate (its errors on
stderr, its rows withheld) · 2 no argument. `--self-test` checks sha256 against the FIPS vectors. Stdout carries rows only; diagnostics go to stderr.
-/
import Lean
open Lean Elab

/-- The one naming rule, Lean side: `PvlChallenge.F` ↦ `F`. -/
def solutionOf? (challenge : Name) : Option Name :=
  let s := challenge.replacePrefix `PvlChallenge .anonymous
  if s == challenge || s.isAnonymous then none else some s

/-- The type's canonical form: its structure, constants, universe levels, literals and binder INFO, and not its
binder NAMES (`(x : Nat)` and `(y : Nat)` state the same thing; `{x : Nat}` and `(x : Nat)` do not). -/
partial def canon : Expr → String
  | .bvar i => s!"#{i}"
  | .fvar id => s!"F({id.name})"
  | .mvar id => s!"M({id.name})"
  | .sort l => s!"S({l})"
  | .const n ls => s!"C({n},{ls})"
  | .app f a => s!"A({canon f},{canon a})"
  | .lam _ t b bi => s!"L{repr bi}({canon t},{canon b})"
  | .forallE _ t b bi => s!"P{repr bi}({canon t},{canon b})"
  | .letE _ t v b nd => s!"E{nd}({canon t},{canon v},{canon b})"
  | .lit (.natVal n) => s!"N{n}"
  | .lit (.strVal v) => s!"T{v.quote}"
  | .mdata _ e => canon e
  | .proj s i e => s!"J({s},{i},{canon e})"

/-! SHA-256 (FIPS 180-4), in Lean: core has none, and a proof-to-statement binding wants a cryptographic hash, not
the 32-bit `Expr.hash` (cop ruling on #4201). Checked against the FIPS vectors by `--self-test`. -/
def sha256K : Array UInt32 := #[
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]

def rotr (x : UInt32) (n : UInt32) : UInt32 := (x >>> n) ||| (x <<< (32 - n))

def sha256 (msg : ByteArray) : String := Id.run do
  let len := msg.size
  let mut m := msg.push 0x80
  while m.size % 64 != 56 do m := m.push 0
  let bits := len.toUInt64 * 8
  for i in [0:8] do m := m.push ((bits >>> (8 * (7 - i).toUInt64)).toUInt8)
  let mut h : Array UInt32 := #[0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
  for blk in [0:m.size / 64] do
    let mut w : Array UInt32 := #[]
    for t in [0:16] do
      let b (k : Nat) : UInt32 := (m.get! (blk * 64 + 4 * t + k)).toUInt32
      w := w.push ((b 0 <<< 24) ||| (b 1 <<< 16) ||| (b 2 <<< 8) ||| b 3)
    for t in [16:64] do
      let s0 := rotr w[t-15]! 7 ^^^ rotr w[t-15]! 18 ^^^ (w[t-15]! >>> 3)
      let s1 := rotr w[t-2]! 17 ^^^ rotr w[t-2]! 19 ^^^ (w[t-2]! >>> 10)
      w := w.push (w[t-16]! + s0 + w[t-7]! + s1)
    let mut a := h[0]!; let mut b := h[1]!; let mut c := h[2]!; let mut d := h[3]!
    let mut e := h[4]!; let mut f := h[5]!; let mut g := h[6]!; let mut hh := h[7]!
    for t in [0:64] do
      let t1 := hh + (rotr e 6 ^^^ rotr e 11 ^^^ rotr e 25) + ((e &&& f) ^^^ (~~~e &&& g)) + sha256K[t]! + w[t]!
      let t2 := (rotr a 2 ^^^ rotr a 13 ^^^ rotr a 22) + ((a &&& b) ^^^ (a &&& c) ^^^ (b &&& c))
      hh := g; g := f; f := e; e := d + t1; d := c; c := b; b := a; a := t1 + t2
    h := #[h[0]! + a, h[1]! + b, h[2]! + c, h[3]! + d, h[4]! + e, h[5]! + f, h[6]! + g, h[7]! + hh]
  let hex (x : UInt32) : String :=
    let s := String.ofList (Nat.toDigits 16 x.toNat)
    "".pushn '0' (8 - s.length) ++ s
  return String.join (h.toList.map hex)

/-- sha256 of `canon`, as 64 hex digits. -/
def typeHash (e : Expr) : String := sha256 (canon e).toUTF8

def jStr (s : String) : String := (Json.str s).compress

def row (name : Name) (ch : Option String) (sol : Option String) (ax : Option (Array Name)) : String :=
  let opt (o : Option String) := o.map jStr |>.getD "null"
  let axs := ax.map (fun a => "[" ++ ", ".intercalate (a.toList.map (jStr ∘ toString)) ++ "]") |>.getD "null"
  "{\"name\": " ++ jStr name.toString ++ ", \"challenge_type_hash\": " ++ opt ch ++
    ", \"solution_type_hash\": " ++ opt sol ++ ", \"axioms\": " ++ axs ++ "}"

/-- Elaborate one Challenge file. `none` when it did not elaborate cleanly (an error message), with the
messages printed to stderr. `sorry` warnings are expected: every challenge is `:= sorry`. -/
def elabFile (path : System.FilePath) : IO (Option Environment) := do
  let input ← IO.FS.readFile path
  let inputCtx := Parser.mkInputContext input path.toString
  let (header, parserState, messages) ← Parser.parseHeader inputCtx
  let opts := Options.empty.setBool `autoImplicit false
  let (env, messages) ← processHeader header opts messages inputCtx (trustLevel := 1024)
  let s ← IO.processCommands inputCtx parserState (Command.mkState env messages opts)
  let msgs := s.commandState.messages
  let errs := msgs.toList.filter (·.severity == .error)
  for m in errs do
    IO.eprint (← m.toString (includeEndPos := true))
  if errs.isEmpty then return some s.commandState.env else return none

def rowsOf (env : Environment) : Array String := Id.run do
  let mine := env.constants.map₂.toList.filterMap fun (n, ci) =>
    if (solutionOf? n).isSome && !n.isInternal then some (n, ci) else none
  let mine := mine.toArray.qsort (fun a b => a.1.toString < b.1.toString)
  let mut out := #[]
  for (n, ci) in mine do
    let some sol := solutionOf? n | continue
    match env.find? sol with
    | none => out := out.push (row sol (some (typeHash ci.type)) none none)
    | some sci =>
      let (_, st) := ((CollectAxioms.collect sol).run env).run {}
      let axs := st.axioms.qsort (fun a b => a.toString < b.toString)
      out := out.push (row sol (some (typeHash ci.type)) (some (typeHash sci.type)) (some axs))
  return out

/-- FIPS 180-4 vectors: the empty string, "abc", and the two-block 448-bit message. -/
def selfTest : IO UInt32 := do
  let cases := [("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    ("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
    ("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1")]
  let mut bad := 0
  for (i, want) in cases do
    let got := sha256 i.toUTF8
    IO.println s!"{if got == want then "ok  " else "FAIL"}  sha256 {i.quote} = {got}"
    if got != want then bad := bad + 1
  return if bad == 0 then 0 else 1

def main (args : List String) : IO UInt32 := do
  if args == ["--self-test"] then return ← selfTest
  if args.isEmpty then
    IO.eprintln "usage: lake env lean --run scripts/Comparator.lean <Challenge file>..."
    return 2
  initSearchPath (← findSysroot)
  let mut rc : UInt32 := 0
  for a in args do
    match ← elabFile a with
    | none =>
      IO.eprintln s!"comparator: {a} did not elaborate; its rows are withheld"
      rc := 1
    | some env =>
      for r in rowsOf env do IO.println r
  return rc
