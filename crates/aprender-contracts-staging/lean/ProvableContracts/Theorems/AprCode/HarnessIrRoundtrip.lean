/-!
# Harness-IR Anthropic Round-Trip (Pillar-5, L4)

Sorry-free proof that the Anthropic wire codec round-trips the canonical
harness IR identically on anthropic-native messages. This is the L4 formal
rung of `contracts/apr-code-harness-ir-v1.yaml` (OBLIG-IR-1), the data-layer
core of "prompt parity works on EITHER Claude Code or Google Antigravity".

The Rust implementation (`crates/aprender-serve/src/harness_ir/`) proves this
over `serde_json::Value` by property test (L2). serde_json is not
BMC/Lean-amenable, so here we prove it over an ALGEBRAIC model of the IR — the
same shape, minus the JSON encoding — by structural induction.

The honest asymmetry is modelled, not hidden: the Anthropic wire `tool_result`
block carries NO tool name (it pairs by `tool_use_id`), so the round-trip is
the identity exactly on blocks whose `toolResult` name is empty
(anthropic-native). This mirrors `is_gemini_native` / the empty-name convention
in the Rust codec and the contract's non_goals.

Uses Lean 4 core only (no Mathlib) — checkable standalone with `lean`.
-/

namespace ProvableContracts.AprCode

/-- Canonical message role (Anthropic `user`/`assistant`). -/
inductive Role where
  | user
  | assistant
  deriving DecidableEq, Repr

/-- Canonical content block. `args`/`resp` stand in for the opaque JSON payload
    (irrelevant to the round-trip, which preserves it verbatim). -/
inductive Block where
  | text (s : String)
  | toolCall (id : Option String) (name : String) (args : String)
  | toolResult (id : Option String) (name : String) (resp : String)
  deriving DecidableEq, Repr

/-- Canonical message: a role and an ordered list of blocks. -/
structure Message where
  role : Role
  blocks : List Block
  deriving DecidableEq, Repr

/-- Abstract Anthropic-wire block. Note `toolResult` has NO name field — the
    honest asymmetry (Anthropic pairs tool results by id, not name). -/
inductive AnthBlock where
  | text (s : String)
  | toolUse (id : Option String) (name : String) (input : String)
  | toolResult (id : Option String) (content : String)
  deriving Repr

/-- Encode a canonical block to the Anthropic wire (drops toolResult name). -/
def encBlock : Block → AnthBlock
  | .text s => .text s
  | .toolCall id name args => .toolUse id name args
  | .toolResult id _name resp => .toolResult id resp

/-- Decode an Anthropic-wire block back to canonical (toolResult name = ""). -/
def decBlock : AnthBlock → Block
  | .text s => .text s
  | .toolUse id name input => .toolCall id name input
  | .toolResult id content => .toolResult id "" content

/-- A block is anthropic-native when its toolResult (if any) has no name —
    exactly the shape the Anthropic wire can represent losslessly. -/
def anthNative : Block → Prop
  | .text _ => True
  | .toolCall _ _ _ => True
  | .toolResult _ name _ => name = ""

/-- OBLIG-IR-1 at block granularity: on anthropic-native blocks, decode∘encode
    is the identity. -/
theorem block_roundtrip (b : Block) (h : anthNative b) :
    decBlock (encBlock b) = b := by
  cases b with
  | text s => rfl
  | toolCall id name args => rfl
  | toolResult id name resp =>
      simp only [anthNative] at h
      subst h
      rfl

/-- Encode a whole message to the wire (role + mapped blocks). -/
def encMsg (m : Message) : Role × List AnthBlock :=
  (m.role, m.blocks.map encBlock)

/-- Decode a whole wire message back to canonical. -/
def decMsg (w : Role × List AnthBlock) : Message :=
  ⟨w.1, w.2.map decBlock⟩

/-- Blockwise round-trip lifts over a list of anthropic-native blocks. -/
theorem list_roundtrip (bs : List Block) (h : ∀ b ∈ bs, anthNative b) :
    (bs.map encBlock).map decBlock = bs := by
  induction bs with
  | nil => rfl
  | cons b bs ih =>
      have hb : anthNative b := h b (List.mem_cons_self b bs)
      have hbs : ∀ b' ∈ bs, anthNative b' := fun b' hb' =>
        h b' (List.mem_cons_of_mem b hb')
      simp only [List.map_cons, block_roundtrip b hb, ih hbs]

/-- OBLIG-IR-1 (message level): the Anthropic round-trip is the identity on
    anthropic-native messages. `decMsg (encMsg m) = m`. -/
theorem message_roundtrip (m : Message) (h : ∀ b ∈ m.blocks, anthNative b) :
    decMsg (encMsg m) = m := by
  cases m with
  | mk role blocks =>
      simp only [encMsg, decMsg, list_roundtrip blocks h]

#check @block_roundtrip
#check @message_roundtrip

end ProvableContracts.AprCode
