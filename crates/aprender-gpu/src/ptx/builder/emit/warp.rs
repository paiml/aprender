//! Warp-level operation emission
//!
//! Handles: ShflDown, ShflIdx, Vote, VoteBallot, Popc, Bfind, Clz, Bfe, Bfi

use crate::ptx::instructions::PtxOp;

/// Emit warp opcode to the output string
pub(crate) fn emit_warp_opcode(op: &PtxOp, s: &mut String) {
    match op {
        PtxOp::ShflDown => s.push_str("shfl.sync.down.b32"),
        PtxOp::ShflIdx => s.push_str("shfl.sync.idx.b32"),
        PtxOp::Vote | PtxOp::VoteBallot => s.push_str("vote.sync.ballot.b32"),
        PtxOp::Popc => s.push_str("popc"),
        PtxOp::Bfind => s.push_str("bfind"),
        PtxOp::Clz => s.push_str("clz"),
        PtxOp::Bfe => s.push_str("bfe"),
        PtxOp::Bfi => s.push_str("bfi"),
        _ => {}
    }
}

/// Check if this is a warp operation
pub(crate) fn is_warp_op(op: &PtxOp) -> bool {
    matches!(
        op,
        PtxOp::ShflDown
            | PtxOp::ShflIdx
            | PtxOp::Vote
            | PtxOp::VoteBallot
            | PtxOp::Popc
            | PtxOp::Bfind
            | PtxOp::Clz
            | PtxOp::Bfe
            | PtxOp::Bfi
    )
}

/// Check if this op requires skipping the type suffix
pub(crate) fn skip_type_for_warp_op(op: &PtxOp) -> bool {
    matches!(op, PtxOp::ShflDown | PtxOp::ShflIdx | PtxOp::Vote | PtxOp::VoteBallot)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_shfl_down() {
        let mut s = String::new();
        emit_warp_opcode(&PtxOp::ShflDown, &mut s);
        assert_eq!(s, "shfl.sync.down.b32");
    }

    #[test]
    fn test_emit_shfl_idx() {
        let mut s = String::new();
        emit_warp_opcode(&PtxOp::ShflIdx, &mut s);
        assert_eq!(s, "shfl.sync.idx.b32");
    }

    #[test]
    fn test_emit_vote() {
        let mut s = String::new();
        emit_warp_opcode(&PtxOp::Vote, &mut s);
        assert_eq!(s, "vote.sync.ballot.b32");
    }

    #[test]
    fn test_emit_vote_ballot() {
        let mut s = String::new();
        emit_warp_opcode(&PtxOp::VoteBallot, &mut s);
        assert_eq!(s, "vote.sync.ballot.b32");
    }

    #[test]
    fn test_emit_popc() {
        let mut s = String::new();
        emit_warp_opcode(&PtxOp::Popc, &mut s);
        assert_eq!(s, "popc");
    }

    #[test]
    fn test_emit_bfind() {
        let mut s = String::new();
        emit_warp_opcode(&PtxOp::Bfind, &mut s);
        assert_eq!(s, "bfind");
    }

    #[test]
    fn test_emit_clz() {
        let mut s = String::new();
        emit_warp_opcode(&PtxOp::Clz, &mut s);
        assert_eq!(s, "clz");
    }

    #[test]
    fn test_emit_bfe() {
        let mut s = String::new();
        emit_warp_opcode(&PtxOp::Bfe, &mut s);
        assert_eq!(s, "bfe");
    }

    #[test]
    fn test_emit_bfi() {
        let mut s = String::new();
        emit_warp_opcode(&PtxOp::Bfi, &mut s);
        assert_eq!(s, "bfi");
    }

    #[test]
    fn test_emit_non_warp_op() {
        let mut s = String::new();
        emit_warp_opcode(&PtxOp::Add, &mut s);
        assert!(s.is_empty());
    }

    #[test]
    fn test_is_warp_op() {
        assert!(is_warp_op(&PtxOp::ShflDown));
        assert!(is_warp_op(&PtxOp::ShflIdx));
        assert!(is_warp_op(&PtxOp::Vote));
        assert!(is_warp_op(&PtxOp::VoteBallot));
        assert!(is_warp_op(&PtxOp::Popc));
        assert!(is_warp_op(&PtxOp::Bfind));
        assert!(is_warp_op(&PtxOp::Clz));
        assert!(is_warp_op(&PtxOp::Bfe));
        assert!(is_warp_op(&PtxOp::Bfi));
        assert!(!is_warp_op(&PtxOp::Add));
    }

    #[test]
    fn test_skip_type_for_warp_op() {
        assert!(skip_type_for_warp_op(&PtxOp::ShflDown));
        assert!(skip_type_for_warp_op(&PtxOp::ShflIdx));
        assert!(skip_type_for_warp_op(&PtxOp::Vote));
        assert!(skip_type_for_warp_op(&PtxOp::VoteBallot));
        assert!(!skip_type_for_warp_op(&PtxOp::Popc));
        assert!(!skip_type_for_warp_op(&PtxOp::Add));
    }
}
