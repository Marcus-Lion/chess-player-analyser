//! Negamax search with alpha-beta pruning, a port of `_negamax` /
//! `_order_moves` (app/games.py:729-890). Shares a transposition table and
//! killer-move table across the whole search, exactly like the Python engine.

use std::collections::HashMap;

use shakmaty::zobrist::Zobrist64;
use shakmaty::{Chess, Color, EnPassantMode, Move, Position, Role};

use crate::eval::{evaluate_white, Weights};

pub const MATE_SCORE: f64 = 1_000_000.0;

// Transposition-table bound types (games.py:639-641).
const TT_EXACT: u8 = 0;
const TT_LOWERBOUND: u8 = 1;
const TT_UPPERBOUND: u8 = 2;

#[derive(Clone, Copy)]
struct TtEntry {
    depth: i32,
    score: f64,
    flag: u8,
    best_move: Option<Move>,
}

/// Mutable state shared across every node of one search.
pub struct SearchState {
    pub weights: Weights,
    pub evals: u64,
    killers: HashMap<i32, Move>,
    tt: HashMap<u64, TtEntry>,
}

impl SearchState {
    pub fn new(weights: Weights) -> Self {
        SearchState {
            weights,
            evals: 0,
            killers: HashMap::new(),
            tt: HashMap::new(),
        }
    }
}

fn piece_points(role: Role) -> i32 {
    match role {
        Role::Pawn => 1,
        Role::Knight => 3,
        Role::Bishop => 3,
        Role::Rook => 5,
        Role::Queen => 9,
        Role::King => 0,
    }
}

fn zobrist(pos: &Chess) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

/// Does `m` give check? shakmaty has no `gives_check`, so play it on a clone
/// and test -- the same cost profile as python-chess's `board.gives_check`.
fn gives_check(pos: &Chess, m: Move) -> bool {
    let mut child = pos.clone();
    child.play_unchecked(m);
    child.is_check()
}

/// Ordering key for a move, higher searches first (`_order_moves`,
/// games.py:744-760): TT move, then captures by MVV-LVA, then checks, then the
/// killer move, then everything else.
fn move_key(
    pos: &Chess,
    m: Move,
    killer: Option<Move>,
    tt_move: Option<Move>,
) -> (i32, i32, i32) {
    if Some(m) == tt_move {
        return (4, 0, 0);
    }
    if m.is_capture() {
        let captured = m.capture().map(piece_points).unwrap_or(0);
        let attacker = piece_points(m.role());
        return (3, captured, -attacker);
    }
    if gives_check(pos, m) {
        return (2, 0, 0);
    }
    if Some(m) == killer {
        return (1, 0, 0);
    }
    (0, 0, 0)
}

/// Legal moves ordered for alpha-beta. Rust's sort is stable, so ties keep
/// shakmaty's generation order (analogous to python-chess's).
fn ordered_moves(pos: &Chess, killer: Option<Move>, tt_move: Option<Move>) -> Vec<Move> {
    let mut moves: Vec<Move> = pos.legal_moves().into_iter().collect();
    moves.sort_by(|a, b| {
        move_key(pos, *b, killer, tt_move).cmp(&move_key(pos, *a, killer, tt_move))
    });
    moves
}

/// Negamax with alpha-beta from the side-to-move's perspective.
pub fn negamax(pos: &Chess, depth: i32, mut alpha: f64, mut beta: f64, state: &mut SearchState) -> f64 {
    if pos.is_checkmate() {
        return -(MATE_SCORE + depth as f64);
    }
    if pos.is_stalemate() || pos.is_insufficient_material() {
        return 0.0;
    }
    if depth <= 0 {
        state.evals += 1;
        let score = evaluate_white(pos, state.weights);
        return if pos.turn() == Color::White { score } else { -score };
    }

    let original_alpha = alpha;
    let key = zobrist(pos);
    let mut tt_move = None;
    if let Some(entry) = state.tt.get(&key) {
        tt_move = entry.best_move;
        if entry.depth >= depth {
            match entry.flag {
                TT_EXACT => return entry.score,
                TT_LOWERBOUND => alpha = alpha.max(entry.score),
                TT_UPPERBOUND => beta = beta.min(entry.score),
                _ => {}
            }
            if alpha >= beta {
                return entry.score;
            }
        }
    }

    let mut best = f64::NEG_INFINITY;
    let mut best_move = None;
    let killer = state.killers.get(&depth).copied();
    for m in ordered_moves(pos, killer, tt_move) {
        let mut child = pos.clone();
        child.play_unchecked(m);
        let score = -negamax(&child, depth - 1, -beta, -alpha, state);
        if score > best {
            best = score;
            best_move = Some(m);
        }
        if best > alpha {
            alpha = best;
        }
        if alpha >= beta {
            if !m.is_capture() {
                state.killers.insert(depth, m);
            }
            break;
        }
    }

    let flag = if best <= original_alpha {
        TT_UPPERBOUND
    } else if best >= beta {
        TT_LOWERBOUND
    } else {
        TT_EXACT
    };
    state.tt.insert(key, TtEntry { depth, score: best, flag, best_move });

    best
}

/// Root-level ordered moves (no killer/TT context yet), matching the
/// `_order_moves(board)` call at the top of `choose_engine_move`.
pub fn root_moves(pos: &Chess) -> Vec<Move> {
    ordered_moves(pos, None, None)
}
