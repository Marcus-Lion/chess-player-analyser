//! `chess_engine` -- a native drop-in for the self-play engine's per-move
//! search. `app/self_play.py` imports this and calls `choose_engine_move`
//! instead of the pure-Python `app.games.choose_engine_move`, keeping move
//! generation and the whole negamax tree in Rust. It falls back to Python if
//! this extension isn't built.

mod eval;
mod search;

use std::collections::HashMap;
use std::env;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use shakmaty::fen::Fen;
use shakmaty::zobrist::Zobrist64;
use shakmaty::{CastlingMode, Chess, EnPassantMode, Position};

use crate::eval::{evaluate_white, mover_material_advantage, Weights};
use crate::search::{negamax, root_moves, SearchState};

// Matches REPETITION_AVOIDANCE_* in games.py:903-904.
const DEFAULT_REPETITION_AVOIDANCE_MATERIAL_PAWNS: i32 = 1;
const REPETITION_AVOIDANCE_PENALTY: f64 = 500.0;

fn repetition_avoidance_material_pawns() -> i32 {
    env::var("REPETITION_AVOIDANCE_MATERIAL_PAWNS")
        .ok()
        .and_then(|raw| raw.trim().parse::<i32>().ok())
        .unwrap_or(DEFAULT_REPETITION_AVOIDANCE_MATERIAL_PAWNS)
        .max(0)
}

fn parse_position(fen: &str) -> PyResult<Chess> {
    let parsed: Fen = fen
        .parse()
        .map_err(|e| PyValueError::new_err(format!("invalid FEN {fen:?}: {e}")))?;
    parsed
        .into_position(CastlingMode::Standard)
        .map_err(|e| PyValueError::new_err(format!("illegal position {fen:?}: {e}")))
}

fn zobrist(pos: &Chess) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

/// Multiset of position keys from prior game history plus the current
/// position, all hashed by shakmaty so the count is internally consistent.
/// Used to reproduce python-chess `board.is_repetition(3)` at the root.
fn repetition_counts(current: &Chess, history_fens: &[String]) -> HashMap<u64, u32> {
    let mut counts: HashMap<u64, u32> = HashMap::new();
    *counts.entry(zobrist(current)).or_insert(0) += 1;
    for fen in history_fens {
        if let Ok(pos) = parse_position(fen) {
            *counts.entry(zobrist(&pos)).or_insert(0) += 1;
        }
    }
    counts
}

/// Pick a move via negamax search -- the Rust equivalent of
/// `app.games.choose_engine_move`.
///
/// Returns `(uci, score, evaluations)`: the chosen move in UCI, its search
/// score, and the number of leaf positions statically evaluated (so
/// self-play's evals/move stats still populate).
///
/// `history_fens` are the FENs of every position that occurred *earlier* in
/// the game (excluding the current one); they drive the repetition-avoidance
/// penalty when the mover is materially ahead.
#[pyfunction]
#[pyo3(signature = (
    fen,
    depth,
    top_k,
    seed,
    legal_moves_weight,
    material_score_weight,
    forward_score_weight,
    center_control_weight,
    checkmate_weight,
    history_fens,
))]
#[allow(clippy::too_many_arguments)]
fn choose_engine_move(
    fen: &str,
    depth: i32,
    top_k: i32,
    seed: Option<u64>,
    legal_moves_weight: f64,
    material_score_weight: f64,
    forward_score_weight: f64,
    center_control_weight: f64,
    checkmate_weight: f64,
    history_fens: Vec<String>,
) -> PyResult<(String, f64, u64)> {
    let pos = parse_position(fen)?;
    let depth = depth.max(1);
    let weights = Weights {
        legal_moves: legal_moves_weight,
        material: material_score_weight,
        forward: forward_score_weight,
        center: center_control_weight,
        checkmate: checkmate_weight,
    };

    let repetition_threshold = repetition_avoidance_material_pawns();
    let avoid_repetition =
        mover_material_advantage(&pos).abs() >= repetition_threshold;
    let rep_counts = if avoid_repetition {
        repetition_counts(&pos, &history_fens)
    } else {
        HashMap::new()
    };

    let mut state = SearchState::new(weights);
    let mut scored: Vec<(f64, shakmaty::Move)> = Vec::new();
    for m in root_moves(&pos) {
        let mut child = pos.clone();
        child.play_unchecked(m);
        let mut value = -negamax(&child, depth - 1, f64::NEG_INFINITY, f64::INFINITY, &mut state);
        if avoid_repetition {
            // python-chess is_repetition(3): the move creates a 3rd occurrence
            // when this position already appears >= 2 times in the history.
            if rep_counts.get(&zobrist(&child)).copied().unwrap_or(0) >= 2 {
                value -= REPETITION_AVOIDANCE_PENALTY;
            }
        }
        scored.push((value, m));
    }

    if scored.is_empty() {
        return Err(PyValueError::new_err("No legal moves available"));
    }

    // Descending by score; stable so ties keep generation order.
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let top_n = top_k.max(1).min(scored.len() as i32) as usize;

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let (score, chosen) = scored[rng.gen_range(0..top_n)];
    let uci = chosen.to_uci(CastlingMode::Standard).to_string();
    Ok((uci, score, state.evals))
}

/// White-perspective static evaluation of a FEN -- exposed for parity testing
/// against `app.games._evaluate_position`.
#[pyfunction]
#[pyo3(signature = (
    fen,
    legal_moves_weight,
    material_score_weight,
    forward_score_weight,
    center_control_weight,
    checkmate_weight,
))]
fn evaluate_position(
    fen: &str,
    legal_moves_weight: f64,
    material_score_weight: f64,
    forward_score_weight: f64,
    center_control_weight: f64,
    checkmate_weight: f64,
) -> PyResult<f64> {
    let pos = parse_position(fen)?;
    Ok(evaluate_white(
        &pos,
        Weights {
            legal_moves: legal_moves_weight,
            material: material_score_weight,
            forward: forward_score_weight,
            center: center_control_weight,
            checkmate: checkmate_weight,
        },
    ))
}

#[pymodule]
fn chess_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(choose_engine_move, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_position, m)?)?;
    Ok(())
}
