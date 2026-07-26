//! `chess_engine` -- the native backend for the self-play engine's per-move
//! search. `app.games.choose_engine_move` calls this module, keeping move
//! generation and the whole negamax tree in Rust. The extension is required
//! at application startup.

mod eval;
mod search;

use std::collections::HashMap;
use std::env;
use std::sync::{Mutex, OnceLock};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use shakmaty::fen::Fen;
use shakmaty::zobrist::Zobrist64;
use shakmaty::{CastlingMode, Chess, EnPassantMode, Position};

use crate::eval::{evaluate_white, forward_4, mover_material_advantage, Weights};
use crate::search::{negamax, root_moves, SearchState};

// Repetition-avoidance configuration used during root-move selection.
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

fn cached_zobrist(fen: &str) -> Option<u64> {
    static POSITION_CACHE: OnceLock<Mutex<HashMap<String, u64>>> = OnceLock::new();
    let cache = POSITION_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hash) = cache.lock().ok()?.get(fen).copied() {
        return Some(hash);
    }
    let hash = parse_position(fen).ok().map(|pos| zobrist(&pos))?;
    cache.lock().ok()?.insert(fen.to_owned(), hash);
    Some(hash)
}

/// Multiset of position keys from prior game history plus the current
/// position, all hashed by shakmaty so the count is internally consistent.
/// Used to reproduce python-chess `board.is_repetition(3)` at the root.
fn repetition_counts(current: &Chess, history_fens: &[String]) -> HashMap<u64, u32> {
    let mut counts: HashMap<u64, u32> = HashMap::new();
    *counts.entry(zobrist(current)).or_insert(0) += 1;
    for fen in history_fens {
        if let Some(hash) = cached_zobrist(fen) {
            *counts.entry(hash).or_insert(0) += 1;
        }
    }
    counts
}

/// Whether the side to move can claim threefold repetition immediately after
/// the candidate position has been added to `prior_counts`.
///
/// A claim is available either because the current position is already the
/// third occurrence, or because a legal move would create the third
/// occurrence. This mirrors python-chess `can_claim_threefold_repetition()`.
fn can_claim_threefold(
    pos: &Chess,
    prior_counts: &HashMap<u64, u32>,
    candidate_hash: u64,
) -> bool {
    if prior_counts.get(&candidate_hash).copied().unwrap_or(0) + 1 >= 3 {
        return true;
    }

    for m in pos.legal_moves() {
        let mut claimed = pos.clone();
        claimed.play_unchecked(m);
        let claimed_hash = zobrist(&claimed);
        let candidate_occurrence = u32::from(claimed_hash == candidate_hash);
        if prior_counts.get(&claimed_hash).copied().unwrap_or(0) + candidate_occurrence >= 2 {
            return true;
        }
    }
    false
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
    top_k_score_threshold=Some(3.0),
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
    top_k_score_threshold: Option<f64>,
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
    let mut repetition_claim_cache: HashMap<u64, bool> = HashMap::new();

    let mut state = SearchState::new(weights);
    let mut scored: Vec<(f64, shakmaty::Move)> = Vec::new();
    let mut immediate_stalemates: Vec<shakmaty::Move> = Vec::new();
    for m in root_moves(&pos) {
        let mut child = pos.clone();
        child.play_unchecked(m);
        if child.is_stalemate() {
            immediate_stalemates.push(m);
        }
        let mut value = -negamax(&child, depth - 1, f64::NEG_INFINITY, f64::INFINITY, &mut state);
        if avoid_repetition {
            // This catches both a move that creates the third occurrence
            // immediately and a move that gives the opponent a legal
            // claim-producing move on their next turn. The prior-position
            // counts are immutable for this root search, so cache the result
            // by candidate position and avoid cloning the full history map.
            let child_hash = zobrist(&child);
            let opponent_can_claim = *repetition_claim_cache
                .entry(child_hash)
                .or_insert_with(|| can_claim_threefold(&child, &rep_counts, child_hash));
            if opponent_can_claim {
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

    // A stalemate is a draw, so preserve it whenever every alternative is
    // evaluated as a loss. This makes the engine take a forced drawing
    // resource instead of choosing a materially or tactically losing move.
    if scored[0].0 < 0.0 && !immediate_stalemates.is_empty() {
        scored.retain(|(_, m)| immediate_stalemates.contains(m));
    }

    let top_n = top_k.max(1).min(scored.len() as i32) as usize;
    let candidate_count = match top_k_score_threshold {
        Some(threshold) => {
            let best_score = scored[0].0;
            scored[..top_n]
                .iter()
                .take_while(|(score, _)| best_score - *score <= threshold.max(0.0))
                .count()
                .max(1)
        }
        None => top_n,
    };

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let (score, chosen) = scored[rng.gen_range(0..candidate_count)];
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

/// Native fourth-order forward control, returned as (white, black).
#[pyfunction]
fn calculate_forward_4(fen: &str) -> PyResult<(i32, i32)> {
    let pos = parse_position(fen)?;
    Ok(forward_4(&pos))
}

#[pymodule]
fn chess_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(choose_engine_move, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_position, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_forward_4, m)?)?;
    Ok(())
}
