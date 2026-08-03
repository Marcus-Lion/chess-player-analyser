//! `chess_engine` -- the native backend for the self-play engine's per-move
//! search. `app.games.choose_engine_move` calls this module, keeping move
//! generation and the whole negamax tree in Rust. The extension is required
//! at application startup.

mod eval;
mod search;

#[cfg(feature = "python")]
use std::collections::HashMap;
#[cfg(feature = "python")]
use std::time::Instant;
#[cfg(feature = "python")]
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use rand::rngs::StdRng;
#[cfg(feature = "python")]
use rand::{RngExt, SeedableRng};

#[cfg(feature = "python")]
use shakmaty::fen::Fen;
#[cfg(feature = "python")]
use shakmaty::san::San;
#[cfg(feature = "python")]
use shakmaty::zobrist::Zobrist64;
#[cfg(feature = "python")]
use shakmaty::{CastlingMode, Chess, EnPassantMode, Position};

#[cfg(feature = "python")]
use crate::eval::{center_control, evaluate_white, flank_control, forward_1_2_3, forward_4, forward_material, king_escape_squares, mate_pressure, mover_material_advantage, phase_value, Weights};
#[cfg(feature = "python")]
use crate::search::{negamax, root_moves, SearchState};

#[cfg(feature = "python")]
const REPETITION_PENALTY: f64 = 500.0;

#[cfg(feature = "python")]
fn parse_position(fen: &str) -> PyResult<Chess> {
    let parsed: Fen = fen
        .parse()
        .map_err(|e| PyValueError::new_err(format!("invalid FEN {fen:?}: {e}")))?;
    parsed
        .into_position(CastlingMode::Standard)
        .map_err(|e| PyValueError::new_err(format!("illegal position {fen:?}: {e}")))
}

#[cfg(feature = "python")]
fn zobrist(pos: &Chess) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

#[cfg(feature = "python")]
fn cached_zobrist(fen: &str) -> PyResult<u64> {
    static POSITION_CACHE: OnceLock<Mutex<HashMap<String, u64>>> = OnceLock::new();
    let cache = POSITION_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let guard = cache
            .lock()
            .map_err(|_| PyValueError::new_err("zobrist cache lock poisoned"))?;
        if let Some(hash) = guard.get(fen).copied() {
            return Ok(hash);
        }
    }

    let hash = zobrist(&parse_position(fen)?);
    let mut guard = cache
        .lock()
        .map_err(|_| PyValueError::new_err("zobrist cache lock poisoned"))?;
    guard.insert(fen.to_owned(), hash);
    Ok(hash)
}

#[cfg(feature = "python")]
#[pyfunction]
fn zobrist_fen(fen: &str) -> PyResult<u64> {
    cached_zobrist(fen)
}

#[cfg(feature = "python")]
fn can_claim_threefold(pos: &Chess, repetition_counts: &HashMap<u64, u32>) -> bool {
    for m in pos.legal_moves() {
        let mut claimed = pos.clone();
        claimed.play_unchecked(m);
        let claimed_hash = zobrist(&claimed);
        if repetition_counts.get(&claimed_hash).copied().unwrap_or(0) >= 3 {
            return true;
        }
    }
    false
}

#[cfg(feature = "python")]
fn auto_depth(pos: &Chess, max_depth: i32) -> i32 {
    let values = [(shakmaty::Role::Pawn, 1), (shakmaty::Role::Knight, 3),
        (shakmaty::Role::Bishop, 3), (shakmaty::Role::Rook, 5),
        (shakmaty::Role::Queen, 9)];
    let board = pos.board();
    let remaining: i32 = values.iter().map(|(role, value)|
        (board.by_role(*role).into_iter().count() as i32) * value).sum();
    let remaining = remaining.clamp(0, 78);
    let fraction = ((78 - remaining) as f64 / 78.0).powf(0.45);
    let max_depth = max_depth.max(1);
    let min_depth = 2.min(max_depth);
    (min_depth as f64 * (max_depth as f64 / min_depth as f64).powf(fraction))
        .round().clamp(min_depth as f64, max_depth as f64) as i32
}

/// Play an entire self-play game without crossing the Python boundary for
/// individual plies. The returned move list is UCI; Python only uses it to
/// build the presentation PGN after the native loop has finished.
#[pyfunction]
#[pyo3(signature = (
    fen,
    max_turns,
    top_k,
    seed,
    legal_moves_weight,
    material_score_weight,
    forward_score_weight,
    center_control_weight,
    black_legal_moves_weight,
    black_material_score_weight,
    black_forward_score_weight,
    black_center_control_weight,
    checkmate_weight,
    depth=None,
    max_depth=2,
    top_k_score_threshold=Some(3.0),
    forward_material_score_weight=0.25,
    black_forward_material_score_weight=0.25,
))]
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "python")]
fn play_self_game_native(
    fen: &str,
    max_turns: i32,
    top_k: i32,
    seed: u64,
    legal_moves_weight: f64,
    material_score_weight: f64,
    forward_score_weight: f64,
    center_control_weight: f64,
    black_legal_moves_weight: f64,
    black_material_score_weight: f64,
    black_forward_score_weight: f64,
    black_center_control_weight: f64,
    checkmate_weight: f64,
    depth: Option<i32>,
    max_depth: i32,
    top_k_score_threshold: Option<f64>,
    forward_material_score_weight: f64,
    black_forward_material_score_weight: f64,
) -> PyResult<(String, String, u32, Vec<String>, u64, Vec<f64>)> {
    let mut pos = parse_position(fen)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut repetitions = HashMap::new();
    repetitions.insert(zobrist(&pos), 1);
    let mut moves = Vec::new();
    let mut turn_durations = Vec::new();
    let mut evals = 0;
    let mut result = String::new();
    let mut termination = String::new();

    for _ in 0..max_turns.max(0) {
        if pos.is_checkmate() { result = if pos.turn() == shakmaty::Color::Black { "1-0" } else { "0-1" }.into(); termination = "checkmate".into(); break; }
        if pos.is_stalemate() { result = "1/2-1/2".into(); termination = "stalemate".into(); break; }
        if pos.is_insufficient_material() { result = "1/2-1/2".into(); termination = "insufficient material".into(); break; }
        if repetitions.get(&zobrist(&pos)).copied().unwrap_or(0) >= 5 { result = "1/2-1/2".into(); termination = "fivefold repetition".into(); break; }
        if pos.halfmoves() >= 150 { result = "1/2-1/2".into(); termination = "75-move rule".into(); break; }
        let move_depth = depth.unwrap_or_else(|| auto_depth(&pos, max_depth));
        let turn_started = Instant::now();
        let weights = if pos.turn() == shakmaty::Color::White {
            Weights { legal_moves: legal_moves_weight, material: material_score_weight,
                forward: forward_score_weight, forward_material: forward_material_score_weight,
                center: center_control_weight, checkmate: checkmate_weight }
        } else {
            Weights { legal_moves: black_legal_moves_weight, material: black_material_score_weight,
                forward: black_forward_score_weight, forward_material: black_forward_material_score_weight,
                center: black_center_control_weight, checkmate: checkmate_weight }
        };
        let mut state = SearchState::new(weights);
        let mut scored = Vec::new();
        let mut immediate_stalemates = Vec::new();
        let mover_advantage = mover_material_advantage(&pos);
        for m in root_moves(&pos) {
            let mut child = pos.clone(); child.play_unchecked(m);
            if child.is_stalemate() { immediate_stalemates.push(m); }
            let mut value = -negamax(&child, move_depth.max(1) - 1, f64::NEG_INFINITY, f64::INFINITY, &mut state);
            let mut candidate_counts = repetitions.clone();
            let child_hash = zobrist(&child);
            *candidate_counts.entry(child_hash).or_insert(0) += 1;
            if candidate_counts.get(&child_hash).copied().unwrap_or(0) >= 3 || can_claim_threefold(&child, &candidate_counts) {
                value -= if mover_advantage > 0 { REPETITION_PENALTY } else if mover_advantage < 0 { -REPETITION_PENALTY } else { 0.0 };
            }
            scored.push((value, m));
        }
        if scored.is_empty() { break; }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        if scored[0].0 < 0.0 && !immediate_stalemates.is_empty() { scored.retain(|(_, m)| immediate_stalemates.contains(m)); }
        let top_n = (top_k.max(1) as usize).min(scored.len());
        let count = match top_k_score_threshold { Some(t) => scored[..top_n].iter().take_while(|(s, _)| scored[0].0 - *s <= t.max(0.0)).count().max(1), None => top_n };
        let chosen_index = rng.random_range(0..count);
        let chosen = scored[chosen_index].1;
        moves.push(chosen.to_uci(CastlingMode::Standard).to_string());
        pos.play_unchecked(chosen);
        *repetitions.entry(zobrist(&pos)).or_insert(0) += 1;
        evals += state.evals;
        turn_durations.push(turn_started.elapsed().as_secs_f64());
    }
    if result.is_empty() {
        if pos.is_checkmate() { result = if pos.turn() == shakmaty::Color::Black { "1-0" } else { "0-1" }.into(); termination = "checkmate".into(); }
        else if pos.is_stalemate() { result = "1/2-1/2".into(); termination = "stalemate".into(); }
        else if pos.is_insufficient_material() { result = "1/2-1/2".into(); termination = "insufficient material".into(); }
        else if repetitions.get(&zobrist(&pos)).copied().unwrap_or(0) >= 5 { result = "1/2-1/2".into(); termination = "fivefold repetition".into(); }
        else if pos.halfmoves() >= 150 { result = "1/2-1/2".into(); termination = "75-move rule".into(); }
        else if repetitions.get(&zobrist(&pos)).copied().unwrap_or(0) >= 3 || can_claim_threefold(&pos, &repetitions) { result = "1/2-1/2".into(); termination = "3-fold-rep".into(); }
        else { result = "1/2-1/2".into(); termination = "max turns".into(); }
    }
    Ok((result, termination, moves.len() as u32, moves, evals, turn_durations))
}

/// Pick a move via negamax search -- the Rust equivalent of
/// `app.games.choose_engine_move`.
///
/// Returns `(uci, score, evaluations)`: the chosen move in UCI, its search
/// score, and the number of leaf positions statically evaluated (so
/// self-play's evals/move stats still populate).
///
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
    repetition_counts,
    top_k_score_threshold=Some(3.0),
    forward_material_score_weight=0.25,
))]
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "python")]
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
    repetition_counts: HashMap<u64, u32>,
    top_k_score_threshold: Option<f64>,
    forward_material_score_weight: f64,
) -> PyResult<(String, f64, u64)> {
    let pos = parse_position(fen)?;
    let depth = depth.max(1);
    let weights = Weights {
        legal_moves: legal_moves_weight,
        material: material_score_weight,
        forward: forward_score_weight,
        forward_material: forward_material_score_weight,
        center: center_control_weight,
        checkmate: checkmate_weight,
    };

    let mover_advantage = mover_material_advantage(&pos);

    let mut state = SearchState::new(weights);
    let mut scored: Vec<(f64, shakmaty::Move)> = Vec::new();
    let mut immediate_stalemates: Vec<shakmaty::Move> = Vec::new();
    for m in root_moves(&pos) {
        let mut child = pos.clone();
        child.play_unchecked(m);
        if child.is_stalemate() {
            immediate_stalemates.push(m);
        }
        let mut value = -negamax(
            &child,
            depth - 1,
            f64::NEG_INFINITY,
            f64::INFINITY,
            &mut state,
        );
        let mut candidate_counts = repetition_counts.clone();
        let child_hash = zobrist(&child);
        *candidate_counts.entry(child_hash).or_insert(0) += 1;
        if candidate_counts.get(&child_hash).copied().unwrap_or(0) >= 3
            || can_claim_threefold(&child, &candidate_counts)
        {
            value -= if mover_advantage > 0 {
                REPETITION_PENALTY
            } else if mover_advantage < 0 {
                -REPETITION_PENALTY
            } else {
                0.0
            };
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
        None => {
            let mut system_rng = rand::rng();
            StdRng::from_rng(&mut system_rng)
        }
    };
    let chosen_index = rng.random_range(0..candidate_count);
    let (score, chosen) = scored[chosen_index];
    let uci = chosen.to_uci(CastlingMode::Standard).to_string();
    Ok((uci, score, state.evals))
}

/// Choose a move while building repetition history from FENs in Rust.
/// Python supplies the historical positions because it owns the python-chess
/// move stack; hashing and repetition bookkeeping stay native.
#[pyfunction]
#[pyo3(signature = (
    fen,
    history_fens,
    depth,
    top_k,
    seed,
    legal_moves_weight,
    material_score_weight,
    forward_score_weight,
    center_control_weight,
    checkmate_weight,
    top_k_score_threshold=Some(3.0),
    forward_material_score_weight=0.25,
))]
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "python")]
fn choose_engine_move_from_history(
    fen: &str,
    history_fens: Vec<String>,
    depth: i32,
    top_k: i32,
    seed: u64,
    legal_moves_weight: f64,
    material_score_weight: f64,
    forward_score_weight: f64,
    center_control_weight: f64,
    checkmate_weight: f64,
    top_k_score_threshold: Option<f64>,
    forward_material_score_weight: f64,
) -> PyResult<(String, f64, u64)> {
    let mut repetition_counts = HashMap::new();
    for historical_fen in history_fens {
        let hash = cached_zobrist(&historical_fen)?;
        *repetition_counts.entry(hash).or_insert(0) += 1;
    }
    if repetition_counts.is_empty() {
        let hash = cached_zobrist(fen)?;
        repetition_counts.insert(hash, 1);
    }
    choose_engine_move(
        fen,
        depth,
        top_k,
        Some(seed),
        legal_moves_weight,
        material_score_weight,
        forward_score_weight,
        center_control_weight,
        checkmate_weight,
        repetition_counts,
        top_k_score_threshold,
        forward_material_score_weight,
    )
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
    forward_material_score_weight=0.25,
))]
#[cfg(feature = "python")]
fn evaluate_position(
    fen: &str,
    legal_moves_weight: f64,
    material_score_weight: f64,
    forward_score_weight: f64,
    center_control_weight: f64,
    checkmate_weight: f64,
    forward_material_score_weight: f64,
) -> PyResult<f64> {
    let pos = parse_position(fen)?;
    Ok(evaluate_white(
        &pos,
        Weights {
            legal_moves: legal_moves_weight,
            material: material_score_weight,
            forward: forward_score_weight,
            forward_material: forward_material_score_weight,
            center: center_control_weight,
            checkmate: checkmate_weight,
        },
    ))
}

/// Native fourth-order forward control, returned as (white, black).
#[pyfunction]
#[cfg(feature = "python")]
fn calculate_forward_4(fen: &str) -> PyResult<(i32, i32)> {
    let pos = parse_position(fen)?;
    forward_4(&pos).map_err(PyValueError::new_err)
}

/// Return first-, second-, and third-order forward control in one native call.
#[pyfunction]
#[cfg(feature = "python")]
fn calculate_forward(fen: &str) -> PyResult<((i32, i32), (i32, i32), (i32, i32))> {
    forward_1_2_3(&parse_position(fen)?).map_err(PyValueError::new_err)
}

#[pyfunction]
#[cfg(feature = "python")]
fn calculate_center_control(fen: &str) -> PyResult<(i32, i32)> {
    Ok(center_control(&parse_position(fen)?.board()))
}

#[pyfunction]
#[cfg(feature = "python")]
fn calculate_flank_control(fen: &str) -> PyResult<(i32, i32)> {
    Ok(flank_control(&parse_position(fen)?.board()))
}

#[pyfunction]
#[cfg(feature = "python")]
fn calculate_forward_material(fen: &str) -> PyResult<(i32, i32)> {
    Ok(forward_material(&parse_position(fen)?.board()))
}

#[pyfunction]
#[cfg(feature = "python")]
fn calculate_king_escape_squares(fen: &str, white: bool) -> PyResult<i32> {
    Ok(king_escape_squares(&parse_position(fen)?.board(), if white { shakmaty::Color::White } else { shakmaty::Color::Black }))
}

#[pyfunction]
#[cfg(feature = "python")]
fn calculate_mate_pressure(fen: &str) -> PyResult<f64> {
    Ok(mate_pressure(&parse_position(fen)?.board()))
}

#[pyfunction]
#[cfg(feature = "python")]
fn calculate_auto_search_depth(fen: &str, max_depth: i32) -> PyResult<i32> {
    Ok(auto_depth(&parse_position(fen)?, max_depth))
}

#[pyfunction]
#[cfg(feature = "python")]
fn calculate_phase_value(fen: &str) -> PyResult<f64> {
    Ok(phase_value(&parse_position(fen)?.board()))
}

/// Generate the legal SAN list, two-ply SAN tree, and first-order move scores
/// without crossing the Python boundary for each candidate move.
#[pyfunction]
#[cfg(feature = "python")]
fn legal_moves_and_tree(
    fen: &str,
) -> PyResult<(Vec<String>, HashMap<String, Vec<String>>, HashMap<String, i32>)> {
    let pos = parse_position(fen)?;
    let mut sans = Vec::new();
    let mut tree = HashMap::new();
    let mut scores = HashMap::new();
    for m in pos.legal_moves() {
        let san = San::from_move(&pos, m).to_string();
        let mut child = pos.clone();
        child.play_unchecked(m);
        let replies = child
            .legal_moves()
            .into_iter()
            .map(|reply| San::from_move(&child, reply).to_string())
            .collect();
        // The viewer's move score is forward-square control difference.
        let forward = crate::eval::board_control(child.board());
        sans.push(san.clone());
        tree.insert(san, replies);
        scores.insert(sans.last().unwrap().clone(), forward.0 - forward.1);
    }
    Ok((sans, tree, scores))
}

#[pymodule]
#[cfg(feature = "python")]
fn chess_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(choose_engine_move, m)?)?;
    m.add_function(wrap_pyfunction!(choose_engine_move_from_history, m)?)?;
    m.add_function(wrap_pyfunction!(play_self_game_native, m)?)?;
    m.add_function(wrap_pyfunction!(zobrist_fen, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_position, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_forward_4, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_forward, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_center_control, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_flank_control, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_forward_material, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_king_escape_squares, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_mate_pressure, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_auto_search_depth, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_phase_value, m)?)?;
    m.add_function(wrap_pyfunction!(legal_moves_and_tree, m)?)?;
    Ok(())
}
