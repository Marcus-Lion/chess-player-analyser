//! Static position evaluation, a faithful port of the White-perspective
//! heuristic in `app/games.py` (`_evaluate_position` and its helpers).
//!
//! It implements the evaluation exposed through the Python API: material
//! differential, first-order forward
//! control on the same 20 forward squares (`get_board_control`), central
//! control of d4/e4/d5/e5, both-side mobility (the `_both_mobilities`
//! turn-flip trick), and the "goal is checkmate" king-pressure term
//! (`_mate_pressure` / `_king_escape_squares`).

use shakmaty::attacks;
use shakmaty::{Board, Chess, Color, Position, Role, Square};

/// Score-blend weights, matching the keyword args threaded through
/// `choose_engine_move` / `_evaluate_position`.
#[derive(Clone, Copy)]
pub struct Weights {
    pub legal_moves: f64,
    pub material: f64,
    pub forward: f64,
    pub center: f64,
    pub checkmate: f64,
}

/// Material point values matching `app.games.PIECE_POINTS`. King is 0.
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

/// White material minus Black material, in points.
fn material_diff(board: &Board) -> i32 {
    let white = board.by_color(Color::White);
    let black = board.by_color(Color::Black);
    let mut score = 0;
    for role in [Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen] {
        let rb = board.by_role(role);
        let w = (rb & white).into_iter().count() as i32;
        let b = (rb & black).into_iter().count() as i32;
        score += (w - b) * piece_points(role);
    }
    score
}

// The forward squares scanned by `app.games.get_board_control`:
// White's are ranks 2-3, files d-h; Black's are ranks 6-7, files d-h.
const WHITE_FORWARD: [Square; 10] = [
    Square::D2, Square::E2, Square::F2, Square::G2, Square::H2,
    Square::D3, Square::E3, Square::F3, Square::G3, Square::H3,
];
const BLACK_FORWARD: [Square; 10] = [
    Square::D6, Square::E6, Square::F6, Square::G6, Square::H6,
    Square::D7, Square::E7, Square::F7, Square::G7, Square::H7,
];

/// (white_control, black_control): how many of each side's forward squares
/// that side attacks. Mirrors `get_board_control`.
fn board_control(board: &Board) -> (i32, i32) {
    let occupied = board.occupied();
    let mut white = 0;
    for &sq in &WHITE_FORWARD {
        if !board.attacks_to(sq, Color::White, occupied).is_empty() {
            white += 1;
        }
    }
    let mut black = 0;
    for &sq in &BLACK_FORWARD {
        if !board.attacks_to(sq, Color::Black, occupied).is_empty() {
            black += 1;
        }
    }
    (white, black)
}

/// White minus Black control of the four central squares.
fn center_diff(board: &Board) -> i32 {
    let occupied = board.occupied();
    let mut white = 0;
    let mut black = 0;
    for sq in [Square::D4, Square::E4, Square::D5, Square::E5] {
        if !board.attacks_to(sq, Color::White, occupied).is_empty() {
            white += 1;
        }
        if !board.attacks_to(sq, Color::Black, occupied).is_empty() {
            black += 1;
        }
    }
    white - black
}

/// Squares the `defender` king could flee to: not blocked by its own pieces
/// and not attacked by the enemy (`app.games._king_escape_squares`).
/// Like python-chess `is_attacked_by`, the king stays on the board when
/// testing enemy attacks, so slider x-rays through the king are ignored.
fn king_escape_squares(board: &Board, king_color: Color) -> i32 {
    let king_sq = match board.king_of(king_color) {
        Some(sq) => sq,
        None => return 0,
    };
    let enemy = king_color.other();
    let occupied = board.occupied();
    let mut escapes = 0;
    for sq in attacks::king_attacks(king_sq) {
        if let Some(piece) = board.piece_at(sq) {
            if piece.color == king_color {
                continue;
            }
        }
        if !board.attacks_to(sq, enemy, occupied).is_empty() {
            continue;
        }
        escapes += 1;
    }
    escapes
}

/// White-perspective checkmate pressure (`app.games._mate_pressure`):
/// reward each side for driving the enemy king toward the edge/corner and
/// stripping its escape squares.
fn mate_pressure(board: &Board) -> f64 {
    let mut pressure = 0.0;
    for (defender, sign) in [(Color::Black, 1.0), (Color::White, -1.0)] {
        let king_sq = match board.king_of(defender) {
            Some(sq) => sq,
            None => continue,
        };
        let file = (king_sq.to_u32() % 8) as f64;
        let rank = (king_sq.to_u32() / 8) as f64;
        // 2 in the centre, up to 14 in a corner.
        let corner_proximity = (2.0 * file - 7.0).abs() + (2.0 * rank - 7.0).abs();
        let escapes = king_escape_squares(board, defender) as f64;
        pressure += sign * (corner_proximity - 2.0 * escapes);
    }
    pressure
}

/// Legal-move counts for (White, Black) regardless of whose turn it is,
/// matching `app.games._both_mobilities`. The side to move uses its
/// real legal moves; the other side is counted by flipping the turn.
fn both_mobilities(pos: &Chess) -> (i32, i32) {
    let stm = pos.turn();
    let stm_count = pos.legal_moves().len() as i32;
    let other_count = opponent_mobility(pos, stm.other());
    if stm == Color::White {
        (stm_count, other_count)
    } else {
        (other_count, stm_count)
    }
}

/// Legal moves available to `color` when it is *not* the side to move.
///
/// python-chess permissively counts these by flipping `board.turn`. shakmaty
/// validates positions, so we swap the turn via `swap_turn`, discarding the
/// now-orphaned en passant square. The one case shakmaty cannot represent is
/// when the real side to move is in check (flipping leaves that king in an
/// "opposite check"); there we fall back to a pseudo-legal count.
fn opponent_mobility(pos: &Chess, color: Color) -> i32 {
    let recovered = pos
        .clone()
        .swap_turn()
        .or_else(|e| e.ignore_invalid_ep_square())
        .or_else(|e| e.ignore_impossible_check());
    match recovered {
        Ok(flipped) => flipped.legal_moves().len() as i32,
        Err(_) => pseudo_mobility(pos.board(), color),
    }
}

/// Pseudo-legal move count for `color`, used only for the rare in-check leaf
/// where the turn cannot be flipped into a legal shakmaty position. Ignores
/// pins and moving into check, so it slightly over-counts, but keeps the
/// mobility term defined and deterministic.
fn pseudo_mobility(board: &Board, color: Color) -> i32 {
    let occupied = board.occupied();
    let own = board.by_color(color);
    let enemy = board.by_color(color.other());
    let mut count = 0;
    for sq in own {
        let piece = match board.piece_at(sq) {
            Some(p) => p,
            None => continue,
        };
        if piece.role == Role::Pawn {
            // Diagonal captures.
            let caps = attacks::pawn_attacks(color, sq) & enemy;
            count += caps.into_iter().count() as i32;
            // Forward pushes.
            let idx = sq.to_u32();
            let rank = idx / 8;
            let (one_step, start_rank) = if color == Color::White {
                (idx + 8, 1)
            } else {
                (idx.wrapping_sub(8), 6)
            };
            if one_step < 64 && board.piece_at(Square::new(one_step)).is_none() {
                count += 1;
                if rank == start_rank {
                    let two_step = if color == Color::White { idx + 16 } else { idx - 16 };
                    if two_step < 64 && board.piece_at(Square::new(two_step)).is_none() {
                        count += 1;
                    }
                }
            }
        } else {
            let targets = attacks::attacks(sq, piece, occupied) & !own;
            count += targets.into_iter().count() as i32;
        }
    }
    count
}

/// Round to 2 decimals, matching Python's `round(x, 2)` closely enough for
/// strategic parity (Python uses banker's rounding; the difference only
/// surfaces on exact .xx5 ties, which are negligible here).
fn round2(x: f64) -> f64 {
    (x * 100.0).round_ties_even() / 100.0
}

/// White-perspective static evaluation exposed by `app.games._evaluate_position`:
/// weighted blend of mobility, material, forward control, center control, and
/// checkmate pressure.
pub fn evaluate_white(pos: &Chess, w: Weights) -> f64 {
    let board = pos.board();
    let (white_control, black_control) = board_control(board);
    let material = material_diff(board);
    let center = center_diff(board);
    let (white_mobility, black_mobility) = both_mobilities(pos);
    let pressure = mate_pressure(board);
    round2(
        w.legal_moves * (white_mobility - black_mobility) as f64
            + w.material * material as f64
            + w.forward * (white_control - black_control) as f64
            + w.center * center as f64
            + w.checkmate * pressure,
    )
}

/// Material lead for the side to move, in points. Positive means the mover is
/// ahead.
pub fn mover_material_advantage(pos: &Chess) -> i32 {
    let advantage = material_diff(pos.board());
    if pos.turn() == Color::White {
        advantage
    } else {
        -advantage
    }
}
