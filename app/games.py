from __future__ import annotations

import math
import re
import random
from dataclasses import dataclass
from io import StringIO

import chess
import chess.pgn
import chess.svg

try:
    import chess_engine
    # Maturin may install the extension as ``chess_engine.chess_engine``
    # when building from a Windows virtual environment. Normalize both wheel
    # layouts to the module API used by this application.
    if not hasattr(chess_engine, "choose_engine_move"):
        from chess_engine import chess_engine as chess_engine
    missing_native = [
        name for name in (
            "play_self_game_native", "calculate_forward", "calculate_forward_4",
            "calculate_center_control", "calculate_flank_control",
            "calculate_king_escape_squares", "calculate_mate_pressure",
            "calculate_auto_search_depth", "calculate_phase_value",
            "legal_moves_and_tree",
            "choose_engine_move_from_history",
        )
        if not hasattr(chess_engine, name)
    ]
    if missing_native:
        raise RuntimeError(
            "The installed chess_engine extension is stale and does not expose "
            f"{', '.join(missing_native)}. Stop running Python processes, then rebuild "
            "it with `uv run --with maturin maturin develop --release "
            "-m engine/Cargo.toml`."
        )
except ImportError as exc:  # pragma: no cover - depends on local wheel installation
    raise RuntimeError(
        "The native chess_engine extension is required. Build/install it with "
        "`maturin develop --release -m engine/Cargo.toml`."
    ) from exc

from app.eco import eco_name


@dataclass
class GameSummary:
    index: int
    white: str
    black: str
    result: str
    date: str
    time_control: str
    eco: str
    eco_name: str
    user_color: str
    user_result: str


@dataclass
class GamePosition:
    ply: int
    move_number: int
    san: str
    side: str
    fen: str
    svg: str
    svg_moves: str
    legal_moves: list[str]
    move_tree: dict[str, list[str]]  # move_san -> list of response_sans
    move_scores: dict[str, int]  # move_san -> strength score of resulting position
    forward_1: dict[str, int]  # {"White": count, "Black": count}
    forward_2: dict[str, int]
    forward_3: dict[str, int]
    forward_4: dict[str, int]
    material: dict[str, int]  # {"White": points, "Black": points}
    center: dict[str, int]  # {"White": count, "Black": count}
    flank: dict[str, int]  # {"White": count, "Black": count}
    phase: str  # Opening, Middlegame, or Endgame
    weight_percentages: dict[str, int]
    forward_score: int  # (W_f1 + W_f2) - (B_f1 + B_f2)
    material_score: int  # White material - Black material
    center_score: int  # White center - Black center
    score: int  # Legal move count for the side to move
    total_score: float  # Weighted blend of legal moves, material, forward, and center
    score_breakdown: list[dict[str, str | float]]  # score, weight, and score * weight per component
    blunder_score: float  # Eval swing in the mover's favor lost to the opponent's best reply
    severity: str  # "" | "Inaccuracy" | "Mistake" | "Blunder", from _move_severity(blunder_score)


@dataclass
class GameDetail:
    index: int
    white: str
    black: str
    result: str
    date: str
    time_control: str
    eco: str
    eco_name: str
    positions: list[GamePosition]


def _outcome_for(color: str, result: str) -> str:
    if result == "1/2-1/2":
        return "Draw"
    if result == "1-0":
        return "Win" if color == "White" else "Loss"
    if result == "0-1":
        return "Win" if color == "Black" else "Loss"
    return "Unknown"


def _result_summary(result: str, white: str = "White", black: str = "Black") -> dict[str, str]:
    """Summarize a finished game in winner/loser/draw terms."""
    if result == "1-0":
        return {
            "status": "White wins",
            "winner": white,
            "loser": black,
            "result": result,
        }
    if result == "0-1":
        return {
            "status": "Black wins",
            "winner": black,
            "loser": white,
            "result": result,
        }
    if result == "1/2-1/2":
        return {
            "status": "Draw",
            "winner": "",
            "loser": "",
            "result": result,
        }
    return {
        "status": "Unknown",
        "winner": "",
        "loser": "",
        "result": result,
    }


def load_game_summaries(pgn_text: str, username: str | None = None) -> list[GameSummary]:
    """Read only the headers of every game in ``pgn_text`` (fast)."""
    username_l = (username or "").lower()
    pgn = StringIO(pgn_text)
    summaries: list[GameSummary] = []
    index = 0

    while True:
        headers = chess.pgn.read_headers(pgn)
        if headers is None:
            break
        index += 1

        white = headers.get("White", "")
        black = headers.get("Black", "")
        result = headers.get("Result", "*")

        if username_l and black.lower() == username_l:
            user_color = "Black"
        elif username_l and white.lower() == username_l:
            user_color = "White"
        else:
            user_color = ""

        summaries.append(
            GameSummary(
                index=index,
                white=white,
                black=black,
                result=result,
                date=headers.get("UTCDate", headers.get("Date", "")),
                time_control=headers.get("TimeControl", ""),
                eco=headers.get("ECO", ""),
                eco_name=eco_name(headers.get("ECO", "")),
                user_color=user_color,
                user_result=_outcome_for(user_color, result) if user_color else "",
            )
        )

    return summaries


def _read_game_at(pgn_text: str, index: int) -> chess.pgn.Game | None:
    pgn = StringIO(pgn_text)
    current = 0
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            return None
        current += 1
        if current == index:
            return game


_ARROW_BORDER_COLOR = "#000000"

PIECE_COLORS = {
    chess.PAWN: "#2ecc71",    # Bright Green
    chess.KNIGHT: "#3498db",  # Bright Blue
    chess.BISHOP: "#9b59b6",  # Amethyst Purple
    chess.ROOK: "#e74c3c",    # Alizarin Red
    chess.QUEEN: "#f1c40f",   # Sunflower Yellow
    chess.KING: "#1abc9c",    # Turquoise
}

PIECE_POINTS = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

# MVV-LVA uses the same relative values as the material heuristic.  The
# multiplier makes the victim value dominate the attacker's value: winning a
# queen with a pawn should be searched before winning a pawn with a queen.
MVV_LVA_VICTIM_MULTIPLIER = 10


def mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    """Score a legal move for capture ordering using MVV-LVA.

    Captures receive ``victim * 10 - attacker``; quiet moves score zero.
    En-passant captures are treated as pawn captures even though the victim
    is not physically on the destination square.  Higher scores are searched
    first.  This is a move-ordering heuristic, not a claim that the capture
    is tactically sound.
    """
    if not board.is_capture(move):
        return 0

    attacker = board.piece_at(move.from_square)
    if attacker is None:
        return 0

    victim = board.piece_at(move.to_square)
    if victim is None and board.is_en_passant(move):
        victim_type = chess.PAWN
    elif victim is None:
        # Defensive fallback for malformed/non-legal moves.
        return 0
    else:
        victim_type = victim.piece_type

    return (
        MVV_LVA_VICTIM_MULTIPLIER * PIECE_POINTS[victim_type]
        - PIECE_POINTS[attacker.piece_type]
    )


def order_moves_mvv_lva(board: chess.Board, moves=None) -> list[chess.Move]:
    """Return legal/candidate moves ordered from highest to lowest MVV-LVA."""
    candidates = list(board.legal_moves if moves is None else moves)
    return sorted(candidates, key=lambda move: mvv_lva_score(board, move), reverse=True)

LEGAL_MOVES_WEIGHT:float = 1.0
MATERIAL_SCORE_WEIGHT:float = 2.0
FORWARD_SCORE_WEIGHT:float = 2.0
FORWARD_MATERIAL_SCORE_WEIGHT: float = 3.0
CENTER_CONTROL_WEIGHT:float = 2.0
PST_SCORE_WEIGHT: float = 0.01
# Weight for the "goal is checkmate" heuristic: how hard the engine leans on
# driving the enemy king to the edge and cutting off its escape squares. Kept
# small relative to material, so it only breaks ties between otherwise-similar
# moves rather than sacrificing material to chase the king.
CHECKMATE_WEIGHT:float = 1.0

PHASE_MULTIPLIERS = {
    "Opening": {"legal": LEGAL_MOVES_WEIGHT * 0.90, "material": MATERIAL_SCORE_WEIGHT, "forward": FORWARD_SCORE_WEIGHT, "control": 1.80, "checkmate": CHECKMATE_WEIGHT},
    "Middlegame": {"legal": LEGAL_MOVES_WEIGHT, "material": MATERIAL_SCORE_WEIGHT * 1.2, "forward": FORWARD_SCORE_WEIGHT, "control": 1.00, "checkmate": CHECKMATE_WEIGHT},
    "Endgame": {"legal": LEGAL_MOVES_WEIGHT * 1.1, "material": MATERIAL_SCORE_WEIGHT * 1.45, "forward": FORWARD_SCORE_WEIGHT * 1.15, "control": 0.75, "checkmate":CHECKMATE_WEIGHT},
}


def _pst(rows: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    """Flatten rank-major PST rows into python-chess square order (a1..h8)."""
    return tuple(value for rank in reversed(rows) for value in rank)


# Values are centipawns, with rank 1 first in the source rows.  They are
# intentionally modest: PSTs should refine material/mobility decisions, not
# make the evaluator sacrifice a piece for a pretty square.
PST_MIDDLEGAME = {
    chess.PAWN: _pst(((0, 0, 0, 0, 0, 0, 0, 0), (5, 10, 10, -20, -20, 10, 10, 5),
                      (5, -5, -10, 0, 0, -10, -5, 5), (0, 0, 0, 20, 20, 0, 0, 0),
                      (5, 5, 10, 25, 25, 10, 5, 5), (10, 10, 20, 30, 30, 20, 10, 10),
                      (50, 50, 50, 50, 50, 50, 50, 50), (0, 0, 0, 0, 0, 0, 0, 0))),
    chess.KNIGHT: _pst(((-50, -40, -30, -30, -30, -30, -40, -50), (-40, -20, 0, 5, 5, 0, -20, -40),
                        (-30, 5, 10, 15, 15, 10, 5, -30), (-30, 0, 15, 20, 20, 15, 0, -30),
                        (-30, 5, 15, 20, 20, 15, 5, -30), (-30, 0, 10, 15, 15, 10, 0, -30),
                        (-40, -20, 0, 0, 0, 0, -20, -40), (-50, -40, -30, -30, -30, -30, -40, -50))),
    chess.BISHOP: _pst(((-20, -10, -10, -10, -10, -10, -10, -20), (-10, 5, 0, 0, 0, 0, 5, -10),
                        (-10, 10, 10, 10, 10, 10, 10, -10), (-10, 0, 10, 10, 10, 10, 0, -10),
                        (-10, 5, 5, 10, 10, 5, 5, -10), (-10, 0, 5, 10, 10, 5, 0, -10),
                        (-10, 0, 0, 0, 0, 0, 0, -10), (-20, -10, -10, -10, -10, -10, -10, -20))),
    chess.ROOK: _pst(((0, 0, 0, 5, 5, 0, 0, 0), (-5, 0, 0, 0, 0, 0, 0, -5),
                      (-5, 0, 0, 0, 0, 0, 0, -5), (-5, 0, 0, 0, 0, 0, 0, -5),
                      (-5, 0, 0, 0, 0, 0, 0, -5), (-5, 0, 0, 0, 0, 0, 0, -5),
                      (5, 10, 10, 10, 10, 10, 10, 5), (0, 0, 0, 0, 0, 0, 0, 0))),
    chess.QUEEN: _pst(((-20, -10, -10, 0, 0, -10, -10, -20), (-10, 0, 5, 0, 0, 0, 0, -10),
                       (-10, 5, 5, 5, 5, 5, 0, -10), (0, 0, 5, 5, 5, 5, 0, -5),
                       (-5, 0, 5, 5, 5, 5, 0, -5), (-10, 0, 5, 5, 5, 5, 0, -10),
                       (-10, 0, 0, 0, 0, 0, 0, -10), (-20, -10, -10, 0, 0, -10, -10, -20))),
    chess.KING: _pst(((-30, -40, -40, -50, -50, -40, -40, -30), (-30, -40, -40, -50, -50, -40, -40, -30),
                      (-30, -40, -40, -50, -50, -40, -40, -30), (-30, -40, -40, -50, -50, -40, -40, -30),
                      (-20, -30, -30, -40, -40, -30, -30, -20), (-10, -20, -20, -20, -20, -20, -20, -10),
                      (20, 20, 0, 0, 0, 0, 20, 20), (20, 30, 10, 0, 0, 10, 30, 20))),
}

PST_ENDGAME = {
    piece_type: values for piece_type, values in PST_MIDDLEGAME.items()
}
PST_ENDGAME[chess.PAWN] = _pst(((0, 0, 0, 0, 0, 0, 0, 0), (10, 10, 10, 10, 10, 10, 10, 10),
                                (10, 10, 10, 10, 10, 10, 10, 10), (20, 20, 20, 20, 20, 20, 20, 20),
                                (30, 30, 30, 30, 30, 30, 30, 30), (40, 40, 40, 40, 40, 40, 40, 40),
                                (60, 60, 60, 60, 60, 60, 60, 60), (0, 0, 0, 0, 0, 0, 0, 0)))
PST_ENDGAME[chess.KING] = _pst(((-50, -30, -30, -30, -30, -30, -30, -50), (-30, -10, 0, 0, 0, 0, -10, -30),
                                (-30, 0, 20, 30, 30, 20, 0, -30), (-30, 0, 30, 40, 40, 30, 0, -30),
                                (-30, 0, 30, 40, 40, 30, 0, -30), (-30, 0, 20, 30, 30, 20, 0, -30),
                                (-30, -10, 0, 0, 0, 0, -10, -30), (-50, -30, -30, -30, -30, -30, -30, -50)))


def _piece_square_score(board: chess.Board, phase: float | None = None) -> float:
    """Return a tapered PST score from White's perspective, in pawn units."""
    phase = _game_phase_value(board) if phase is None else max(0.0, min(1.0, phase))
    score = 0.0
    for square, piece in board.piece_map().items():
        table_square = square if piece.color == chess.WHITE else chess.square(
            chess.square_file(square), 7 - chess.square_rank(square)
        )
        middle = PST_MIDDLEGAME[piece.piece_type][table_square]
        end = PST_ENDGAME[piece.piece_type][table_square]
        value = (middle + phase * (end - middle)) / 100.0
        score += value if piece.color == chess.WHITE else -value
    return round(score, 3)


def _game_phase_value(board: chess.Board) -> float:
    """Return 0 for the opening and 1 for a simplified endgame.

    Non-pawn material is used so an opening pawn sacrifice does not
    prematurely turn the position into an endgame.
    """
    return float(chess_engine.calculate_phase_value(board.fen()))


def _phase_name(value: float) -> str:
    if value < 0.30:
        return "Opening"
    if value < 0.68:
        return "Middlegame"
    return "Endgame"


def _phase_multipliers(value: float) -> dict[str, float]:
    if value < 0.30:
        return PHASE_MULTIPLIERS["Opening"]
    if value < 0.68:
        return PHASE_MULTIPLIERS["Middlegame"]
    return PHASE_MULTIPLIERS["Endgame"]


def _phase_weight_percentages(value: float) -> dict[str, int]:
    multipliers = _phase_multipliers(value)
    effective = {
        "Material": abs(MATERIAL_SCORE_WEIGHT * multipliers["material"]),
        "Legal Moves": abs(LEGAL_MOVES_WEIGHT * multipliers["legal"]),
        "Center Control": abs(CENTER_CONTROL_WEIGHT * multipliers["control"]),
        "Forward Score": abs(FORWARD_SCORE_WEIGHT * multipliers["forward"]),
    }
    total = sum(effective.values())
    percentages = {name: round(weight / total * 100) for name, weight in effective.items()}
    # Keep the displayed values summing to exactly 100 after rounding.
    percentages["Forward Score"] += 100 - sum(percentages.values())
    return percentages


def _style_arrows(svg: str) -> str:
    """Make the overlaid legal-move arrows smaller and add a border.

    ``python-chess`` renders each arrow as a ``<line class="arrow">`` shaft and
    a ``<polygon class="arrow">`` head with no size/border options, so we
    post-process the generated SVG: the shaft is thinned and drawn on top of a
    slightly wider border underlay, and the arrowhead gets an outline.
    """

    def line_repl(match: re.Match[str]) -> str:
        tag = match.group(0)

        # Scale down the arrowhead
        scale = 0.35

        # We need to adjust x2, y2 so the line stops at the base of the smaller arrowhead.
        # And we want the tip to point to the middle of the square.
        # python-chess arrows have the tip offset by 0.1 * square_size (4.5 units)
        # from the center. Since stroke-width is 0.2 * square_size, the offset is
        # exactly 0.5 * stroke_width.

        width_match = re.search(r'stroke-width="([\d.]+)"', tag)
        if not width_match:
            return tag
        original_width = float(width_match.group(1))

        # Re-calculate the tip position based on the line end (x2, y2) and width.
        # python-chess: tip = (x2, y2) + unit_vector * (width * 3.75)
        # Wait, the example: line x1=217.5, y1=307.5, x2=217.5, y2=255.75.
        # Vector is (0, -51.75). Unit vector is (0, -1).
        # tip_y = 222.0. y2 = 255.75. tip_y - y2 = -33.75.
        # -33.75 / -1 = 33.75. 33.75 / 9.0 = 3.75. Correct.

        x1 = float(re.search(r'x1="([\d.-]+)"', tag).group(1))
        y1 = float(re.search(r'y1="([\d.-]+)"', tag).group(1))
        x2 = float(re.search(r'x2="([\d.-]+)"', tag).group(1))
        y2 = float(re.search(r'y2="([\d.-]+)"', tag).group(1))

        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2)**0.5
        if length > 0:
            ux = dx / length
            uy = dy / length
            tip_x = x2 + ux * original_width * 3.75
            tip_y = y2 + uy * original_width * 3.75

            # Move tip to the center of the square
            new_tip_x = tip_x + ux * original_width * 0.5
            new_tip_y = tip_y + uy * original_width * 0.5

            # New x2, y2 is scaled towards new_tip
            new_x2 = round(new_tip_x + (x2 - tip_x) * scale, 2)
            new_y2 = round(new_tip_y + (y2 - tip_y) * scale, 2)

            tag = re.sub(r'x2="[\d.-]+"', f'x2="{new_x2}"', tag)
            tag = re.sub(r'y2="[\d.-]+"', f'y2="{new_y2}"', tag)

        thin = round(original_width * 0.1, 2)
        border = max(0.5, round(thin * 0.4, 2))
        thin_tag = re.sub(r'stroke-width="[\d.]+"', f'stroke-width="{thin}"', tag)
        under_tag = re.sub(r'stroke-width="[\d.]+"',
                           f'stroke-width="{round(thin + 2 * border, 2)}"', tag)
        under_tag = re.sub(r'stroke="[^"]*"',
                           f'stroke="{_ARROW_BORDER_COLOR}"', under_tag)
        under_tag = re.sub(r'opacity="[^"]*"\s*', "", under_tag)
        
        # Ensure the thin line (shaft) is fully opaque so it doesn't blend with the black border underneath
        thin_tag = re.sub(r'opacity="[^"]*"\s*', "", thin_tag)
        # If it was an 8-digit hex, we should probably strip the alpha if we want 100% opacity
        thin_tag = re.sub(r'stroke="#([0-9a-fA-F]{6})[0-9a-fA-F]{2}"', r'stroke="#\1"', thin_tag)

        # Border underlay first so the coloured shaft is drawn on top of it.
        return under_tag + thin_tag

    def poly_repl(match: re.Match[str]) -> str:
        tag = match.group(0)

        # Scale down the arrowhead
        points_match = re.search(r'points="([\d.,\s]+)"', tag)
        if points_match:
            points_str = points_match.group(1)
            # Ensure the arrowhead is fully opaque
            tag = re.sub(r'opacity="[^"]*"\s*', "", tag)
            tag = re.sub(r'fill="#([0-9a-fA-F]{6})[0-9a-fA-F]{2}"', r'fill="#\1"', tag)
            try:
                # points="x1,y1 x2,y2 x3,y3"
                pts = [p.split(',') for p in points_str.split()]
                pts = [(float(p[0]), float(p[1])) for p in pts]

                # Scale factor: 0.35 means 35% of original size
                scale = 0.35
                # The first point is the tip of the arrow
                tip = pts[0]
                # The other points form the base
                m_x = (pts[1][0] + pts[2][0]) / 2
                m_y = (pts[1][1] + pts[2][1]) / 2
                
                # Calculate unit vector from base midpoint to tip
                d_x = tip[0] - m_x
                d_y = tip[1] - m_y
                dist = (d_x**2 + d_y**2)**0.5
                if dist > 0:
                    ux = d_x / dist
                    uy = d_y / dist
                    # python-chess offset is 0.5 * stroke-width. 
                    # stroke-width = dist / 3.75.
                    # so offset = dist / 7.5
                    offset = dist / 7.5
                    new_tip_x = tip[0] + ux * offset
                    new_tip_y = tip[1] + uy * offset
                else:
                    new_tip_x, new_tip_y = tip[0], tip[1]

                new_pts = []
                for p in pts:
                    new_x = round(new_tip_x + (p[0] - tip[0]) * scale, 2)
                    new_y = round(new_tip_y + (p[1] - tip[1]) * scale, 2)
                    new_pts.append(f"{new_x},{new_y}")

                new_points_str = " ".join(new_pts)
                tag = re.sub(r'points="[^"]*"', f'points="{new_points_str}"', tag)
            except (ValueError, IndexError):
                pass

        if "stroke=" in tag:
            return tag
        return (tag[:-2].rstrip()
                + f' stroke="{_ARROW_BORDER_COLOR}" stroke-width="0.5"'
                  ' stroke-linejoin="round"/>')

    svg = re.sub(r'<line\b[^>]*class="arrow"[^>]*/>', line_repl, svg)
    svg = re.sub(r'<polygon\b[^>]*class="arrow"[^>]*/>', poly_repl, svg)
    return svg


def _calculate_forward(board: chess.Board) -> tuple[dict[str, int], dict[str, int]]:
    """Calculate first- and second-order forward control in Rust."""
    f1, f2, _ = chess_engine.calculate_forward(board.fen())
    return (
        {"White": int(f1[0]), "Black": int(f1[1])},
        {"White": int(f2[0]), "Black": int(f2[1])},
    )


def _calculate_forward_3(board: chess.Board) -> dict[str, int]:
    """Calculate 3rd order forward: average own control two plies out.

    For each side's own legal move, averages the control after every one of
    the opponent's replies. This is O(N^2) in the branching factor, so unlike
    ``_calculate_forward`` it is not used on the self-play engine's hot move
    -selection path -- only for the game viewer's display, where it runs once
    per position instead of once per candidate move.
    """
    _, _, f3 = chess_engine.calculate_forward(board.fen())
    return {"White": int(f3[0]), "Black": int(f3[1])}


def _calculate_forward_4(board: chess.Board) -> dict[str, int]:
    """Calculate 4th-order forward control after three plies.

    This averages control after the side's move, the opponent's reply, and
    the side's next move. It is used only by the game viewer because the
    branching factor grows quickly at this depth.
    """
    white, black = chess_engine.calculate_forward_4(board.fen())
    return {"White": int(white), "Black": int(black)}


def _calculate_forward_material(board: chess.Board) -> dict[str, int]:
    """Return material value occupying each side's forward zone."""
    white, black = chess_engine.calculate_forward_material(board.fen())
    return {"White": int(white), "Black": int(black)}


def get_board_control(board: chess.Board) -> dict[str, int]:
    """Count squares attacked on the forward two ranks for each side.

    For White, this is ranks 2 and 3. For Black, this is ranks 7 and 6.
    Only checks the 16 relevant squares instead of all 64.
    """
    white_control = 0
    black_control = 0

    # White forward squares: d2, e2, f2, g2, h2, d3, e3, f3, g3, h3
    for sq in (chess.D2, chess.E2, chess.F2, chess.G2, chess.H2,
               chess.D3, chess.E3, chess.F3, chess.G3, chess.H3):
        if board.is_attacked_by(chess.WHITE, sq):
            white_control += 1

    # Black forward squares: d6, e6, f6, g6, h6, d7, e7, f7, g7, h7
    for sq in (chess.D6, chess.E6, chess.F6, chess.G6, chess.H6,
               chess.D7, chess.E7, chess.F7, chess.G7, chess.H7):
        if board.is_attacked_by(chess.BLACK, sq):
            black_control += 1

    return {"White": white_control, "Black": black_control}


def _calculate_material(board: chess.Board) -> dict[str, int]:
    """Count material points for each side."""
    white = 0
    black = 0
    for piece_type, points in PIECE_POINTS.items():
        white += len(board.pieces(piece_type, chess.WHITE)) * points
        black += len(board.pieces(piece_type, chess.BLACK)) * points
    return {"White": white, "Black": black}


MAX_STARTING_MATERIAL = (
    8 * PIECE_POINTS[chess.PAWN]
    + 2 * PIECE_POINTS[chess.KNIGHT]
    + 2 * PIECE_POINTS[chess.BISHOP]
    + 2 * PIECE_POINTS[chess.ROOK]
    + PIECE_POINTS[chess.QUEEN]
) # 39
MAX_TOTAL_MATERIAL = MAX_STARTING_MATERIAL * 2
MIN_AUTO_SEARCH_DEPTH = 2
# Keep the default native self-play profile comfortably within a 10-second
# single-process budget for a 100-ply game.  Callers can still request deeper
# endgame search explicitly through ``max_depth``.
MAX_AUTO_SEARCH_DEPTH = 2
# Exponent applied to the traded-material fraction before the exponential
# curve. Below 1, it front-loads the ramp so depth climbs early, well before
# the endgame, instead of hugging the minimum until material is nearly gone.
# 2 is equals 1 white and 1 black
AUTO_SEARCH_DEPTH_CURVE_EXPONENT = 0.45


def _auto_search_depth(
    board: chess.Board,
    game_id: str | int | None = None,
    *,
    max_depth: int = MAX_AUTO_SEARCH_DEPTH,
) -> int:
    """Derive negamax search depth, inversely proportional to material left.

    A full board (combined material 78, both sides at their starting value)
    has the largest branching factor and is the most expensive to search
    deeply, so it gets the shallowest depth; as material is traded off
    the board thins out (fewer legal replies per turn) and depth scales
    exponentially up to ``max_depth`` at material 0, where deeper search is both
    affordable and needed for endgame precision. The traded fraction is
    root-scaled (``AUTO_SEARCH_DEPTH_CURVE_EXPONENT``) so depth ramps up
    quickly as soon as trades start, rather than waiting until the endgame.

    ``max_depth`` replaces the default upper bound for this calculation. If
    it is below ``MIN_AUTO_SEARCH_DEPTH``, it becomes both the minimum and
    maximum so the requested cap is always respected.
    """
    return int(chess_engine.calculate_auto_search_depth(board.fen(), max_depth))


def _calculate_center_control(board: chess.Board) -> dict[str, int]:
    """Count control of the 4 central squares (d4, e4, d5, e5)."""
    white, black = chess_engine.calculate_center_control(board.fen())
    return {"White": int(white), "Black": int(black)}


def _calculate_flank_control(board: chess.Board) -> dict[str, int]:
    """Count control of the outer files where king-and-pawn endgames unfold."""
    white, black = chess_engine.calculate_flank_control(board.fen())
    return {"White": int(white), "Black": int(black)}


def _king_escape_squares(board: chess.Board, king_color: chess.Color) -> int:
    """Count squares the ``king_color`` king could flee to.

    A square counts only if it is not blocked by one of the king's own pieces
    and is not attacked by the opponent. Fewer escape squares means the king is
    closer to being mated, so this is the raw signal the mate heuristic wants to
    minimise for the side under pressure.
    """
    return int(chess_engine.calculate_king_escape_squares(board.fen(), king_color == chess.WHITE))


def _mate_pressure(board: chess.Board) -> float:
    """Positional pressure toward checkmate, from White's perspective.

    Positive favours White. For each king we reward its opponent for pushing it
    toward the board edge/corner and for stripping away its escape squares --
    the two conditions that precede a forced mate. This is what tells the
    self-play engine that *the goal is checkmate*, not merely a material lead:
    once ahead, it keeps herding the enemy king instead of shuffling.
    """
    return float(chess_engine.calculate_mate_pressure(board.fen()))


def _king_activity(board: chess.Board, phase: float) -> float:
    """Reward castling safety early and king centralisation late."""
    score = 0.0
    for color, sign in ((chess.WHITE, 1.0), (chess.BLACK, -1.0)):
        king_sq = board.king(color)
        if king_sq is None:
            continue
        file = chess.square_file(king_sq)
        rank = chess.square_rank(king_sq)
        if phase < 0.5:
            safety = 2.0 if file in (2, 6) and rank in (0, 7) else 0.0
            safety -= 1.0 if file == 4 and rank in (0, 7) else 0.0
            score += sign * safety * (1.0 - phase)
        else:
            centrality = 4.0 - abs(3.5 - file) - abs(3.5 - rank)
            score += sign * centrality * phase
    return score


def _color_mobility(board: chess.Board, color: chess.Color) -> int:
    """Count legal moves available to ``color``, regardless of whose turn it is.

    Uses the same turn-flip trick as ``_calculate_forward`` to get a
    color-specific move count out of python-chess.
    """
    original_turn = board.turn
    board.turn = color
    count = len(list(board.legal_moves))
    board.turn = original_turn
    return count


def _both_mobilities(board: chess.Board) -> tuple[int, int]:
    """Calculate legal move counts for both sides in one pass.

    Returns (white_count, black_count) without flipping turn back/forth twice.
    """
    original_turn = board.turn

    board.turn = chess.WHITE
    white_count = sum(1 for _ in board.legal_moves)

    board.turn = chess.BLACK
    black_count = sum(1 for _ in board.legal_moves)

    board.turn = original_turn
    return white_count, black_count


def _evaluate_position(
    board: chess.Board,
    *,
    legal_moves_weight: float = LEGAL_MOVES_WEIGHT,
    material_score_weight: float = MATERIAL_SCORE_WEIGHT,
    forward_score_weight: float = FORWARD_SCORE_WEIGHT,
    forward_material_score_weight: float = FORWARD_MATERIAL_SCORE_WEIGHT,
    center_control_weight: float = CENTER_CONTROL_WEIGHT,
    checkmate_weight: float = CHECKMATE_WEIGHT,
) -> float:
    """White-perspective static evaluation performed by the Rust engine.

    Blends material, first-order forward control, center control, and mobility -- each as a
    White-minus-Black differential so the value is well-defined at any node
    of a multi-ply search -- plus the "goal is checkmate" king-pressure term.
    Uses only ``get_board_control`` (not ``_calculate_forward``'s pricier
    second-order term, which itself generates a full ply of moves) since
    this runs at every leaf of the search tree.
    """
    native_score = float(
        chess_engine.evaluate_position(
            board.fen(),
            legal_moves_weight,
            material_score_weight,
            forward_score_weight,
            center_control_weight,
            checkmate_weight,
            forward_material_score_weight,
        )
    )
    # The native evaluator owns the hot search path. Add the inexpensive PST
    # refinement here for Python-side analyses (blunder scoring, RL, and the
    # position viewer), preserving the extension's existing ABI.
    return native_score + PST_SCORE_WEIGHT * _piece_square_score(board)


MATE_SCORE = 1_000_000.0


def _terminal_aware_evaluate(board: chess.Board) -> float:
    """Like ``_evaluate_position``, but scores checkmate/stalemate by their
    actual game value instead of raw material.

    ``_evaluate_position`` only looks at material/mobility, so on a
    checkmated or stalemated board it would score however the pieces happen
    to sit -- e.g. a stalemate reached from a materially lost position would
    still read as a big material deficit, when a draw is always better than
    losing. Used wherever a 1-ply lookahead (e.g. ``_blunder_score``) needs to
    tell a forced draw apart from an outright loss.
    """
    if board.is_checkmate():
        return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    return _evaluate_position(board)


# The static evaluator combines mobility, material, control, and king
# pressure, so its values are not literal pawn units. These bands are
# deliberately wider than centipawn thresholds: small positional differences
# should not be reported as serious errors.
INACCURACY_THRESHOLD = 5.0
MISTAKE_THRESHOLD = 10.0
BLUNDER_THRESHOLD = 20.0


def _blunder_score(board: chess.Board, move: chess.Move) -> float:
    """How much worse ``move`` is than the best move in the position.

    This is a shallow move-quality comparison, not a full engine search. For
    every legal candidate, evaluate the worst position the opponent can reach
    after their next reply, from the candidate mover's perspective. The score
    is the gap between the best candidate and ``move``. Comparing candidates
    is important: comparing a move with the position before it incorrectly
    treats a normal good reply by the opponent as a blunder by the mover.

    The one-ply model is intended to catch immediate tactical blunders such as
    hanging a piece; deeper tactics need a real search.
    Post-move positions are scored with ``_terminal_aware_evaluate`` so that
    a stalemate escape (a draw) is correctly valued above losing outright,
    and a reply that delivers checkmate is valued as an outright loss rather
    than whatever the material count happens to be.
    """
    mover = board.turn
    values: dict[chess.Move, float] = {}

    def candidate_value(candidate: chess.Move) -> float:
        if candidate in values:
            return values[candidate]
        board.push(candidate)
        try:
            replies = order_moves_mvv_lva(board)
            if not replies:
                value = _terminal_aware_evaluate(board)
                values[candidate] = -value if mover == chess.BLACK else value
                return values[candidate]

            worst_for_mover = math.inf
            for reply in replies:
                board.push(reply)
                try:
                    after = _terminal_aware_evaluate(board)
                finally:
                    board.pop()
                after_for_mover = -after if mover == chess.BLACK else after
                worst_for_mover = min(worst_for_mover, after_for_mover)
            values[candidate] = worst_for_mover
            return values[candidate]
        finally:
            board.pop()

    candidates = order_moves_mvv_lva(board)
    if not candidates:
        return 0.0

    best_value = max(candidate_value(candidate) for candidate in candidates)
    played_value = candidate_value(move)
    return round(max(0.0, best_value - played_value), 2)


def _move_severity(blunder_score: float) -> str:
    """Classify a move's eval swing into standard pawn-loss severity tiers."""
    if blunder_score >= BLUNDER_THRESHOLD:
        return "Blunder"
    if blunder_score >= MISTAKE_THRESHOLD:
        return "Mistake"
    if blunder_score >= INACCURACY_THRESHOLD:
        return "Inaccuracy"
    return ""


def choose_engine_move(
    board: chess.Board,
    rng: random.Random | None = None,
    top_k: int = 1,
    *,
    legal_moves_weight: float = LEGAL_MOVES_WEIGHT,
    material_score_weight: float = MATERIAL_SCORE_WEIGHT,
    forward_score_weight: float = FORWARD_SCORE_WEIGHT,
    forward_material_score_weight: float = FORWARD_MATERIAL_SCORE_WEIGHT,
    center_control_weight: float = CENTER_CONTROL_WEIGHT,
    checkmate_weight: float = CHECKMATE_WEIGHT,
    depth: int = 3,
    top_k_score_threshold: float | None = 3.0,
    blunder_control: float = 0.0,
    eval_counter: list[int] | None = None,
) -> tuple[chess.Move, float]:
    """Pick a move using the native Rust negamax implementation.

    ``eval_counter``, if given, is a single-element ``[count]`` list that
    accumulates one increment per leaf position statically evaluated during
    the search -- callers use this to report how many evaluations a move
    (or a whole game) cost.

    ``top_k_score_threshold`` limits random selection to moves whose score is
    no more than that amount below the best move. ``top_k`` remains the hard
    candidate-count cap. The default is 3.0; ``None`` enables unrestricted
    top-K selection.

    """
    rng = rng or random.Random()
    history_fens = []
    history_board = board.copy(stack=True)
    while True:
        history_fens.append(history_board.fen())
        if not history_board.move_stack:
            break
        history_board.pop()

    native_args = (
        board.fen(), history_fens, max(1, depth), max(1, top_k),
        rng.getrandbits(64), legal_moves_weight, material_score_weight,
        forward_score_weight, center_control_weight, checkmate_weight,
        None if top_k_score_threshold is None else max(0.0, top_k_score_threshold),
    )
    # Keep already-running processes compatible with an older installed
    # extension. The new argument is only needed when the feature is used;
    # omitting it preserves the old engine's exact default behaviour.
    if blunder_control or forward_material_score_weight != FORWARD_MATERIAL_SCORE_WEIGHT:
        try:
            uci, score, evaluations = chess_engine.choose_engine_move_from_history(
                *native_args, max(0.0, min(1.0, blunder_control)),
                forward_material_score_weight,
            )
        except TypeError as exc:
            if "positional arguments" not in str(exc):
                raise
            # An older loaded extension cannot apply the control. Keep the
            # game playable until the process is restarted with the rebuilt
            # wheel; the requested control is applied by the new extension.
            uci, score, evaluations = chess_engine.choose_engine_move_from_history(*native_args)
    else:
        uci, score, evaluations = chess_engine.choose_engine_move_from_history(*native_args)
    if eval_counter is not None:
        eval_counter[0] += int(evaluations)
    return chess.Move.from_uci(uci), float(score)


def play_self_game_native(
    fen: str,
    max_turns: int,
    top_k: int,
    seed: int,
    *,
    legal_moves_weight: float = LEGAL_MOVES_WEIGHT,
    material_score_weight: float = MATERIAL_SCORE_WEIGHT,
    forward_score_weight: float = FORWARD_SCORE_WEIGHT,
    forward_material_score_weight: float = FORWARD_MATERIAL_SCORE_WEIGHT,
    center_control_weight: float = CENTER_CONTROL_WEIGHT,
    black_legal_moves_weight: float | None = None,
    black_material_score_weight: float | None = None,
    black_forward_score_weight: float | None = None,
    black_forward_material_score_weight: float | None = None,
    black_center_control_weight: float | None = None,
    checkmate_weight: float = CHECKMATE_WEIGHT,
    depth: int | None = None,
    max_depth: int = MAX_AUTO_SEARCH_DEPTH,
    top_k_score_threshold: float | None = 3.0,
    blunder_control: float = 0.0,
) -> tuple[str, str, int, list[str], int, list[float]]:
    """Run the complete self-play move loop in the native engine."""
    black_legal_moves_weight = legal_moves_weight if black_legal_moves_weight is None else black_legal_moves_weight
    black_material_score_weight = material_score_weight if black_material_score_weight is None else black_material_score_weight
    black_forward_score_weight = forward_score_weight if black_forward_score_weight is None else black_forward_score_weight
    black_forward_material_score_weight = forward_material_score_weight if black_forward_material_score_weight is None else black_forward_material_score_weight
    black_center_control_weight = center_control_weight if black_center_control_weight is None else black_center_control_weight
    native_args = (
        fen, max_turns, top_k, seed, legal_moves_weight,
        material_score_weight, forward_score_weight, center_control_weight,
        black_legal_moves_weight, black_material_score_weight,
        black_forward_score_weight, black_center_control_weight,
        checkmate_weight, depth, max_depth, top_k_score_threshold,
    )
    if blunder_control or forward_material_score_weight != FORWARD_MATERIAL_SCORE_WEIGHT or black_forward_material_score_weight != FORWARD_MATERIAL_SCORE_WEIGHT:
        try:
            return chess_engine.play_self_game_native(
                *native_args, max(0.0, min(1.0, blunder_control)),
                forward_material_score_weight, black_forward_material_score_weight,
            )
        except TypeError as exc:
            if "positional arguments" not in str(exc):
                raise
            # Compatibility with a server that has not yet reloaded the
            # rebuilt extension. The game completes, but blunders remain off.
            return chess_engine.play_self_game_native(*native_args)
    return chess_engine.play_self_game_native(*native_args)


def _calculate_total_score(
    mobility_score: int,
    material_score: int,
    forward_score: int,
    center_score: float = 0.0,
    pressure: float = 0.0,
    pst_score: float = 0.0,
    *,
    legal_moves_weight: float = LEGAL_MOVES_WEIGHT,
    material_score_weight: float = MATERIAL_SCORE_WEIGHT,
    forward_score_weight: float = FORWARD_SCORE_WEIGHT,
    center_control_weight: float = CENTER_CONTROL_WEIGHT,
    checkmate_weight: float = CHECKMATE_WEIGHT,
    phase: float | None = None,
) -> float:
    """Blend mobility, material, forward, center control, and pressure into one position score.

    Formula:
        total_score = w1 * mobility + w2 * material + w3 * forward + w4 * center + w5 * pressure

    The weights keep material as the strongest signal, while still letting
    other factors move the score in a visible way.
    """
    if phase is not None:
        multipliers = _phase_multipliers(phase)
        legal_moves_weight *= multipliers["legal"]
        material_score_weight *= multipliers["material"]
        forward_score_weight *= multipliers["forward"]
        center_control_weight *= multipliers["control"]
        checkmate_weight *= multipliers["checkmate"]
    return round(legal_moves_weight * mobility_score
        + material_score_weight * material_score
        + forward_score_weight * forward_score
        + center_control_weight * center_score
        + PST_SCORE_WEIGHT * pst_score
        + checkmate_weight * pressure, 2)


def _score_breakdown(
    mobility_score: float,
    material_score: float,
    forward_score: float,
    center_score: float,
    mate_pressure: float,
    king_activity: float,
    pst_score: float,
    phase: float | None = None,
) -> list[dict[str, str | float]]:
    """Return the raw score, effective weight, and contribution per component."""
    weights = {
        "Mobility": LEGAL_MOVES_WEIGHT,
        "Material": MATERIAL_SCORE_WEIGHT,
        "Forward": FORWARD_SCORE_WEIGHT,
        "Center": CENTER_CONTROL_WEIGHT,
        "Mate pressure": CHECKMATE_WEIGHT,
        "King activity": CHECKMATE_WEIGHT,
        "PST": PST_SCORE_WEIGHT,
    }
    if phase is not None:
        multipliers = _phase_multipliers(phase)
        weights["Mobility"] *= multipliers["legal"] / LEGAL_MOVES_WEIGHT
        weights["Material"] *= multipliers["material"] / MATERIAL_SCORE_WEIGHT
        weights["Forward"] *= multipliers["forward"] / FORWARD_SCORE_WEIGHT
        weights["Center"] *= multipliers["control"] / CENTER_CONTROL_WEIGHT
        weights["Mate pressure"] *= multipliers["checkmate"] / CHECKMATE_WEIGHT
        weights["King activity"] *= multipliers["checkmate"] / CHECKMATE_WEIGHT

    scores = {
        "Mobility": mobility_score,
        "Material": material_score,
        "Forward": forward_score,
        "Center": center_score,
        "Mate pressure": mate_pressure,
        "King activity": king_activity,
        "PST": pst_score,
    }
    return [
        {
            "component": component,
            "score": round(scores[component], 3),
            "weight": round(weights[component], 3),
            "weighted_score": round(scores[component] * weights[component], 3),
        }
        for component in scores
    ]


def _legal_move_arrows(board: chess.Board) -> list[chess.svg.Arrow]:
    """One color-by-piece arrow per legal move, for the "show all valid moves" board style."""
    arrows = []
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        color_hex = PIECE_COLORS.get(piece.piece_type, "#15781B") if piece else "#15781B"
        arrows.append(chess.svg.Arrow(move.from_square, move.to_square, color=color_hex))
    return arrows


def render_board_svgs(board: chess.Board, lastmove: chess.Move | None = None) -> tuple[str, str]:
    """Render a board's plain SVG and its legal-move-arrows SVG.

    Shared by the game viewer and the play page so every board in the app
    uses the same visual style (piece-colored, thin-bordered arrows).
    """
    svg = chess.svg.board(board, size=420, lastmove=lastmove)
    svg_moves = _style_arrows(
        chess.svg.board(board, size=420, lastmove=lastmove, arrows=_legal_move_arrows(board))
    )
    return svg, svg_moves


def _native_legal_move_data(
    board: chess.Board,
) -> tuple[list[str], dict[str, list[str]], dict[str, int]]:
    """Return legal SAN moves, their reply tree, and native move scores."""
    legal_moves, tree, move_scores = chess_engine.legal_moves_and_tree(board.fen())
    return list(legal_moves), dict(tree), dict(move_scores)


def _position_metrics(board: chess.Board) -> tuple:
    """Calculate the non-rendering metrics displayed for one position."""
    f1, f2 = _calculate_forward(board)
    f3 = _calculate_forward_3(board)
    f4 = _calculate_forward_4(board)
    material = _calculate_material(board)
    center = _calculate_center_control(board)
    flank = _calculate_flank_control(board)
    phase_value = _game_phase_value(board)
    phase = _phase_name(phase_value)
    weight_percentages = _phase_weight_percentages(phase_value)
    forward_score = (f1["White"] + f2["White"]) - (f1["Black"] + f2["Black"])
    material_score = material["White"] - material["Black"]
    center_score = center["White"] - center["Black"]
    flank_score = flank["White"] - flank["Black"]
    strategic_control_score = round((1.0 - phase_value) * center_score + phase_value * flank_score, 2)
    mobility_w, mobility_b = _both_mobilities(board)
    mobility_score = mobility_w - mobility_b
    mate_pressure = _mate_pressure(board)
    king_activity = _king_activity(board, phase_value)
    pressure = mate_pressure + king_activity
    pst_score = _piece_square_score(board, phase_value)
    score_breakdown = _score_breakdown(
        mobility_score, material_score, forward_score,
        strategic_control_score, mate_pressure, king_activity,
        pst_score, phase_value,
    )
    total_score = _calculate_total_score(
        mobility_score, material_score, forward_score, strategic_control_score, pressure,
        pst_score,
        phase=phase_value,
    )
    return (
        f1, f2, f3, f4, material, center, flank,
        forward_score, material_score, strategic_control_score,
        total_score, score_breakdown, phase, weight_percentages,
    )


def _legal_moves_and_tree(board: chess.Board, lastmove: chess.Move | None = None) -> tuple:
    """Render a position and combine native move data with viewer metrics."""
    legal_moves, tree, move_scores = _native_legal_move_data(board)
    arrows = _legal_move_arrows(board)

    (
        f1, f2, f3, f4, material, center, flank,
        forward_score, material_score, strategic_control_score,
        total_score, score_breakdown, phase, weight_percentages,
    ) = _position_metrics(board)
    score = len(legal_moves)

    sans = list(legal_moves)
    svg = chess.svg.board(board, size=420, lastmove=lastmove, arrows=arrows)
    return (_style_arrows(svg), sans, tree, f1, f2, f3, f4, material, center, flank,
            forward_score, material_score, strategic_control_score, score,
            total_score, score_breakdown, move_scores, phase, weight_percentages)


def load_game_detail(pgn_text: str, index: int) -> GameDetail | None:
    """Parse a single game and render every board position as SVG."""
    game = _read_game_at(pgn_text, index)
    if game is None:
        return None

    headers = game.headers
    board = game.board()

    start_moves_svg, start_legal, start_tree, start_f1, start_f2, start_f3, start_f4, start_material, start_center, start_flank, start_forward_score, start_material_score, start_center_score, start_score, start_total_score, start_score_breakdown, start_scores, start_phase, start_weight_percentages = _legal_moves_and_tree(board)
    positions: list[GamePosition] = [
        GamePosition(
            ply=0,
            move_number=0,
            san="",
            side="",
            fen=board.fen(),
            svg=chess.svg.board(board, size=420),
            svg_moves=start_moves_svg,
            legal_moves=start_legal,
            move_tree=start_tree,
            move_scores=start_scores,
            forward_1=start_f1,
            forward_2=start_f2,
            forward_3=start_f3,
            forward_4=start_f4,
            material=start_material,
            center=start_center,
            flank=start_flank,
            phase=start_phase,
            weight_percentages=start_weight_percentages,
            forward_score=start_forward_score,
            material_score=start_material_score,
            center_score=start_center_score,
            score=start_score,
            total_score=start_total_score,
            score_breakdown=start_score_breakdown,
            blunder_score=0.0,
            severity="",
        )
    ]

    ply = 0
    for move in game.mainline_moves():
        ply += 1
        side = "White" if board.turn == chess.WHITE else "Black"
        move_number = board.fullmove_number
        san = board.san(move)
        blunder_score = _blunder_score(board, move)
        board.push(move)
        moves_svg, legal, tree, f1, f2, f3, f4, material, center, flank, forward_score, material_score, center_score, score, total_score, score_breakdown, scores, phase, weight_percentages = _legal_moves_and_tree(board, lastmove=move)
        positions.append(
            GamePosition(
                ply=ply,
                move_number=move_number,
                san=san,
                side=side,
                fen=board.fen(),
                svg=chess.svg.board(board, size=420, lastmove=move),
                svg_moves=moves_svg,
                legal_moves=legal,
                move_tree=tree,
                move_scores=scores,
                forward_1=f1,
                forward_2=f2,
                forward_3=f3,
                forward_4=f4,
                material=material,
                center=center,
                flank=flank,
                phase=phase,
                weight_percentages=weight_percentages,
                forward_score=forward_score,
                material_score=material_score,
                center_score=center_score,
                score=score,
                total_score=total_score,
                score_breakdown=score_breakdown,
                blunder_score=blunder_score,
                severity=_move_severity(blunder_score),
            )
        )

    return GameDetail(
        index=index,
        white=headers.get("White", ""),
        black=headers.get("Black", ""),
        result=headers.get("Result", "*"),
        date=headers.get("UTCDate", headers.get("Date", "")),
        time_control=headers.get("TimeControl", ""),
        eco=headers.get("ECO", ""),
        eco_name=eco_name(headers.get("ECO", "")),
        positions=positions,
    )
