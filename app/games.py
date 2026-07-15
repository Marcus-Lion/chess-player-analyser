from __future__ import annotations

import math
import re
import random
from dataclasses import dataclass
from io import StringIO

import chess
import chess.pgn
import chess.svg

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
    material: dict[str, int]  # {"White": points, "Black": points}
    center: dict[str, int]  # {"White": count, "Black": count}
    forward_score: int  # (W_f1 + W_f2) - (B_f1 + B_f2)
    material_score: int  # White material - Black material
    center_score: int  # White center - Black center
    score: int  # Legal move count for the side to move
    total_score: float  # Weighted blend of legal moves, material, forward, and center
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

LEGAL_MOVES_WEIGHT:float = 0.3
MATERIAL_SCORE_WEIGHT:float = 0.35
FORWARD_SCORE_WEIGHT:float = 0.20
CENTER_CONTROL_WEIGHT:float = 0.125
# Weight for the "goal is checkmate" heuristic: how hard the engine leans on
# driving the enemy king to the edge and cutting off its escape squares. Kept
# small relative to material so it only breaks ties between otherwise-similar
# moves rather than sacrificing material to chase the king.
CHECKMATE_WEIGHT:float = 1.0


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
    """Calculate 1st and 2nd order forward on the forward two ranks."""
    f1 = get_board_control(board)
    
    f2 = {"White": 0, "Black": 0}
    original_turn = board.turn
    
    # White 2nd order
    board.turn = chess.WHITE
    w_moves = list(board.legal_moves)
    if w_moves:
        total_w_c1 = 0
        for m in w_moves:
            board.push(m)
            total_w_c1 += get_board_control(board)["White"]
            board.pop()
        f2["White"] = int(total_w_c1 / len(w_moves))
    else:
        f2["White"] = f1["White"]

    # Black 2nd order
    board.turn = chess.BLACK
    b_moves = list(board.legal_moves)
    if b_moves:
        total_b_c1 = 0
        for m in b_moves:
            board.push(m)
            total_b_c1 += get_board_control(board)["Black"]
            board.pop()
        f2["Black"] = int(total_b_c1 / len(b_moves))
    else:
        f2["Black"] = f1["Black"]
        
    board.turn = original_turn
    return f1, f2


def _calculate_forward_3(board: chess.Board) -> dict[str, int]:
    """Calculate 3rd order forward: average own control two plies out.

    For each side's own legal move, averages the control after every one of
    the opponent's replies. This is O(N^2) in the branching factor, so unlike
    ``_calculate_forward`` it is not used on the self-play engine's hot move
    -selection path -- only for the game viewer's display, where it runs once
    per position instead of once per candidate move.
    """
    f3 = {"White": 0, "Black": 0}
    original_turn = board.turn

    for color, key in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        board.turn = color
        own_moves = list(board.legal_moves)
        total = 0
        count = 0
        for own_move in own_moves:
            board.push(own_move)
            reply_moves = list(board.legal_moves)
            if reply_moves:
                for reply in reply_moves:
                    board.push(reply)
                    total += get_board_control(board)[key]
                    count += 1
                    board.pop()
            else:
                total += get_board_control(board)[key]
                count += 1
            board.pop()
        f3[key] = int(total / count) if count else get_board_control(board)[key]

    board.turn = original_turn
    return f3


def get_board_control(board: chess.Board) -> dict[str, int]:
    """Count squares attacked on the forward two ranks for each side.

    For White, this is ranks 2 and 3. For Black, this is ranks 7 and 6.
    """
    forward_ranks = {
        chess.WHITE: {1, 2},
        chess.BLACK: {5, 6},
    }

    return {
        "White": sum(
            1
            for sq in chess.SQUARES
            if chess.square_rank(sq) in forward_ranks[chess.WHITE] and board.is_attacked_by(chess.WHITE, sq)
        ),
        "Black": sum(
            1
            for sq in chess.SQUARES
            if chess.square_rank(sq) in forward_ranks[chess.BLACK] and board.is_attacked_by(chess.BLACK, sq)
        ),
    }


def _calculate_material(board: chess.Board) -> dict[str, int]:
    """Count material points for each side."""
    white = 0
    black = 0
    for piece_type, points in PIECE_POINTS.items():
        white += len(board.pieces(piece_type, chess.WHITE)) * points
        black += len(board.pieces(piece_type, chess.BLACK)) * points
    return {"White": white, "Black": black}


MAX_STARTING_MATERIAL = 39
MAX_TOTAL_MATERIAL = MAX_STARTING_MATERIAL * 2
MIN_AUTO_SEARCH_DEPTH = 1
MAX_AUTO_SEARCH_DEPTH = 6
# Exponent applied to the traded-material fraction before the exponential
# curve. Below 1, it front-loads the ramp so depth climbs early, well before
# the endgame, instead of hugging the minimum until material is nearly gone.
AUTO_SEARCH_DEPTH_CURVE_EXPONENT = 0.55


def _auto_search_depth(board: chess.Board) -> int:
    """Derive negamax search depth, inversely proportional to material left.

    A full board (combined material 78, both sides at their starting value)
    has the largest branching factor and is the most expensive to search
    deeply, so it gets the shallowest depth (1); as material is traded off
    the board thins out (fewer legal replies per turn) and depth scales
    exponentially up to 6 at material 0, where deeper search is both
    affordable and needed for endgame precision. The traded fraction is
    root-scaled (``AUTO_SEARCH_DEPTH_CURVE_EXPONENT``) so depth ramps up
    quickly as soon as trades start, rather than waiting until the endgame.
    """
    material = _calculate_material(board)
    remaining = material["White"] + material["Black"]
    remaining = max(0, min(MAX_TOTAL_MATERIAL, remaining))
    fraction_traded = (MAX_TOTAL_MATERIAL - remaining) / MAX_TOTAL_MATERIAL
    scaled_fraction = fraction_traded ** AUTO_SEARCH_DEPTH_CURVE_EXPONENT
    depth_ratio = MAX_AUTO_SEARCH_DEPTH / MIN_AUTO_SEARCH_DEPTH
    depth = MIN_AUTO_SEARCH_DEPTH * depth_ratio ** scaled_fraction
    depth_int = max(MIN_AUTO_SEARCH_DEPTH, min(MAX_AUTO_SEARCH_DEPTH, round(depth)))
    print(f"{board.fullmove_number}. material: {material} -> depth: {round(depth,2)} -> {depth_int}")
    return depth_int


def _calculate_center_control(board: chess.Board) -> dict[str, int]:
    """Count control of the 4 central squares (d4, e4, d5, e5)."""
    center_squares = {chess.D4, chess.E4, chess.D5, chess.E5}
    white_control = sum(1 for sq in center_squares if board.is_attacked_by(chess.WHITE, sq))
    black_control = sum(1 for sq in center_squares if board.is_attacked_by(chess.BLACK, sq))
    return {"White": white_control, "Black": black_control}


def _king_escape_squares(board: chess.Board, king_color: chess.Color) -> int:
    """Count squares the ``king_color`` king could flee to.

    A square counts only if it is not blocked by one of the king's own pieces
    and is not attacked by the opponent. Fewer escape squares means the king is
    closer to being mated, so this is the raw signal the mate heuristic wants to
    minimise for the side under pressure.
    """
    king_sq = board.king(king_color)
    if king_sq is None:
        return 0
    enemy = not king_color
    escapes = 0
    for square in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]):
        occupant = board.piece_at(square)
        if occupant is not None and occupant.color == king_color:
            continue
        if board.is_attacked_by(enemy, square):
            continue
        escapes += 1
    return escapes


def _mate_pressure(board: chess.Board) -> float:
    """Positional pressure toward checkmate, from White's perspective.

    Positive favours White. For each king we reward its opponent for pushing it
    toward the board edge/corner and for stripping away its escape squares --
    the two conditions that precede a forced mate. This is what tells the
    self-play engine that *the goal is checkmate*, not merely a material lead:
    once ahead, it keeps herding the enemy king instead of shuffling.
    """
    pressure = 0.0
    for defender, sign in ((chess.BLACK, 1.0), (chess.WHITE, -1.0)):
        king_sq = board.king(defender)
        if king_sq is None:
            continue
        file = chess.square_file(king_sq)
        rank = chess.square_rank(king_sq)
        # 2 in the centre, up to 14 in a corner: the further out, the better
        # for the attacker.
        corner_proximity = abs(2 * file - 7) + abs(2 * rank - 7)
        escapes = _king_escape_squares(board, defender)
        pressure += sign * (corner_proximity - 2.0 * escapes)
    return pressure


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


def _evaluate_position(
    board: chess.Board,
    *,
    legal_moves_weight: float = LEGAL_MOVES_WEIGHT,
    material_score_weight: float = MATERIAL_SCORE_WEIGHT,
    forward_score_weight: float = FORWARD_SCORE_WEIGHT,
    center_control_weight: float = CENTER_CONTROL_WEIGHT,
    checkmate_weight: float = CHECKMATE_WEIGHT,
) -> float:
    """White-perspective static evaluation used at search leaves.

    Blends material, first-order forward control, center control, and mobility -- each as a
    White-minus-Black differential so the value is well-defined at any node
    of a multi-ply search -- plus the "goal is checkmate" king-pressure term.
    Uses only ``get_board_control`` (not ``_calculate_forward``'s pricier
    second-order term, which itself generates a full ply of moves) since
    this runs at every leaf of the search tree.
    """
    material = _calculate_material(board)
    control = get_board_control(board)
    center = _calculate_center_control(board)
    mobility_score = _color_mobility(board, chess.WHITE) - _color_mobility(board, chess.BLACK)
    material_score = material["White"] - material["Black"]
    forward_score = control["White"] - control["Black"]
    center_score = center["White"] - center["Black"]
    return round(
        legal_moves_weight * mobility_score
        + material_score_weight * material_score
        + forward_score_weight * forward_score
        + center_control_weight * center_score
        + checkmate_weight * _mate_pressure(board),
        2,
    )


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


# Severity tiers for how much of a move's evaluation swing (in eval points,
# lost to the opponent's best reply) is "unjustified" -- mirroring the
# inaccuracy/mistake/blunder tiers chess sites use, from a slight edge given
# up (Inaccuracy) through a real threat (Mistake) to roughly "lost a piece
# for a pawn" (Blunder). material_score_weight scales evaluation points per
# point of material, so each *_MATERIAL_PAWNS constant is that tier's cutoff
# in pawns of unjustified swing.
INACCURACY_MATERIAL_PAWNS = 1.0
MISTAKE_MATERIAL_PAWNS = 1.75
BLUNDER_MATERIAL_PAWNS = 2.5
INACCURACY_THRESHOLD = MATERIAL_SCORE_WEIGHT * INACCURACY_MATERIAL_PAWNS
MISTAKE_THRESHOLD = MATERIAL_SCORE_WEIGHT * MISTAKE_MATERIAL_PAWNS
BLUNDER_THRESHOLD = MATERIAL_SCORE_WEIGHT * BLUNDER_MATERIAL_PAWNS


def _blunder_score(board: chess.Board, move: chess.Move) -> float:
    """How much ``move`` costs its mover once the opponent replies optimally.

    A 1-ply lookahead, not a search: it evaluates the position right after
    ``move``, then again after every opponent reply, and returns how far the
    worst of those replies drags the evaluation down from before the move
    (from the mover's perspective). This only catches immediate tactical
    blunders such as hanging a piece -- deeper tactics need a real search.
    Post-move positions are scored with ``_terminal_aware_evaluate`` so that
    a stalemate escape (a draw) is correctly valued above losing outright,
    and a reply that delivers checkmate is valued as an outright loss rather
    than whatever the material count happens to be.
    """
    mover = board.turn
    before_for_mover = _evaluate_position(board)
    if mover == chess.BLACK:
        before_for_mover = -before_for_mover

    board.push(move)
    try:
        replies = list(board.legal_moves)
        if not replies:
            worst_for_mover = _terminal_aware_evaluate(board)
            if mover == chess.BLACK:
                worst_for_mover = -worst_for_mover
        else:
            worst_for_mover = math.inf
            for reply in replies:
                board.push(reply)
                try:
                    after = _terminal_aware_evaluate(board)
                finally:
                    board.pop()
                after_for_mover = -after if mover == chess.BLACK else after
                worst_for_mover = min(worst_for_mover, after_for_mover)
    finally:
        board.pop()

    return round(before_for_mover - worst_for_mover, 2)


def _move_severity(blunder_score: float) -> str:
    """Classify a move's eval swing into a chess.com-style severity tier."""
    if blunder_score >= BLUNDER_THRESHOLD:
        return "Blunder"
    if blunder_score >= MISTAKE_THRESHOLD:
        return "Mistake"
    if blunder_score >= INACCURACY_THRESHOLD:
        return "Inaccuracy"
    return ""


def _order_moves(board: chess.Board) -> list[chess.Move]:
    """Order legal moves so alpha-beta pruning cuts more of the tree.

    Captures first (most valuable capture first), then checks, then
    everything else.
    """

    def move_key(move: chess.Move) -> tuple[int, int]:
        if board.is_capture(move):
            if board.is_en_passant(move):
                captured_value = PIECE_POINTS[chess.PAWN]
            else:
                captured_piece = board.piece_at(move.to_square)
                captured_value = PIECE_POINTS[captured_piece.piece_type] if captured_piece else 0
            return (2, captured_value)
        if board.gives_check(move):
            return (1, 0)
        return (0, 0)

    return sorted(board.legal_moves, key=move_key, reverse=True)


def _negamax(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    *,
    legal_moves_weight: float = LEGAL_MOVES_WEIGHT,
    material_score_weight: float = MATERIAL_SCORE_WEIGHT,
    forward_score_weight: float = FORWARD_SCORE_WEIGHT,
    center_control_weight: float = CENTER_CONTROL_WEIGHT,
    checkmate_weight: float = CHECKMATE_WEIGHT,
    eval_counter: list[int] | None = None,
) -> float:
    """Negamax search with alpha-beta pruning, from the side-to-move's view.

    Terminal nodes score mate distance-adjusted (a mate found with more
    depth left to search is closer to the root, so it scores higher) and
    draws as 0. Leaves are scored by ``_evaluate_position``.
    """
    if board.is_checkmate():
        return -(MATE_SCORE + depth)
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if depth <= 0:
        if eval_counter is not None:
            eval_counter[0] += 1
        score = _evaluate_position(
            board,
            legal_moves_weight=legal_moves_weight,
            material_score_weight=material_score_weight,
            forward_score_weight=forward_score_weight,
            center_control_weight=center_control_weight,
            checkmate_weight=checkmate_weight,
        )
        return score if board.turn == chess.WHITE else -score

    best = -math.inf
    for move in _order_moves(board):
        board.push(move)
        try:
            score = -_negamax(
                board,
                depth - 1,
                -beta,
                -alpha,
                legal_moves_weight=legal_moves_weight,
                material_score_weight=material_score_weight,
                forward_score_weight=forward_score_weight,
                center_control_weight=center_control_weight,
                checkmate_weight=checkmate_weight,
                eval_counter=eval_counter,
            )
        finally:
            board.pop()
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def choose_engine_move(
    board: chess.Board,
    rng: random.Random | None = None,
    top_k: int = 3,
    *,
    legal_moves_weight: float = LEGAL_MOVES_WEIGHT,
    material_score_weight: float = MATERIAL_SCORE_WEIGHT,
    forward_score_weight: float = FORWARD_SCORE_WEIGHT,
    center_control_weight: float = CENTER_CONTROL_WEIGHT,
    checkmate_weight: float = CHECKMATE_WEIGHT,
    depth: int = 3,
    eval_counter: list[int] | None = None,
) -> tuple[chess.Move, float]:
    """Pick a move via negamax search.

    ``eval_counter``, if given, is a single-element ``[count]`` list that
    accumulates one increment per leaf position statically evaluated during
    the search -- callers use this to report how many evaluations a move
    (or a whole game) cost.
    """
    rng = rng or random.Random()
    depth = max(1, depth)
    scored_moves: list[tuple[float, chess.Move]] = []
    for move in _order_moves(board):
        board.push(move)
        try:
            value = -_negamax(
                board,
                depth - 1,
                -math.inf,
                math.inf,
                legal_moves_weight=legal_moves_weight,
                material_score_weight=material_score_weight,
                forward_score_weight=forward_score_weight,
                center_control_weight=center_control_weight,
                checkmate_weight=checkmate_weight,
                eval_counter=eval_counter,
            )
        finally:
            board.pop()
        scored_moves.append((value, move))

    if not scored_moves:
        raise ValueError("No legal moves available")

    scored_moves.sort(key=lambda item: item[0], reverse=True)
    top_n = scored_moves[: max(1, min(top_k, len(scored_moves)))]
    score, move = rng.choice(top_n)
    return move, score


def _calculate_total_score(
    legal_moves: int,
    material_score: int,
    forward_score: int,
    center_score: int = 0,
    *,
    legal_moves_weight: float = LEGAL_MOVES_WEIGHT,
    material_score_weight: float = MATERIAL_SCORE_WEIGHT,
    forward_score_weight: float = FORWARD_SCORE_WEIGHT,
    center_control_weight: float = CENTER_CONTROL_WEIGHT,
) -> float:
    """Blend mobility, material, forward, and center control into one position score.

    Formula:
        total_score = legal_moves_weight * legal_moves
                    + material_score_weight * material_score
                    + forward_score_weight * forward_score
                    + center_control_weight * center_score

    The weights keep material as the strongest signal, while still letting
    other factors move the score in a visible way.
    """
    return round(
        legal_moves_weight * legal_moves
        + material_score_weight * material_score
        + forward_score_weight * forward_score
        + center_control_weight * center_score
    , 2)


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


def _legal_moves_and_tree(board: chess.Board, lastmove: chess.Move | None = None) -> tuple[str, list[str], dict[str, list[str]], dict[str, int], dict[str, int], dict[str, int], dict[str, int], dict[str, int], int, int, int, int, int, dict[str, int]]:
    """Render board with legal moves arrows, and return SAN list, 2-ply move tree, control metrics, material metrics, scores and move scores."""
    tree = {}
    move_scores = {}
    legal_moves = list(board.legal_moves)
    arrows = _legal_move_arrows(board)

    # Pre-calculate current control and material
    f1, f2 = _calculate_forward(board)
    f3 = _calculate_forward_3(board)
    material = _calculate_material(board)
    center = _calculate_center_control(board)
    forward_score = (f1["White"] + f2["White"]) - (f1["Black"] + f2["Black"])
    material_score = material["White"] - material["Black"]
    center_score = center["White"] - center["Black"]
    score = len(legal_moves)
    total_score = _calculate_total_score(score, material_score, forward_score, center_score)

    for move in legal_moves:
        san = board.san(move)

        # 1-ply deep lookahead for the tree and scores
        board.push(move)
        tree[san] = [board.san(m) for m in board.legal_moves]

        # Optimization: for move suggestions, use 1st order control ONLY to save time
        # 2nd order control is expensive to calculate for every legal move (O(N^2))
        sc1 = get_board_control(board)

        # Move score is simplified to 1st order difference after the move
        move_scores[san] = sc1["White"] - sc1["Black"]

        board.pop()

    sans = [board.san(move) for move in legal_moves]
    svg = chess.svg.board(board, size=420, lastmove=lastmove, arrows=arrows)
    return _style_arrows(svg), sans, tree, f1, f2, f3, material, center, forward_score, material_score, center_score, score, total_score, move_scores


def load_game_detail(pgn_text: str, index: int) -> GameDetail | None:
    """Parse a single game and render every board position as SVG."""
    game = _read_game_at(pgn_text, index)
    if game is None:
        return None

    headers = game.headers
    board = game.board()

    start_moves_svg, start_legal, start_tree, start_f1, start_f2, start_f3, start_material, start_center, start_forward_score, start_material_score, start_center_score, start_score, start_total_score, start_scores = _legal_moves_and_tree(board)
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
            material=start_material,
            center=start_center,
            forward_score=start_forward_score,
            material_score=start_material_score,
            center_score=start_center_score,
            score=start_score,
            total_score=start_total_score,
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
        moves_svg, legal, tree, f1, f2, f3, material, center, forward_score, material_score, center_score, score, total_score, scores = _legal_moves_and_tree(board, lastmove=move)
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
                material=material,
                center=center,
                forward_score=forward_score,
                material_score=material_score,
                center_score=center_score,
                score=score,
                total_score=total_score,
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
