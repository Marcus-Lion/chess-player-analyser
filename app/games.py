from __future__ import annotations

import re
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


_ARROW_BORDER_COLOR = "#0a3d0a"


def _style_arrows(svg: str) -> str:
    """Make the overlaid legal-move arrows smaller and add a border.

    ``python-chess`` renders each arrow as a ``<line class="arrow">`` shaft and
    a ``<polygon class="arrow">`` head with no size/border options, so we
    post-process the generated SVG: the shaft is thinned and drawn on top of a
    slightly wider border underlay, and the arrowhead gets an outline.
    """

    def line_repl(match: re.Match[str]) -> str:
        tag = match.group(0)
        width_match = re.search(r'stroke-width="([\d.]+)"', tag)
        if not width_match:
            return tag
        original = float(width_match.group(1))
        thin = round(original * 0.45, 2)
        border = max(1.4, round(thin * 0.4, 2))
        thin_tag = re.sub(r'stroke-width="[\d.]+"', f'stroke-width="{thin}"', tag)
        under_tag = re.sub(r'stroke-width="[\d.]+"',
                           f'stroke-width="{round(thin + 2 * border, 2)}"', tag)
        under_tag = re.sub(r'stroke="[^"]*"',
                           f'stroke="{_ARROW_BORDER_COLOR}"', under_tag)
        under_tag = re.sub(r'opacity="[^"]*"\s*', "", under_tag)
        # Border underlay first so the coloured shaft is drawn on top of it.
        return under_tag + thin_tag

    def poly_repl(match: re.Match[str]) -> str:
        tag = match.group(0)
        if "stroke=" in tag:
            return tag
        return (tag[:-2].rstrip()
                + f' stroke="{_ARROW_BORDER_COLOR}" stroke-width="1.4"'
                  ' stroke-linejoin="round"/>')

    svg = re.sub(r'<line\b[^>]*class="arrow"[^>]*/>', line_repl, svg)
    svg = re.sub(r'<polygon\b[^>]*class="arrow"[^>]*/>', poly_repl, svg)
    return svg


def _legal_moves_svg(board: chess.Board, lastmove: chess.Move | None = None) -> tuple[str, list[str]]:
    """Render a board that overlays arrows for every legal move.

    Returns the SVG string and the list of legal moves in SAN notation.
    """
    arrows = [
        chess.svg.Arrow(move.from_square, move.to_square, color="#15781B80")
        for move in board.legal_moves
    ]
    sans = [board.san(move) for move in board.legal_moves]
    svg = chess.svg.board(board, size=420, lastmove=lastmove, arrows=arrows)
    return _style_arrows(svg), sans


def load_game_detail(pgn_text: str, index: int) -> GameDetail | None:
    """Parse a single game and render every board position as SVG."""
    game = _read_game_at(pgn_text, index)
    if game is None:
        return None

    headers = game.headers
    board = game.board()

    start_moves_svg, start_legal = _legal_moves_svg(board)
    positions: list[GamePosition] = [
        GamePosition(
            ply=0,
            move_number=0,
            san="Start",
            side="",
            fen=board.fen(),
            svg=chess.svg.board(board, size=420),
            svg_moves=start_moves_svg,
            legal_moves=start_legal,
        )
    ]

    ply = 0
    for move in game.mainline_moves():
        ply += 1
        side = "White" if board.turn == chess.WHITE else "Black"
        move_number = board.fullmove_number
        san = board.san(move)
        board.push(move)
        moves_svg, legal = _legal_moves_svg(board, lastmove=move)
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
