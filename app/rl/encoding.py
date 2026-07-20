from __future__ import annotations

import chess
import numpy as np

BOARD_PLANES = 12
BOARD_SQUARES = 64
BOARD_EXTRA_FEATURES = 1 + 4 + 8 + 2
BOARD_FEATURE_DIM = BOARD_PLANES * BOARD_SQUARES + BOARD_EXTRA_FEATURES

MOVE_FEATURE_DIM = 64 + 64 + 5 + 6 + 4 + 2


def _square_one_hot(square: int) -> np.ndarray:
    vec = np.zeros(BOARD_SQUARES, dtype=np.float32)
    vec[square] = 1.0
    return vec


def encode_board(board: chess.Board) -> np.ndarray:
    features = np.zeros(BOARD_FEATURE_DIM, dtype=np.float32)
    offset = 0
    plane_order = (
        (chess.WHITE, chess.PAWN),
        (chess.WHITE, chess.KNIGHT),
        (chess.WHITE, chess.BISHOP),
        (chess.WHITE, chess.ROOK),
        (chess.WHITE, chess.QUEEN),
        (chess.WHITE, chess.KING),
        (chess.BLACK, chess.PAWN),
        (chess.BLACK, chess.KNIGHT),
        (chess.BLACK, chess.BISHOP),
        (chess.BLACK, chess.ROOK),
        (chess.BLACK, chess.QUEEN),
        (chess.BLACK, chess.KING),
    )
    for color, piece_type in plane_order:
        for square in board.pieces(piece_type, color):
            features[offset + square] = 1.0
        offset += BOARD_SQUARES

    features[offset] = 1.0 if board.turn == chess.WHITE else 0.0
    offset += 1
    features[offset] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    features[offset + 1] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    features[offset + 2] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    features[offset + 3] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    offset += 4

    if board.ep_square is not None:
        features[offset + chess.square_file(board.ep_square)] = 1.0
    offset += 8

    features[offset] = min(board.halfmove_clock, 100) / 100.0
    features[offset + 1] = min(board.fullmove_number, 200) / 200.0
    return features


def encode_move(board: chess.Board, move: chess.Move) -> np.ndarray:
    features = np.zeros(MOVE_FEATURE_DIM, dtype=np.float32)
    offset = 0
    features[offset + move.from_square] = 1.0
    offset += 64
    features[offset + move.to_square] = 1.0
    offset += 64

    promotion_order = {
        None: 0,
        chess.QUEEN: 1,
        chess.ROOK: 2,
        chess.BISHOP: 3,
        chess.KNIGHT: 4,
    }
    features[offset + promotion_order.get(move.promotion, 0)] = 1.0
    offset += 5

    moving_piece = board.piece_at(move.from_square)
    if moving_piece is not None:
        features[offset + (moving_piece.piece_type - 1)] = 1.0
    offset += 6

    features[offset] = 1.0 if board.is_capture(move) else 0.0
    features[offset + 1] = 1.0 if board.gives_check(move) else 0.0
    features[offset + 2] = 1.0 if board.is_castling(move) else 0.0
    features[offset + 3] = 1.0 if board.is_en_passant(move) else 0.0
    offset += 4

    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    features[offset] = (to_file - from_file) / 7.0
    features[offset + 1] = (to_rank - from_rank) / 7.0
    return features
