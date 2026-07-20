# Gameplay rules

How the self-play engine (`app/games.py`, `app/self_play.py`) picks moves and
ends games. This is the reference for the negamax search, its pruning
optimizations, and the self-play termination rules.

## Move selection: negamax with alpha-beta pruning

`choose_engine_move` picks a move by searching the game tree with negamax
(`_negamax`), from the position after each legal root move down to a fixed
depth. Depth is counted in **plies** (half-moves) — depth 4 means "look 4
half-moves ahead" (two of the mover's turns and two of the opponent's), not
4 full move-pairs.

At each node the search recurses into every child (one per legal move),
negating each child's score to convert it into "value from the current
side-to-move's perspective" — that's the "negamax" trick: no separate
min/max logic for White vs Black.

### Alpha-beta pruning

`alpha`/`beta` bound the range of scores still worth searching. After each
move is scored, if `alpha >= beta` the remaining, not-yet-searched sibling
moves at that node are skipped (`break`) — they're provably too good for the
opponent to ever allow. This never changes which move is chosen; it only
skips searching branches whose outcome can't affect the result.

Root moves themselves are always searched with a full `(-inf, inf)` window
(no pruning at the root), since every root move's exact score is needed to
rank the top `top_k` candidates.

### Move ordering

Cutoffs only fire early if the strongest moves are searched first, so
`_order_moves` ranks legal moves before each node is searched, in this
priority order:

1. **Transposition-table move** — the best move found the last time this
   exact position was searched (see below).
2. **Captures**, ranked by captured piece value, ties broken by the
   *cheapest attacker* first (MVV-LVA) — e.g. pawn-takes-queen outranks
   queen-takes-queen, since losing the pawn is cheaper if the capture is
   recaptured.
3. **Checks**.
4. **Killer move** — a quiet (non-capture) move that caused a beta cutoff in
   a *sibling* branch at the same depth, tried early on the theory that
   what worked for one sibling often works for another.
5. Everything else.

### Killer-move table

`_negamax` takes an optional shared `killer_moves: {depth: move}` dict. When
a quiet move triggers `alpha >= beta`, it's recorded at that depth so the
next sibling subtree at the same depth tries it first. One table is created
per `choose_engine_move` call and threaded through the whole search.

### Transposition table

`_negamax` also takes an optional shared `transposition_table` dict, keyed by
`chess.polyglot.zobrist_hash(board)` — a hash of the position itself, not the
move sequence that reached it. Chess positions commonly transpose (different
move orders reaching the same position), so caching by position catches
repeats a killer table (keyed only by depth) can't.

Each entry stores `(depth, score, flag, best_move)`, where `flag` is one of:

- `TT_EXACT` — the stored score is the position's true negamax value.
- `TT_LOWERBOUND` — the search that produced it was cut off by `beta`, so
  the true value is only known to be *at least* the stored score.
- `TT_UPPERBOUND` — cut off by `alpha`, so the true value is only known to
  be *at most* the stored score.

On lookup, if the cached entry was searched to at least the requested depth,
its score/bound is reused instead of re-searching the subtree (or narrows
`alpha`/`beta` before searching), and its `best_move` is tried first via
`_order_moves`. One table is created per `choose_engine_move` call, shared
across every root move's subtree, so a transposition reached via a different
root move still benefits.

### Threefold-repetition avoidance

When the side to move is ahead by at least `REPETITION_AVOIDANCE_MATERIAL_PAWNS`
(2) pawns of material, moves that would trigger `board.is_repetition(3)` are
penalized by `REPETITION_AVOIDANCE_PENALTY` (500) after scoring, so the
engine presses for a win instead of settling for a draw when it's already
ahead. Below that material threshold, repetitions are scored normally.

## Position evaluation

Leaf nodes (`depth <= 0`) are scored by `_evaluate_position`, a weighted
blend of:

| Component | Weight | What it measures |
|---|---|---|
| Legal moves | `LEGAL_MOVES_WEIGHT` (0.3) | Mobility — how many legal replies the side to move has. |
| Material | `MATERIAL_SCORE_WEIGHT` (0.35) | Piece points: pawn=1, knight/bishop=3, rook=5, queen=9. |
| Forward control | `FORWARD_SCORE_WEIGHT` (0.20) | Squares attacked on each side's forward two ranks. |
| Center control | `CENTER_CONTROL_WEIGHT` (0.125) | Attackers on d4/e4/d5/e5. |
| Checkmate pressure | `CHECKMATE_WEIGHT` (1.0) | King-safety/mate-threat heuristic (`_mate_pressure`). |

Checkmate is scored as `MATE_SCORE` (1,000,000) adjusted by remaining search
depth (a mate found with more depth left to search is closer to the root, so
it scores higher); stalemate and insufficient material score 0.

## Search depth

Self-play defaults to **auto-scaled depth** (`_auto_search_depth`): depth is
inversely proportional to material remaining on the board, scaled between
`MIN_AUTO_SEARCH_DEPTH` (3) and `MAX_AUTO_SEARCH_DEPTH` (7) by an exponential
curve (`AUTO_SEARCH_DEPTH_CURVE_EXPONENT`, 0.45) over the fraction of the
starting material (39 per side, 78 combined) that's been traded off. A full
board gets the shallowest depth (largest branching factor, most expensive to
search); as material thins out, depth ramps up toward the endgame, where
deeper search matters most for precision. Pass `--depth`/`depth=` to pin a
fixed depth for the whole game instead.

## Self-play game termination

Checked in this order by `_terminal_reason` (`app/self_play.py`) after every
move:

1. Checkmate
2. Stalemate
3. Insufficient material
4. Fivefold repetition (automatic backstop)
5. 75-move rule (automatic backstop)
6. Threefold repetition (**adjudicated automatically as soon as claimable**
   — self-play has no player to claim a draw)
7. Fifty-move rule (also adjudicated automatically)

If none of these trigger, the game is called a draw once `--max-turns` is
reached (reported as `"max turns reached"`; the count covers every
half-move, i.e. both White's and Black's turns).
