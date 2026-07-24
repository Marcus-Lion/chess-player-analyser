# Gameplay rules

How the self-play engine (`app/games.py`, `app/self_play.py`, `engine/`) picks
moves and ends games. This is the reference for the native negamax search,
its pruning optimizations, and the self-play termination rules.

## Move selection: negamax with alpha-beta pruning

`app.games.choose_engine_move` passes the position to the required native
`chess_engine` extension. Its Rust `negamax` implementation searches from
each legal root move down to the selected depth. Depth is counted in
**plies** (half-moves) — depth 4 means "look 4 half-moves ahead" (two of the
mover's turns and two of the opponent's), not 4 full move-pairs.

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

After ranking, `top_k` sets the maximum candidate count. The optional
`top_k_score_threshold` then removes candidates whose score is more than that
amount below the best score. A threshold of 0 allows only moves tied for best;
the default threshold is 3.0, and setting the API/config value to `None`
preserves unrestricted Top-K selection. One move is chosen randomly from the
remaining candidates.

### Move ordering

Cutoffs only fire early if the strongest moves are searched first, so
The Rust engine's `ordered_moves` function ranks legal moves before each node is searched, in this
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

Each native search owns a shared killer-move table keyed by depth. When
a quiet move triggers `alpha >= beta`, it's recorded at that depth so the
next sibling subtree at the same depth tries it first. One table is created
per `choose_engine_move` call and threaded through the whole search.

### Transposition table

The native search also owns a transposition table keyed by a Zobrist hash of
the position itself, not the move sequence that reached it. Chess positions
commonly transpose (different move orders reaching the same position), so
caching by position catches repeats a killer table (keyed only by depth)
can't.

Each Rust entry stores its depth, score, bound flag, and best move. Its bound
flag is exact, lower-bound, or upper-bound:

- `TT_EXACT` — the stored score is the position's true negamax value.
- `TT_LOWERBOUND` — the search that produced it was cut off by `beta`, so
  the true value is only known to be *at least* the stored score.
- `TT_UPPERBOUND` — cut off by `alpha`, so the true value is only known to
  be *at most* the stored score.

On lookup, if the cached entry was searched to at least the requested depth,
its score/bound is reused instead of re-searching the subtree (or narrows
`alpha`/`beta` before searching), and its best move is tried first by move
ordering. One table is created per `choose_engine_move` call, shared
across every root move's subtree, so a transposition reached via a different
root move still benefits.

### Threefold-repetition avoidance

When the side to move is ahead by at least `REPETITION_AVOIDANCE_MATERIAL_PAWNS`
(1 by default) pawn of material, moves that would create a third occurrence are
penalized by `REPETITION_AVOIDANCE_PENALTY` (500) after scoring, so the
engine presses for a win instead of settling for a draw when it's already
ahead. Below that material threshold, repetitions are scored normally.

## Position evaluation

Leaf nodes (`depth <= 0`) are scored by the native static evaluator, also
exposed through `app.games._evaluate_position`, using this weighted blend:

| Component | Weight | What it measures |
|---|---|---|
| Legal moves | `LEGAL_MOVES_WEIGHT` (-2.0) | White-minus-Black mobility. |
| Material | `MATERIAL_SCORE_WEIGHT` (1.0) | Piece points: pawn=1, knight/bishop=3, rook=5, queen=9. |
| Forward control | `FORWARD_SCORE_WEIGHT` (1.0) | Squares attacked on each side's forward two ranks. |
| Center control | `CENTER_CONTROL_WEIGHT` (1.0) | Attackers on d4/e4/d5/e5. |
| Checkmate pressure | `CHECKMATE_WEIGHT` (1.0) | King-safety/mate-threat heuristic (`_mate_pressure`). |

Checkmate is scored as `MATE_SCORE` (1,000,000) adjusted by remaining search
depth (a mate found with more depth left to search is closer to the root, so
it scores higher); stalemate and insufficient material score 0.

## Search depth

Self-play defaults to **auto-scaled depth** (`_auto_search_depth`): depth is
inversely proportional to material remaining on the board, scaled between
`MIN_AUTO_SEARCH_DEPTH` (3) and a configurable maximum (7 by default) by an
exponential curve (`AUTO_SEARCH_DEPTH_CURVE_EXPONENT`, 0.45) over the fraction
of the starting material (39 per side, 78 combined) that's been traded off. A
full board gets the shallowest depth; as material thins out, depth ramps toward
the configured cap. Set **Max depth** in the web harness, pass `--max-depth`,
or set `SelfPlayConfig.max_depth` to change the cap. Values below 3 are strict
caps and therefore keep every automatically selected depth at that value.
Pass `--depth`/`depth=` to bypass automatic scaling and pin a fixed depth for
the whole game instead.

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
