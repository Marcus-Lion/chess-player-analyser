from __future__ import annotations

from dataclasses import dataclass
import random

from app.games import (
    CENTER_CONTROL_WEIGHT,
    FORWARD_SCORE_WEIGHT,
    LEGAL_MOVES_WEIGHT,
    MATERIAL_SCORE_WEIGHT,
)

BASE_WEIGHT_MEANS = {
    "legal_moves_weight": LEGAL_MOVES_WEIGHT,
    "material_score_weight": MATERIAL_SCORE_WEIGHT,
    "forward_score_weight": FORWARD_SCORE_WEIGHT,
    "center_control_weight": CENTER_CONTROL_WEIGHT,
}


@dataclass(frozen=True)
class PlayerProfile:
    player_id: str
    name: str
    description: str
    legal_moves_weight: float
    material_score_weight: float
    forward_score_weight: float
    center_control_weight: float

    @property
    def weights(self) -> dict[str, float]:
        return {
            "legal_moves_weight": self.legal_moves_weight,
            "material_score_weight": self.material_score_weight,
            "forward_score_weight": self.forward_score_weight,
            "center_control_weight": self.center_control_weight,
        }


_PLAYER_SPECS: list[tuple[str, str, str, int]] = [
    ("iron-sentinel", "Iron Sentinel", "Disciplined and stubborn; prefers safe structures and steady accumulation.", 101),
    ("velvet-gambit", "Velvet Gambit", "Soft on the surface, sharp underneath; trades space for initiative.", 102),
    ("north-star", "North Star", "Keeps the game on a fixed course with calm, positional pressure.", 103),
    ("clockwork-fox", "Clockwork Fox", "Mechanical tempo player that punishes loose move order and wasted turns.", 104),
    ("granite-owl", "Granite Owl", "Slow, heavy, and difficult to dislodge; leans hard on material safety.", 105),
    ("ember-knight", "Ember Knight", "Aggressive and direct; likes active pieces and forward momentum.", 106),
    ("silver-sloop", "Silver Sloop", "Light-footed and opportunistic, always looking for a tactical lane.", 107),
    ("lunar-broker", "Lunar Broker", "Balances the board like a market maker, shifting value without panic.", 108),
    ("copper-raven", "Copper Raven", "Scavenger style: grabs small edges and compounds them into pressure.", 109),
    ("azure-pike", "Azure Pike", "Straight-line attacker that prefers open files and clean lines.", 110),
    ("storm-weaver", "Storm Weaver", "Builds cross-board tension and turns complexity into control.", 111),
    ("cedar-oracle", "Cedar Oracle", "Patient and far-sighted; anticipates structure changes before they happen.", 112),
    ("prism-viper", "Prism Viper", "Sees one move deeper than it should; thrives in tactical fragmentation.", 113),
    ("hollow-atlas", "Hollow Atlas", "Flexible and evasive, content to reshape the board around the opponent.", 114),
    ("cinder-monk", "Cinder Monk", "Minimalist grinder that values piece activity over raw force.", 115),
    ("frost-herald", "Frost Herald", "Cool under pressure and hard to rush; favors balance and restraint.", 116),
]


def _clamp(value: float, low: float = -4.0, high: float = 4.0) -> float:
    return min(max(value, low), high)


def _build_profile(player_id: str, name: str, description: str, seed: int) -> PlayerProfile:
    rng = random.Random(seed)
    stddev = 0.35
    return PlayerProfile(
        player_id=player_id,
        name=name,
        description=description,
        legal_moves_weight=_clamp(rng.gauss(BASE_WEIGHT_MEANS["legal_moves_weight"], stddev)),
        material_score_weight=_clamp(rng.gauss(BASE_WEIGHT_MEANS["material_score_weight"], stddev)),
        forward_score_weight=_clamp(rng.gauss(BASE_WEIGHT_MEANS["forward_score_weight"], stddev)),
        center_control_weight=_clamp(rng.gauss(BASE_WEIGHT_MEANS["center_control_weight"], stddev)),
    )


PLAYER_ROSTER: list[PlayerProfile] = [
    _build_profile(player_id, name, description, seed)
    for player_id, name, description, seed in _PLAYER_SPECS
]
PLAYER_BY_ID = {player.player_id: player for player in PLAYER_ROSTER}


def get_player_roster() -> list[PlayerProfile]:
    return list(PLAYER_ROSTER)


def get_player(player_id: str) -> PlayerProfile | None:
    return PLAYER_BY_ID.get(player_id)


def pick_two_players(
    rng: random.Random,
    skill_levels: dict[str, float] | None = None,
    roster: list[PlayerProfile] | None = None,
) -> tuple[PlayerProfile, PlayerProfile]:
    """Pick a self-play pairing.

    When ``skill_levels`` is provided, pair adjacent players in skill order so
    the matchup stays near the same strength band. Without skill data, fall
    back to an unbiased random pair.
    """

    pool = roster or PLAYER_ROSTER

    if skill_levels:
        ranked = sorted(
            pool,
            key=lambda player: (skill_levels.get(player.player_id, 1500.0), player.player_id),
        )
        if len(ranked) >= 2:
            start = rng.randrange(len(ranked) - 1)
            first, second = ranked[start], ranked[start + 1]
            if rng.random() < 0.5:
                return first, second
            return second, first

    white, black = rng.sample(pool, 2)
    return white, black
