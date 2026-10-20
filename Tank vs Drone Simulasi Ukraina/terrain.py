# =============================================================================
# BAGIAN 1 -- KONSTANTA DUNIA & JENIS MEDAN
# =============================================================================
ROWS = 16
COLS = 22
CELL = 30

DIRT = 0
TREE = 1
ROCK = 2
RIVER = 3
BRIDGE = 4
ARTILLERY = 5
CAMP = 6
RUIN = 7

TANK_TERRAIN_COST = {DIRT: 1, BRIDGE: 1}
DRONE_TERRAIN_COST = {
    DIRT: 1,
    BRIDGE: 1,
    ROCK: 2,
    RUIN: 2,
    RIVER: 2,
    TREE: 4,
    CAMP: 6,
}


def is_blocked(t):
    """Aturan TANK (jalan darat)."""
    return t in (TREE, ROCK, RIVER, ARTILLERY, CAMP, RUIN)


def is_blocked_for_drone(t):
    """Aturan DRONE (terbang) -- hanya artileri anti-udara yang menghalangi."""
    return t == ARTILLERY


SIGHT_BLOCKERS = {TREE, ROCK, CAMP, RUIN}


def blocks_sight(t):
    return t in SIGHT_BLOCKERS
