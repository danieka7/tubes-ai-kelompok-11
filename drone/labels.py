from battle_constants import (
    ACT_ATTACK,
    ACT_DEFEND,
    ACT_REPAIR,
    MODE_ALPHABETA,
    MODE_ANNEAL,
    MODE_EXPECTIMAX,
    MODE_HILL,
    MODE_MINIMAX,
)


ACTION_LABELS = {ACT_ATTACK: "Serang", ACT_DEFEND: "Bertahan", ACT_REPAIR: "Reparasi", None: "ROOT"}
MODE_LABELS = {
    MODE_MINIMAX: "Minimax", MODE_ALPHABETA: "Alpha-Beta", MODE_EXPECTIMAX: "Expectimax",
    MODE_HILL: "Hill Climbing", MODE_ANNEAL: "Simulated Annealing",
}
KIND_LABELS = {"terminal": "terminal (utility)", "cutoff": "cutoff (evaluasi)", "internal": "internal"}
