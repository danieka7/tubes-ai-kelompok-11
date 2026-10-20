# =============================================================================
# BAGIAN 5B -- MODE BATTLE: MINIMAX / ALPHA-BETA (ADVERSARIAL SEARCH)
# =============================================================================
# Begitu drone cukup dekat dengan tank, permainan berpindah dari pathfinding
# di grid (UCS/A*) ke sub-sistem battle terpisah yang state-nya sederhana
# (HP, kit reparasi, status bertahan) dan dikendalikan Minimax/Alpha-Beta.
#
# Dua kelas algoritma pencarian AI yang berbeda dipakai untuk dua masalah
# yang berbeda:
#   - A* / UCS   -> pathfinding: pencarian rute di ruang keadaan grid.
#   - Minimax    -> pengambilan keputusan adversarial: game dua pemain,
#                   giliran bergantian, tujuan berlawanan.

PLAYER = "PLAYER"
NPC = "NPC"

ACT_ATTACK = "ATTACK"
ACT_DEFEND = "DEFEND"
ACT_REPAIR = "REPAIR"

BATTLE_START_REPAIR_KITS = 3
ATTACK_DAMAGE = 18
DEFEND_REDUCTION = 0.5          # damage masuk dikurangi 50% kalau lawan bertahan
REPAIR_HEAL_AMOUNT = 30         # HP yang dipulihkan mesin per satu kit reparasi

BATTLE_WIN_SCORE = 1000.0
BATTLE_DEFAULT_DEPTH = 4        # MAX_DEPTH -- batas cutoff, branching <=3 aksi

BATTLE_TRIGGER_DISTANCE = 1     # jarak (Manhattan) drone<->tank yang memicu battle
BATTLE_END_PAUSE_MS = 1800      # jeda menampilkan hasil sebelum kembali ke eksplorasi
BATTLE_LOG_MAX = 6
ROSTER_TOP = 132          # offset daftar drone dari atas area battle
ROSTER_ROW_H = 20
ROSTER_BAR_X = 96
ROSTER_BAR_W = 64
ROSTER_BAR_H = 8
ROSTER_TAG_X = 232
BATTLE_ACTION_TOP = 306   # offset tombol aksi dari atas area battle
BATTLE_ACTION_H = 36

# Mode algoritma pengambilan keputusan NPC (dipilih lewat tombol, bukan toggle)
MODE_MINIMAX = "minimax"
MODE_ALPHABETA = "alphabeta"
MODE_EXPECTIMAX = "expectimax"
MODE_HILL = "hillclimb"
MODE_ANNEAL = "anneal"
LOCAL_MODES = (MODE_HILL, MODE_ANNEAL)
