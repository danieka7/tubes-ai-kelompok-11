"""
Tank vs Drone -- UCS vs A* Pathfinding (versi Pygame)
======================================================
Port dari simulasi HTML/JS asli. Logika medan, biaya tempuh, algoritma
pencarian (UCS / A*), fog-of-war (radius pandang + line-of-sight),
pola pencarian boustrophedon, dan sekuens pemindaian awal (scanning ->
moving -> searching/chasing/investigating) diusahakan sama persis
dengan versi web-nya. Bagian visual disederhanakan supaya cocok
digambar dengan primitif Pygame (rect/circle/line), bukan re-implementasi
1:1 dari canvas API.

Mode Battle (Minimax/Alpha-Beta): begitu drone cukup dekat dengan tank,
permainan berpindah dari mode eksplorasi (UCS/A* di grid) ke mode battle
-- sebuah sub-sistem terpisah dengan state sederhana (HP, potion, status
bertahan) yang dikendalikan lewat adversarial search (Minimax / Alpha-Beta
pruning), bukan lagi pathfinding.

Kontrol:
  - Panah / WASD           : gerakkan tank (mode eksplorasi)
  - Klik tombol di panel kanan : ganti algoritma / heuristik / radius
                                  pandang, acak medan, reset posisi,
                                  toggle auto-chase, jalankan eksperimen
  - Mode Battle             : [1] Serang  [2] Bertahan  [3] Potion
                               [B] toggle alpha-beta   [,] / [.] ubah depth
                               [T] tampilkan/sembunyikan pohon pencarian (debug)
                               [ [ ] ubah kedalaman pohon yang digambar
"""

import math
import random
import sys
import time

import pygame

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


# =============================================================================
# BAGIAN 2 -- UTILITAS ACAK & PEMBANGKIT MEDAN PERANG
# =============================================================================
def rand_int(lo, hi):
    return random.randint(lo, hi)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def generate_river_path():
    cells = []
    visited = set()
    r = 0
    c = clamp(COLS // 2 + rand_int(-2, 2), 2, COLS - 3)
    cells.append((r, c))
    visited.add((r, c))

    dir_pool = [(1, 0), (1, -1), (1, 1), (0, -1), (0, 1)]

    guard = ROWS * COLS * 3
    while r < ROWS - 1 and guard > 0:
        guard -= 1
        dr, dc = dir_pool[rand_int(0, len(dir_pool) - 1)]
        nr = clamp(r + dr, 0, ROWS - 1)
        nc = clamp(c + dc, 2, COLS - 3)
        k = (nr, nc)
        if k not in visited:
            visited.add(k)
            cells.append(k)
        r, c = nr, nc

    while r < ROWS - 1:
        r += 1
        k = (r, c)
        if k not in visited:
            visited.add(k)
            cells.append(k)
    return cells


def pick_bridge_cells(river_cells, count):
    by_row = {}
    for (r, c) in river_cells:
        if r <= 0 or r >= ROWS - 1:
            continue
        by_row.setdefault(r, []).append((r, c))

    rows_pool = list(by_row.keys())
    bridges = []
    while len(bridges) < count and rows_pool:
        idx = rand_int(0, len(rows_pool) - 1)
        row = rows_pool.pop(idx)
        bridges.extend(by_row[row])
    return bridges


def build_bridge_exclusion_zone(bridge_cells):
    zone = set()
    for (br, bc) in bridge_cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = br + dr, bc + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    zone.add((nr, nc))
    return zone


def can_place_block_2x2(grid, r, c, exclude):
    if r + 1 >= ROWS or c + 1 >= COLS:
        return False
    for dr in (0, 1):
        for dc in (0, 1):
            if grid[r + dr][c + dc] != DIRT:
                return False
            if exclude and (r + dr, c + dc) in exclude:
                return False
    return True


def is_reachable(grid, start, goal):
    visited = [[False] * COLS for _ in range(ROWS)]
    queue = [start]
    visited[start[0]][start[1]] = True
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        cur = queue.pop(0)
        if cur == goal:
            return True
        for dr, dc in dirs:
            nr, nc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            if visited[nr][nc]:
                continue
            if is_blocked(grid[nr][nc]):
                continue
            visited[nr][nc] = True
            queue.append((nr, nc))
    return False


def generate_map(start_pos, goal_pos):
    attempt = 0
    grid = None
    while True:
        attempt += 1
        grid = [[DIRT] * COLS for _ in range(ROWS)]

        river_cells = generate_river_path()
        for (r, c) in river_cells:
            grid[r][c] = RIVER

        bridge_cells = pick_bridge_cells(river_cells, 2)
        for (r, c) in bridge_cells:
            grid[r][c] = BRIDGE

        bridge_zone = build_bridge_exclusion_zone(bridge_cells)

        tree_budget = int(ROWS * COLS * 0.08)
        tree_safety = tree_budget * 40
        while tree_budget > 0 and tree_safety > 0:
            tree_safety -= 1
            r, c = rand_int(0, ROWS - 1), rand_int(0, COLS - 1)
            if grid[r][c] == DIRT and (r, c) not in bridge_zone:
                grid[r][c] = TREE
                tree_budget -= 1

        rock_budget = int(ROWS * COLS * 0.06)
        rock_safety = rock_budget * 40
        while rock_budget > 0 and rock_safety > 0:
            rock_safety -= 1
            r, c = rand_int(0, ROWS - 1), rand_int(0, COLS - 1)
            if grid[r][c] == DIRT and (r, c) not in bridge_zone:
                grid[r][c] = ROCK
                rock_budget -= 1

        art_budget = 5
        art_safety = 400
        while art_budget > 0 and art_safety > 0:
            art_safety -= 1
            r, c = rand_int(0, ROWS - 1), rand_int(0, COLS - 1)
            if grid[r][c] == DIRT and (r, c) not in bridge_zone:
                grid[r][c] = ARTILLERY
                art_budget -= 1

        camp_budget = 2
        safety = 400
        while camp_budget > 0 and safety > 0:
            safety -= 1
            r, c = rand_int(0, ROWS - 2), rand_int(0, COLS - 2)
            if can_place_block_2x2(grid, r, c, bridge_zone):
                grid[r][c] = grid[r][c + 1] = grid[r + 1][c] = grid[r + 1][c + 1] = CAMP
                camp_budget -= 1

        ruin_budget = 2
        safety = 400
        while ruin_budget > 0 and safety > 0:
            safety -= 1
            r, c = rand_int(0, ROWS - 2), rand_int(0, COLS - 2)
            if can_place_block_2x2(grid, r, c, bridge_zone):
                grid[r][c] = grid[r][c + 1] = grid[r + 1][c] = grid[r + 1][c + 1] = RUIN
                ruin_budget -= 1

        grid[start_pos[0]][start_pos[1]] = DIRT
        grid[goal_pos[0]][goal_pos[1]] = DIRT

        if is_reachable(grid, start_pos, goal_pos) or attempt >= 25:
            break
    return grid


def pick_random_start_positions():
    margin = 1
    min_distance = (ROWS + COLS) // 3

    drone_pos = tank_pos = None
    attempts = 0
    while True:
        drone_pos = (rand_int(margin, ROWS - 1 - margin), rand_int(margin, COLS - 1 - margin))
        tank_pos = (rand_int(margin, ROWS - 1 - margin), rand_int(margin, COLS - 1 - margin))
        attempts += 1
        dist = abs(drone_pos[0] - tank_pos[0]) + abs(drone_pos[1] - tank_pos[1])
        if attempts >= 60 or dist >= min_distance:
            break

    if drone_pos == tank_pos:
        tank_pos = (clamp(tank_pos[0] + 1, margin, ROWS - 1 - margin), tank_pos[1])

    return drone_pos, tank_pos


# =============================================================================
# BAGIAN 3 -- ALGORITMA PENCARIAN: UCS & A* DALAM SATU FUNGSI
# =============================================================================
def neighbors(node, grid):
    # Urutan atas, kanan, kiri, bawah -- menentukan arah penyebaran gelombang
    # saat beberapa node punya f yang sama.
    dirs = [(-1, 0), (0, 1), (0, -1), (1, 0)]
    result = []
    r, c = node
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < ROWS and 0 <= nc < COLS):
            continue
        terrain = grid[nr][nc]
        if is_blocked_for_drone(terrain):
            continue
        result.append(((nr, nc), DRONE_TERRAIN_COST[terrain]))
    return result


def h_manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def h_euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def h_chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def h_zero(a, b):
    return 0


HEURISTICS = {
    "manhattan": h_manhattan,
    "euclidean": h_euclidean,
    "chebyshev": h_chebyshev,
    "zero": h_zero,
}


class SearchResult:
    __slots__ = ("path", "cost", "expanded_count", "expanded_order", "time_ms", "found")

    def __init__(self, path, cost, expanded_count, expanded_order, time_ms, found):
        self.path = path
        self.cost = cost
        self.expanded_count = expanded_count
        self.expanded_order = expanded_order
        self.time_ms = time_ms
        self.found = found


def search(grid, start, goal, algorithm, heuristic_name):
    t0 = time.perf_counter()
    h_func = h_zero if algorithm == "ucs" else HEURISTICS[heuristic_name]

    open_list = [{"pos": start, "g": 0, "f": h_func(start, goal)}]
    came_from = {}
    g_score = {start: 0}
    closed = set()
    expanded_count = 0
    expanded_order = []  # [(pos, g), ...] urutan ekspansi, untuk animasi gelombang

    while open_list:
        best_idx = 0
        for i in range(1, len(open_list)):
            if open_list[i]["f"] < open_list[best_idx]["f"]:
                best_idx = i
        current = open_list.pop(best_idx)
        cur_pos = current["pos"]

        if cur_pos in closed:
            continue
        closed.add(cur_pos)
        expanded_count += 1
        expanded_order.append((cur_pos, current["g"]))

        if cur_pos == goal:
            path = [cur_pos]
            walker = cur_pos
            while walker in came_from:
                walker = came_from[walker]
                path.append(walker)
            path.reverse()
            return SearchResult(
                path, g_score[cur_pos], expanded_count, expanded_order,
                (time.perf_counter() - t0) * 1000.0, True,
            )

        for nb_pos, step_cost in neighbors(cur_pos, grid):
            if nb_pos in closed:
                continue
            tentative_g = current["g"] + step_cost
            if nb_pos not in g_score or tentative_g < g_score[nb_pos]:
                g_score[nb_pos] = tentative_g
                came_from[nb_pos] = cur_pos
                open_list.append({"pos": nb_pos, "g": tentative_g, "f": tentative_g + h_func(nb_pos, goal)})

    return SearchResult([], math.inf, expanded_count, expanded_order, (time.perf_counter() - t0) * 1000.0, False)


# =============================================================================
# BAGIAN 3B -- FOG OF WAR: LINE OF SIGHT & POLA PENCARIAN
# =============================================================================
def bresenham_line(r0, c0, r1, c1):
    points = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return points


def has_line_of_sight(grid, a, b):
    line = bresenham_line(a[0], a[1], b[0], b[1])
    for (r, c) in line[1:-1]:
        if blocks_sight(grid[r][c]):
            return False
    return True


def can_detect_tank(grid, drone_pos, tank_pos, vision_radius):
    dist = math.hypot(drone_pos[0] - tank_pos[0], drone_pos[1] - tank_pos[1])
    if dist > vision_radius:
        return False
    return has_line_of_sight(grid, drone_pos, tank_pos)


def generate_search_pattern(grid):
    waypoints = []
    for r in range(ROWS):
        cols = range(COLS) if r % 2 == 0 else range(COLS - 1, -1, -1)
        for c in cols:
            if not is_blocked_for_drone(grid[r][c]):
                waypoints.append((r, c))
    return waypoints


def find_nearest_search_index(pattern, pos):
    best_idx, best_dist = 0, math.inf
    for i, wp in enumerate(pattern):
        d = abs(wp[0] - pos[0]) + abs(wp[1] - pos[1])
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


# =============================================================================
# BAGIAN 4 -- WARNA & KONSTANTA TAMPILAN
# =============================================================================
COLORS = {
    "bg_deep": (27, 26, 21),
    "panel": (35, 32, 25),
    "panel_border": (58, 52, 39),
    "text_cream": (232, 225, 207),
    "text_dim": (167, 158, 136),
    "accent": (224, 134, 43),
    "accent_hover": (240, 160, 75),

    "dirt": (138, 122, 84),
    "dirt_alt": (127, 111, 75),
    "tree_canopy": (75, 90, 46),
    "tree_trunk": (74, 54, 35),
    "rock": (110, 106, 98),
    "rock_dark": (84, 81, 74),
    "river": (74, 102, 112),
    "river_alt": (82, 112, 129),
    "bridge": (122, 91, 58),
    "artillery_body": (62, 65, 54),
    "artillery_barrel": (42, 44, 36),
    "camp_tent": (140, 122, 78),
    "camp_tent_dark": (110, 95, 60),
    "ruin_wall": (122, 117, 106),
    "ruin_crack": (78, 74, 66),

    "tank": (92, 107, 60),
    "tank_dark": (64, 73, 42),
    "drone": (193, 68, 58),
    "drone_dark": (140, 46, 39),
    "path": (240, 212, 139),
    "danger": (193, 68, 58),

    "search_active": (110, 168, 254),
    "search_trail": (94, 181, 199),
    "vision_ring": (224, 134, 43),

    "investigate_path": (217, 164, 65),
    "lkp_marker": (240, 160, 75),

    "scan_wave_near": (143, 217, 232),
    "scan_wave_far": (91, 63, 160),
    "scan_frontier": (224, 134, 43),

    "cost_label": (245, 240, 224),
    "cost_label_blocked": (240, 160, 75),
}

SCAN_REVEAL_PER_TICK = 2
SCAN_TICK_MS = 45
SCAN_PATH_PAUSE_MS = 550
INTRO_MOVE_TICK_MS = 320
AUTO_CHASE_TICK_MS = 450

BOARD_X, BOARD_Y = 20, 56
BOARD_W, BOARD_H = COLS * CELL, ROWS * CELL

PANEL_X = BOARD_X + BOARD_W + 24
PANEL_Y = 20
PANEL_W = 320

LEGEND_Y = BOARD_Y + BOARD_H + 12
STATUS_Y = LEGEND_Y + 190

SCREEN_W = PANEL_X + PANEL_W + 20
SCREEN_H = max(STATUS_Y + 60, PANEL_Y + 760)


def lerp_color(c_near, c_far, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c_near, c_far))


LEGEND_ITEMS = [
    ("tank", "Tank (pemain)"),
    ("drone", "Drone (pengejar)"),
    ("tree_canopy", "Pohon -- tank terhalang, drone lewat (biaya 4)"),
    ("rock", "Bebatuan -- tank terhalang, drone lewat (biaya 2)"),
    ("river", "Sungai -- tank terhalang, drone lewat (biaya 2)"),
    ("bridge", "Jembatan (biaya 1, tank & drone)"),
    ("artillery_body", "Artileri anti-udara -- terhalang tank & drone"),
    ("camp_tent", "Kamp tentara -- tank terhalang, drone lewat (biaya 6)"),
    ("ruin_wall", "Bangunan runtuh -- tank terhalang, drone lewat (biaya 2)"),
    ("path", "Jalur kejar (tahu posisi tank)"),
    ("scan_wave_near", "Tersisir -- biaya rendah (dekat)"),
    ("scan_wave_far", "Tersisir -- biaya tinggi (jauh)"),
    ("scan_frontier", "Ujung gelombang saat ini"),
    ("search_active", "Jalur pencarian (belum tahu)"),
    ("search_trail", "Jejak area tersisir"),
    ("investigate_path", "Jalur menuju posisi terakhir"),
    ("lkp_marker", "Penanda posisi terakhir tank"),
]


# =============================================================================
# BAGIAN 5 -- WIDGET UI SEDERHANA (tombol, dropdown, checkbox)
# =============================================================================
class Button:
    def __init__(self, rect, label, on_click, primary=False, enabled=True):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click
        self.primary = primary
        self.enabled = enabled

    def draw(self, screen, font):
        if not self.enabled:
            bg = (43, 39, 32)
            fg = COLORS["text_dim"]
        elif self.primary:
            bg = COLORS["accent"]
            fg = (27, 26, 21)
        else:
            bg = (43, 39, 32)
            fg = COLORS["text_cream"]
        pygame.draw.rect(screen, bg, self.rect, border_radius=7)
        pygame.draw.rect(screen, COLORS["panel_border"], self.rect, width=1, border_radius=7)
        txt = font.render(self.label, True, fg)
        screen.blit(txt, txt.get_rect(center=self.rect.center))

    def handle_click(self, pos):
        if self.enabled and self.rect.collidepoint(pos):
            self.on_click()
            return True
        return False


def wrap_text(text, font, max_width):
    words = text.split(" ")
    lines = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if font.size(trial)[0] <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# =============================================================================
# BAGIAN 5B -- MODE BATTLE: MINIMAX / ALPHA-BETA (ADVERSARIAL SEARCH)
# =============================================================================
# Begitu drone cukup dekat dengan tank, permainan berpindah dari pathfinding
# di grid (UCS/A*) ke sub-sistem battle terpisah yang state-nya sederhana
# (HP, potion, status bertahan) dan dikendalikan Minimax/Alpha-Beta.
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
ACT_POTION = "POTION"

BATTLE_MAX_HP = 100
BATTLE_START_POTIONS = 3
ATTACK_DAMAGE = 18
DEFEND_REDUCTION = 0.5          # damage masuk dikurangi 50% kalau lawan bertahan
POTION_HEAL = 30

BATTLE_WIN_SCORE = 1000.0
BATTLE_DEFAULT_DEPTH = 4        # MAX_DEPTH -- batas cutoff, branching <=3 aksi

# Bobot fungsi evaluasi heuristik (dipakai di cutoff non-terminal, BUKAN utility)
EVAL_W1 = 1.0    # selisih HP (agresivitas)
EVAL_W2 = 6.0    # selisih jumlah potion (penghargaan resource)
EVAL_W3 = 8.0    # bonus/penalti status bertahan

BATTLE_TRIGGER_DISTANCE = 1     # jarak (Manhattan) drone<->tank yang memicu battle
BATTLE_END_PAUSE_MS = 1800      # jeda menampilkan hasil sebelum kembali ke eksplorasi
BATTLE_LOG_MAX = 6


class BattleState:
    """State battle sederhana: HP, potion, status bertahan, giliran.

    s = (playerHP, npcHP, playerPotions, npcPotions,
         playerDefending, npcDefending, turnOwner, turnIndex)
    """
    __slots__ = (
        "playerHP", "npcHP", "playerPotions", "npcPotions",
        "playerDefending", "npcDefending", "turnOwner", "turnIndex",
    )

    def __init__(self, playerHP, npcHP, playerPotions, npcPotions,
                 playerDefending, npcDefending, turnOwner, turnIndex):
        self.playerHP = playerHP
        self.npcHP = npcHP
        self.playerPotions = playerPotions
        self.npcPotions = npcPotions
        self.playerDefending = playerDefending
        self.npcDefending = npcDefending
        self.turnOwner = turnOwner
        self.turnIndex = turnIndex

    @staticmethod
    def initial():
        return BattleState(
            playerHP=BATTLE_MAX_HP, npcHP=BATTLE_MAX_HP,
            playerPotions=BATTLE_START_POTIONS, npcPotions=BATTLE_START_POTIONS,
            playerDefending=False, npcDefending=False,
            turnOwner=PLAYER, turnIndex=0,
        )

    def is_terminal(self):
        """Terminal(s) = (playerHP <= 0) or (npcHP <= 0)."""
        return self.playerHP <= 0 or self.npcHP <= 0

    def legal_actions(self):
        """Branching <=3: ATTACK, DEFEND selalu ada; POTION hilang kalau stok habis."""
        actions = [ACT_ATTACK, ACT_DEFEND]
        potions = self.playerPotions if self.turnOwner == PLAYER else self.npcPotions
        if potions > 0:
            actions.append(ACT_POTION)
        return actions

    def apply_action(self, action):
        """Kembalikan BattleState BARU (tidak memutasi state ini) supaya aman
        dipakai berulang kali di pohon pencarian minimax."""
        if self.turnOwner == PLAYER:
            own_hp, opp_hp = self.playerHP, self.npcHP
            own_potions = self.playerPotions
            opp_defending = self.npcDefending
        else:
            own_hp, opp_hp = self.npcHP, self.playerHP
            own_potions = self.npcPotions
            opp_defending = self.playerDefending

        new_opp_hp = opp_hp
        new_own_hp = own_hp
        new_own_potions = own_potions

        if action == ACT_ATTACK:
            dmg = ATTACK_DAMAGE
            if opp_defending:
                dmg *= (1.0 - DEFEND_REDUCTION)
            new_opp_hp = opp_hp - dmg
        elif action == ACT_POTION:
            new_own_hp = min(BATTLE_MAX_HP, own_hp + POTION_HEAL)
            new_own_potions = own_potions - 1
        # ACT_DEFEND: tidak menyerang, tidak memulihkan -- hanya set flag di bawah

        new_own_defending = (action == ACT_DEFEND)
        new_opponent = NPC if self.turnOwner == PLAYER else PLAYER

        if self.turnOwner == PLAYER:
            return BattleState(
                playerHP=new_own_hp, npcHP=new_opp_hp,
                playerPotions=new_own_potions, npcPotions=self.npcPotions,
                playerDefending=new_own_defending, npcDefending=opp_defending,
                turnOwner=new_opponent, turnIndex=self.turnIndex + 1,
            )
        else:
            return BattleState(
                playerHP=new_opp_hp, npcHP=new_own_hp,
                playerPotions=self.playerPotions, npcPotions=new_own_potions,
                playerDefending=opp_defending, npcDefending=new_own_defending,
                turnOwner=new_opponent, turnIndex=self.turnIndex + 1,
            )


def battle_utility(state):
    """U(s) -- hanya didefinisikan di terminal SEJATI, sudut pandang NPC (MAX).

    U(s) = +WIN_SCORE - turnIndex   jika playerHP <= 0 (NPC menang)
    U(s) = -WIN_SCORE + turnIndex   jika npcHP <= 0    (player menang)

    -turnIndex / +turnIndex membuat NPC lebih menyukai kemenangan cepat
    dan kekalahan lambat -- ini asumsi desain, bukan hasil perhitungan.
    """
    if state.playerHP <= 0:
        return BATTLE_WIN_SCORE - state.turnIndex
    return -BATTLE_WIN_SCORE + state.turnIndex


def battle_eval_default(state):
    """Eval(s) -- dipakai di cutoff NON-terminal (depth limit), bukan utility.
    Heuristik "masuk akal", bukan nilai eksak."""
    score = EVAL_W1 * (state.npcHP - state.playerHP)
    score += EVAL_W2 * (state.npcPotions - state.playerPotions)
    score += EVAL_W3 * (1 if state.npcDefending else 0)
    score -= EVAL_W3 * (1 if state.playerDefending else 0)
    return score


class TreeNode:
    """Satu node di pohon pencarian minimax, dicatat untuk visualisasi debug.

    `action`  -- aksi yang membawa dari parent ke node ini (None untuk root).
    `owner`   -- 'MAX' (giliran NPC) atau 'MIN' (giliran player) di node ini.
    `kind`    -- 'internal' | 'terminal' | 'cutoff' (lihat Cutoff(s, depth) di desain).
    `pruned`  -- True kalau node ini sendiri TIDAK PERNAH dieksplorasi karena
                 alpha-beta memangkas cabang saudaranya (dipakai untuk menggambar
                 cabang yang terpotong, contoh klasik "move ordering" di laporan).
    """
    __slots__ = ("action", "depth", "owner", "value", "kind", "pruned", "children", "x", "y")

    def __init__(self, action, depth, owner=None):
        self.action = action
        self.depth = depth
        self.owner = owner
        self.value = None
        self.kind = None
        self.pruned = False
        self.children = []
        self.x = 0.0
        self.y = 0.0


class MinimaxAgent:
    """Agen NPC: memilih aksi lewat minimax / alpha-beta pruning.

    Mencatat nodes_expanded & root_action_scores -- persis yang dibutuhkan
    untuk debug overlay di BattleHUD -- dan opsional membangun `last_tree`
    (TreeNode) untuk divisualisasikan sebagai pohon (lihat TreeView/BattleHUD).
    """

    def __init__(self):
        self.nodes_expanded = 0
        self.root_action_scores = []  # [(action, score), ...] milik keputusan terakhir
        self.last_tree = None         # TreeNode root milik keputusan terakhir (untuk visualisasi)

    def _minimax(self, state, depth, alpha, beta, max_depth, use_ab, eval_fn, node):
        self.nodes_expanded += 1
        if node is not None:
            node.owner = "MAX" if state.turnOwner == NPC else "MIN"

        # Cutoff(s, depth) = Terminal(s) or (depth >= MAX_DEPTH)
        if state.is_terminal():
            val = battle_utility(state)            # terminal sejati -> utility
            if node is not None:
                node.kind = "terminal"
                node.value = val
            return val
        if depth >= max_depth:
            val = eval_fn(state)                    # cutoff karena depth -> evaluation
            if node is not None:
                node.kind = "cutoff"
                node.value = val
            return val

        if node is not None:
            node.kind = "internal"

        actions = state.legal_actions()
        if state.turnOwner == NPC:  # node MAX
            best = -math.inf
            for i, a in enumerate(actions):
                child = TreeNode(a, depth + 1) if node is not None else None
                val = self._minimax(state.apply_action(a), depth + 1, alpha, beta, max_depth, use_ab, eval_fn, child)
                if child is not None:
                    child.value = val
                    node.children.append(child)
                if val > best:
                    best = val
                if use_ab:
                    alpha = max(alpha, best)
                    if alpha >= beta:
                        if node is not None:
                            for remaining in actions[i + 1:]:
                                stub = TreeNode(remaining, depth + 1)
                                stub.pruned = True
                                node.children.append(stub)
                        break  # beta cutoff
            if node is not None:
                node.value = best
            return best
        else:  # node MIN
            best = math.inf
            for i, a in enumerate(actions):
                child = TreeNode(a, depth + 1) if node is not None else None
                val = self._minimax(state.apply_action(a), depth + 1, alpha, beta, max_depth, use_ab, eval_fn, child)
                if child is not None:
                    child.value = val
                    node.children.append(child)
                if val < best:
                    best = val
                if use_ab:
                    beta = min(beta, best)
                    if beta <= alpha:
                        if node is not None:
                            for remaining in actions[i + 1:]:
                                stub = TreeNode(remaining, depth + 1)
                                stub.pruned = True
                                node.children.append(stub)
                        break  # alpha cutoff
            if node is not None:
                node.value = best
            return best

    def choose_action(self, state, depth=BATTLE_DEFAULT_DEPTH, use_alpha_beta=True,
                       eval_fn=battle_eval_default, build_tree=True):
        self.nodes_expanded = 0
        self.root_action_scores = []
        self.last_tree = None

        alpha, beta = -math.inf, math.inf
        best_action = None
        best_score = -math.inf

        root = TreeNode(None, 0, "MAX" if state.turnOwner == NPC else "MIN") if build_tree else None

        # Catatan: di root sendiri SEMUA aksi tetap dicoba (tidak dipangkas) supaya
        # root_action_scores lengkap untuk overlay debug -- pruning hanya terjadi
        # di level yang lebih dalam.
        for a in state.legal_actions():
            child_state = state.apply_action(a)
            child_node = TreeNode(a, 1) if build_tree else None
            score = self._minimax(child_state, 1, alpha, beta, depth, use_alpha_beta, eval_fn, child_node)
            if child_node is not None:
                child_node.value = score
                root.children.append(child_node)
            self.root_action_scores.append((a, score))
            if score > best_score:
                best_score = score
                best_action = a
            if use_alpha_beta:
                alpha = max(alpha, best_score)

        if build_tree:
            root.value = best_score
            root.kind = "internal"
            self.last_tree = root

        return best_action


# -----------------------------------------------------------------------
# Visualisasi pohon pencarian (debug): layout + gambar TreeNode sebagai
# kotak-kotak yang terhubung garis, mirip diagram pohon minimax di textbook.
# Kedalaman yang digambar (vis_depth) sengaja dibatasi terpisah dari
# kedalaman pencarian sesungguhnya (bisa sampai 6) supaya tetap terbaca --
# node di batas vis_depth yang masih punya anak diberi label "+N cabang...".
# -----------------------------------------------------------------------
TREE_ROW_GAP = 60
TREE_NODE_H = 34
TREE_MARGIN_X = 40


def _tree_visible_children(node, vis_depth):
    if node.depth >= vis_depth:
        return []
    return node.children


def _tree_count_leaves(node, vis_depth):
    children = _tree_visible_children(node, vis_depth)
    if not children:
        return 1
    return sum(_tree_count_leaves(c, vis_depth) for c in children)


def _tree_best_fit_depth(root, requested_depth, max_leaves):
    """Kalau requested_depth bikin daun > max_leaves, turunkan depth tampilan
    bertahap supaya pohon tetap terbaca di lebar layar yang tersedia."""
    d = min(requested_depth, root.depth + 6)
    d = max(1, d)
    while d > 1 and _tree_count_leaves(root, d) > max_leaves:
        d -= 1
    return d


def _tree_assign_positions(node, vis_depth, node_w, gap, leaf_counter, origin_x, origin_y):
    node.y = origin_y + node.depth * TREE_ROW_GAP
    children = _tree_visible_children(node, vis_depth)
    if not children:
        node.x = origin_x + leaf_counter[0] * (node_w + gap) + node_w / 2.0
        leaf_counter[0] += 1
        return node.x
    xs = [_tree_assign_positions(c, vis_depth, node_w, gap, leaf_counter, origin_x, origin_y) for c in children]
    node.x = sum(xs) / len(xs)
    return node.x


def draw_search_tree(screen, fonts, root, vis_depth, area):
    """Gambar pohon pencarian NPC (root = TreeNode) di dalam `area` (pygame.Rect)."""
    font_small = fonts["small"]

    if root is None:
        t = fonts["ui"].render("Belum ada pohon pencarian -- tunggu NPC mengambil keputusan.", True, COLORS["text_dim"])
        screen.blit(t, (area.x + 20, area.y + 20))
        return

    avail_w = max(200, area.w - 2 * TREE_MARGIN_X)
    max_leaves = max(1, avail_w // 40)
    eff_depth = _tree_best_fit_depth(root, vis_depth, max_leaves)

    leaves = _tree_count_leaves(root, eff_depth)
    node_w = int(max(30, min(96, avail_w / leaves - 12)))
    gap = 12

    leaf_counter = [0]
    _tree_assign_positions(root, eff_depth, node_w, gap, leaf_counter, area.x + TREE_MARGIN_X, area.y + 20)

    show_text = node_w >= 40

    def draw_edges(node):
        for c in _tree_visible_children(node, eff_depth):
            col = (76, 70, 58) if not c.pruned else (58, 50, 46)
            pygame.draw.line(
                screen, col,
                (node.x, node.y + TREE_NODE_H / 2), (c.x, c.y - TREE_NODE_H / 2),
                1,
            )
            draw_edges(c)

    draw_edges(root)

    def draw_node(node):
        rect = pygame.Rect(0, 0, node_w, TREE_NODE_H)
        rect.center = (int(node.x), int(node.y))

        if node.pruned:
            bg, border = (40, 38, 33), (110, 70, 62)
        elif node.owner == "MAX":
            bg, border = (46, 39, 30), COLORS["drone"]
        else:
            bg, border = (33, 39, 29), COLORS["tank"]
        pygame.draw.rect(screen, bg, rect, border_radius=6)
        width = 2 if node.kind == "terminal" else 1
        pygame.draw.rect(screen, border, rect, width=width, border_radius=6)

        if show_text:
            label = node.action if node.action else "ROOT"
            if node.pruned:
                t1 = font_small.render(label, True, (150, 120, 110))
                t2 = font_small.render("(prune)", True, (130, 105, 98))
            else:
                val_txt = f"{node.value:.1f}" if node.value is not None else "?"
                t1 = font_small.render(label, True, COLORS["text_cream"])
                t2 = font_small.render(val_txt, True, COLORS["text_dim"])
            screen.blit(t1, t1.get_rect(center=(rect.centerx, rect.centery - 7)))
            screen.blit(t2, t2.get_rect(center=(rect.centerx, rect.centery + 7)))

        if not node.pruned and node.depth == eff_depth and node.children:
            more = font_small.render(f"+{len(node.children)} cabang...", True, COLORS["text_dim"])
            screen.blit(more, (rect.centerx - more.get_width() // 2, rect.bottom + 2))

        for c in _tree_visible_children(node, eff_depth):
            draw_node(c)

    draw_node(root)

    if eff_depth < vis_depth:
        note = font_small.render(
            f"(kedalaman tampil diturunkan otomatis ke {eff_depth} dari {vis_depth} agar tetap terbaca)",
            True, COLORS["text_dim"],
        )
        screen.blit(note, (area.x + TREE_MARGIN_X, area.bottom - 18))


class BattleController:
    """Mesin giliran mode battle:
    input player -> apply -> cek terminal ->
    giliran NPC lewat MinimaxAgent -> apply -> cek terminal -> ulang."""

    def __init__(self, game):
        self.game = game
        self.state = BattleState.initial()
        self.agent = MinimaxAgent()
        self.depth = BATTLE_DEFAULT_DEPTH
        self.use_alpha_beta = True
        self.awaiting_player = True
        self.log = ["Battle dimulai! Pilih aksi: [1] Serang  [2] Bertahan  [3] Potion"]
        self.result_text = None
        self.end_deadline = None
        self.last_npc_action = None

        self.show_tree = False          # toggle visualisasi pohon pencarian (debug)
        self.tree_vis_depth = 3         # kedalaman pohon yang DIGAMBAR (terpisah dari self.depth)

    def _log(self, text):
        self.log.append(text)
        if len(self.log) > BATTLE_LOG_MAX:
            self.log.pop(0)

    def handle_keydown(self, key):
        if self.result_text is not None:
            return  # sedang menampilkan hasil, tunggu auto-kembali ke eksplorasi
        if not self.awaiting_player:
            return

        action_map = {
            pygame.K_1: ACT_ATTACK, pygame.K_KP1: ACT_ATTACK,
            pygame.K_2: ACT_DEFEND, pygame.K_KP2: ACT_DEFEND,
            pygame.K_3: ACT_POTION, pygame.K_KP3: ACT_POTION,
        }
        action = action_map.get(key)
        if action is None:
            return
        if action == ACT_POTION and self.state.playerPotions <= 0:
            self._log("Potion habis!")
            return

        self.player_act(action)

    def player_act(self, action):
        self.state = self.state.apply_action(action)
        self._log(f"Player: {action}  (HP {int(max(0, self.state.playerHP))} / Potion {self.state.playerPotions})")
        self.awaiting_player = False

        if self.state.is_terminal():
            self._finish()
            return

        self.npc_act()

    def npc_act(self):
        action = self.agent.choose_action(
            self.state, depth=self.depth, use_alpha_beta=self.use_alpha_beta,
            eval_fn=battle_eval_default,
        )
        self.last_npc_action = action
        self.state = self.state.apply_action(action)
        self._log(f"NPC   : {action}  (HP {int(max(0, self.state.npcHP))} / Potion {self.state.npcPotions})")

        if self.state.is_terminal():
            self._finish()
            return

        self.awaiting_player = True

    def _finish(self):
        self.awaiting_player = False
        if self.state.playerHP <= 0:
            self.result_text = "NPC MENANG -- tank hancur!"
        else:
            self.result_text = "PLAYER MENANG -- drone dilumpuhkan!"
        self._log(self.result_text)
        self.end_deadline = pygame.time.get_ticks() + BATTLE_END_PAUSE_MS

    def toggle_alpha_beta(self):
        self.use_alpha_beta = not self.use_alpha_beta

    def change_depth(self, delta):
        self.depth = max(1, min(6, self.depth + delta))

    def toggle_tree(self):
        self.show_tree = not self.show_tree

    def change_tree_depth(self, delta):
        self.tree_vis_depth = max(1, min(self.depth, self.tree_vis_depth + delta))

    def update(self, now_ms):
        if self.end_deadline is not None and now_ms >= self.end_deadline:
            self.end_deadline = None
            self.game.end_battle()

    def draw(self, screen, fonts):
        if self.show_tree:
            BattleHUD.draw_tree(screen, fonts, self)
        else:
            BattleHUD.draw(screen, fonts, self)


class BattleHUD:
    """Render panel debug overlay (aksi NPC & skornya, node count) + HP bar,
    terpisah dari HUD mode eksplorasi supaya tanggung jawab tidak campur."""

    @staticmethod
    def _hp_bar(screen, x, y, w, h, hp, max_hp, color_fg, font):
        pygame.draw.rect(screen, (22, 21, 16), (x, y, w, h), border_radius=5)
        ratio = max(0.0, min(1.0, hp / max_hp))
        fill_w = int((w - 4) * ratio)
        if fill_w > 0:
            pygame.draw.rect(screen, color_fg, (x + 2, y + 2, fill_w, h - 4), border_radius=4)
        pygame.draw.rect(screen, COLORS["panel_border"], (x, y, w, h), width=1, border_radius=5)
        txt = font.render(f"{max(0, int(hp))} / {int(max_hp)}", True, COLORS["text_cream"])
        screen.blit(txt, txt.get_rect(center=(x + w // 2, y + h // 2)))

    @staticmethod
    def draw(screen, fonts, controller):
        font_ui_bold = fonts["ui_bold"]
        font_small = fonts["small"]
        font_title = fonts["title"]

        area = pygame.Rect(BOARD_X, BOARD_Y, BOARD_W + 24 + PANEL_W, BOARD_H)
        pygame.draw.rect(screen, COLORS["panel"], area, border_radius=12)
        pygame.draw.rect(screen, COLORS["panel_border"], area, width=1, border_radius=12)

        title = font_title.render("MODE BATTLE -- Minimax / Alpha-Beta", True, COLORS["text_cream"])
        screen.blit(title, (area.x + 20, area.y + 16))

        state = controller.state
        left_x = area.x + 20
        right_x = area.x + area.w // 2 + 20
        bars_y = area.y + 60

        lbl_p = font_ui_bold.render("TANK (Player)", True, COLORS["tank"])
        screen.blit(lbl_p, (left_x, bars_y))
        BattleHUD._hp_bar(screen, left_x, bars_y + 20, 260, 26, state.playerHP, BATTLE_MAX_HP, COLORS["tank"], font_small)
        pot_p = font_small.render(
            f"Potion: {state.playerPotions}  {'| BERTAHAN' if state.playerDefending else ''}",
            True, COLORS["text_dim"],
        )
        screen.blit(pot_p, (left_x, bars_y + 52))

        lbl_n = font_ui_bold.render("DRONE (NPC)", True, COLORS["drone"])
        screen.blit(lbl_n, (right_x, bars_y))
        BattleHUD._hp_bar(screen, right_x, bars_y + 20, 260, 26, state.npcHP, BATTLE_MAX_HP, COLORS["drone"], font_small)
        pot_n = font_small.render(
            f"Potion: {state.npcPotions}  {'| BERTAHAN' if state.npcDefending else ''}",
            True, COLORS["text_dim"],
        )
        screen.blit(pot_n, (right_x, bars_y + 52))

        instr_y = bars_y + 90
        if controller.result_text:
            txt = font_ui_bold.render(controller.result_text, True, COLORS["accent"])
        elif controller.awaiting_player:
            txt = font_ui_bold.render("Giliranmu -- [1] Serang  [2] Bertahan  [3] Potion", True, COLORS["text_cream"])
        else:
            txt = font_ui_bold.render("Drone berpikir...", True, COLORS["text_dim"])
        screen.blit(txt, (left_x, instr_y))

        log_y = instr_y + 30
        for line in controller.log[-BATTLE_LOG_MAX:]:
            t = font_small.render(line, True, COLORS["text_cream"])
            screen.blit(t, (left_x, log_y))
            log_y += 16

        dbg_x = right_x
        dbg_y = instr_y
        dbg_hdr = font_ui_bold.render(
            f"[Debug NPC] {'alpha-beta' if controller.use_alpha_beta else 'minimax'}  depth={controller.depth}",
            True, COLORS["text_dim"],
        )
        screen.blit(dbg_hdr, (dbg_x, dbg_y))
        dbg_y += 20
        if controller.agent.root_action_scores:
            for action, score in controller.agent.root_action_scores:
                picked = action == controller.last_npc_action
                color = COLORS["accent_hover"] if picked else COLORS["text_cream"]
                marker = " <- dipilih" if picked else ""
                line = f"{action:<8}: skor {score:7.1f}{marker}"
                t = font_small.render(line, True, color)
                screen.blit(t, (dbg_x, dbg_y))
                dbg_y += 16
            dbg_y += 4
            node_txt = font_small.render(f"Node diekspansi: {controller.agent.nodes_expanded}", True, COLORS["text_dim"])
            screen.blit(node_txt, (dbg_x, dbg_y))
        else:
            t = font_small.render("(belum ada keputusan NPC)", True, COLORS["text_dim"])
            screen.blit(t, (dbg_x, dbg_y))

        ctrl_y = area.bottom - 34
        ctrl_txt = font_small.render(
            "[B] toggle alpha-beta   [,] / [.] ubah depth (1-6)   [T] lihat pohon pencarian",
            True, COLORS["text_dim"],
        )
        screen.blit(ctrl_txt, (left_x, ctrl_y))

    @staticmethod
    def draw_tree(screen, fonts, controller):
        """Overlay visualisasi pohon pencarian minimax/alpha-beta milik
        keputusan NPC yang terakhir diambil -- alat bantu debug."""
        font_title = fonts["title"]
        font_ui_bold = fonts["ui_bold"]
        font_small = fonts["small"]

        area = pygame.Rect(BOARD_X, BOARD_Y, BOARD_W + 24 + PANEL_W, BOARD_H)
        pygame.draw.rect(screen, COLORS["panel"], area, border_radius=12)
        pygame.draw.rect(screen, COLORS["panel_border"], area, width=1, border_radius=12)

        title = font_title.render("Pohon Pencarian NPC (Debug)", True, COLORS["text_cream"])
        screen.blit(title, (area.x + 20, area.y + 12))

        algo_label = "Alpha-Beta Pruning" if controller.use_alpha_beta else "Minimax murni"
        info = font_ui_bold.render(
            f"{algo_label}  |  depth pencarian={controller.depth}  |  tampil s.d. depth={controller.tree_vis_depth}  |  "
            f"node diekspansi={controller.agent.nodes_expanded}",
            True, COLORS["text_dim"],
        )
        screen.blit(info, (area.x + 20, area.y + 34))

        legend_y = area.y + 34
        legend_x = area.right - 300
        for label, color in (("MAX (NPC)", COLORS["drone"]), ("MIN (Player)", COLORS["tank"]), ("dipangkas", (110, 70, 62))):
            pygame.draw.rect(screen, color, (legend_x, legend_y + 3, 10, 10), border_radius=2)
            t = font_small.render(label, True, COLORS["text_dim"])
            screen.blit(t, (legend_x + 16, legend_y))
            legend_x += 16 + t.get_width() + 14

        tree_area = pygame.Rect(area.x, area.y + 58, area.w, area.h - 58 - 26)
        draw_search_tree(screen, fonts, controller.agent.last_tree, controller.tree_vis_depth, tree_area)

        ctrl_txt = font_small.render(
            "[T] kembali ke HUD battle   [ [ ] ubah kedalaman tampil   [1/2/3] tetap bisa dipakai untuk beraksi",
            True, COLORS["text_dim"],
        )
        screen.blit(ctrl_txt, (area.x + 20, area.bottom - 20))


# =============================================================================
# BAGIAN 6 -- KELAS GAME UTAMA
# =============================================================================
class Game:
    def __init__(self, screen):
        self.screen = screen
        self.font_small = pygame.font.SysFont("consolas,couriernew,monospace", 12)
        self.font_ui = pygame.font.SysFont("segoeui,arial,sans-serif", 13)
        self.font_ui_bold = pygame.font.SysFont("segoeui,arial,sans-serif", 13, bold=True)
        self.font_title = pygame.font.SysFont("segoeui,arial,sans-serif", 20, bold=True)
        self.font_sub = pygame.font.SysFont("segoeui,arial,sans-serif", 12)
        self.font_cost = pygame.font.SysFont("arial", 10, bold=True)
        self.font_legend = pygame.font.SysFont("segoeui,arial,sans-serif", 11)
        self.fonts = {
            "small": self.font_small,
            "ui": self.font_ui,
            "ui_bold": self.font_ui_bold,
            "title": self.font_title,
        }

        self.algo = "astar"          # 'ucs' | 'astar'
        self.heuristic = "manhattan"  # 'manhattan' | 'euclidean' | 'chebyshev'
        self.vision_radius = 4
        self.auto_chase = False

        self.mode = "explore"        # 'explore' | 'battle'
        self.battle_controller = None

        self.stats_lines = ["Belum ada pencarian dijalankan."]
        self.status_text = "Drone melakukan pemindaian awal untuk menemukan posisi awal tank..."
        self.status_caught = False
        self.drone_state_text = "MEMINDAI (mencari posisi awal tank)"

        self.experiment_lines = []
        self.experiment_note = ""

        self.initial_drone_pos, self.initial_tank_pos = pick_random_start_positions()
        self.tank = self.initial_tank_pos
        self.drone = self.initial_drone_pos
        self.tank_dir = (0, -1)
        self.grid = generate_map(self.drone, self.tank)
        self.last_path = []
        self.game_over = False

        self.drone_state = "searching"
        self.search_pattern = generate_search_pattern(self.grid)
        self.search_index = 0
        self.visited_cells = set()
        self.last_known_pos = None

        self.intro_phase = None
        self.intro_tank_pos = None
        self.scan_expanded_order = []
        self.scan_revealed_count = 0
        self.scan_max_g = 0
        self.scan_next_tick = None
        self.scan_pause_deadline = None
        self.intro_path = []
        self.intro_path_index = 0
        self.intro_move_next_tick = None

        self.auto_next_tick = None

        self._build_ui()
        self.start_intro_sequence()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        x = PANEL_X + 16
        w = PANEL_W - 32
        y = PANEL_Y + 40

        self.btn_algo_ucs = Button((x, y, w // 2 - 4, 28), "UCS", lambda: self.set_algo("ucs"))
        self.btn_algo_astar = Button((x + w // 2 + 4, y, w // 2 - 4, 28), "A*", lambda: self.set_algo("astar"))
        y += 40

        heur_w = w // 3 - 4
        self.btn_heur_manhattan = Button((x, y, heur_w, 26), "Manhattan", lambda: self.set_heuristic("manhattan"))
        self.btn_heur_euclidean = Button((x + heur_w + 6, y, heur_w, 26), "Euclidean", lambda: self.set_heuristic("euclidean"))
        self.btn_heur_chebyshev = Button((x + 2 * (heur_w + 6), y, heur_w, 26), "Chebyshev", lambda: self.set_heuristic("chebyshev"))
        y += 38

        vis_w = w // 5 - 4
        self.vision_options = [2, 3, 4, 6, 8]
        self.btn_visions = []
        for i, v in enumerate(self.vision_options):
            btn = Button((x + i * (vis_w + 5), y, vis_w, 26), str(v), (lambda vv=v: self.set_vision(vv)))
            self.btn_visions.append(btn)
        y += 38

        self.chk_auto_rect = pygame.Rect(x, y, 18, 18)
        self.btn_auto_chase = Button((x, y - 2, w, 24), "", self.toggle_auto_chase)
        y += 34

        y += 60  # ruang untuk status drone (digambar manual)

        self.btn_random_map = Button((x, y, w // 2 - 4, 32), "Acak Medan", self.on_random_map, primary=True)
        self.btn_reset_pos = Button((x + w // 2 + 4, y, w // 2 - 4, 32), "Reset Posisi", self.on_reset_pos)
        y += 46

        self.stats_panel_y = y
        y += 100

        self.btn_experiment = Button((x, y, w, 28), "Jalankan Eksperimen", self.on_experiment)
        self.experiment_y = y + 36

        self.panel_x = x
        self.panel_w = w

    def set_algo(self, algo):
        self.algo = algo

    def set_heuristic(self, h):
        if self.algo == "astar":
            self.heuristic = h

    def set_vision(self, v):
        self.vision_radius = v

    def toggle_auto_chase(self):
        self.auto_chase = not self.auto_chase
        self.auto_next_tick = pygame.time.get_ticks() + AUTO_CHASE_TICK_MS if self.auto_chase else None

    def on_random_map(self):
        picked_drone, picked_tank = pick_random_start_positions()
        self.initial_drone_pos, self.initial_tank_pos = picked_drone, picked_tank
        self.tank = self.initial_tank_pos
        self.drone = self.initial_drone_pos
        self.tank_dir = (0, -1)
        self.grid = generate_map(self.drone, self.tank)
        self.last_path = []
        self.game_over = False
        self.reset_drone_knowledge()
        self.start_intro_sequence()

    def on_reset_pos(self):
        self.tank = self.initial_tank_pos
        self.drone = self.initial_drone_pos
        self.tank_dir = (0, -1)
        self.last_path = []
        self.game_over = False
        self.reset_drone_knowledge()
        self.start_intro_sequence()

    def on_experiment(self):
        combos = [
            ("UCS", "ucs", "zero"),
            ("A* + Manhattan", "astar", "manhattan"),
            ("A* + Euclidean", "astar", "euclidean"),
            ("A* + Chebyshev", "astar", "chebyshev"),
        ]
        results = []
        for label, algo, heur in combos:
            r = search(self.grid, self.drone, self.tank, algo, heur)
            results.append((label, r))

        min_expanded = min(r.expanded_count for _, r in results)
        self.experiment_lines = []
        for label, r in results:
            best = r.expanded_count == min_expanded
            cost_txt = str(r.cost) if r.found else "-"
            self.experiment_lines.append((label, r.expanded_count, cost_txt, f"{r.time_ms:.2f}", best))

        best_label = next(label for label, r in results if r.expanded_count == min_expanded)
        self.experiment_note = (
            f'Pada posisi ini, "{best_label}" mengekspansi node paling sedikit ({min_expanded}). '
            "Karena gerakan hanya 4 arah, heuristik Manhattan biasanya menang karena nilainya paling "
            "mendekati biaya asli tanpa pernah melebihi (tetap admissible), sehingga A* memangkas "
            "lebih banyak cabang pencarian dibanding UCS, Euclidean, atau Chebyshev. "
            "Catatan: eksperimen ini membandingkan jalur KEJAR drone->tank secara langsung (seolah "
            "tank sudah terdeteksi), bukan pola sisir yang dipakai selagi posisi tank belum diketahui."
        )

    # ------------------------------------------------------------------
    # MODE BATTLE
    # ------------------------------------------------------------------
    def check_battle_trigger(self):
        """Cek jarak tank<->drone setiap giliran; kalau <= ambang, masuk battle mode."""
        if self.mode != "explore" or self.game_over or self.intro_phase:
            return False
        dist = abs(self.drone[0] - self.tank[0]) + abs(self.drone[1] - self.tank[1])
        if dist <= BATTLE_TRIGGER_DISTANCE:
            self.start_battle()
            return True
        return False

    def start_battle(self):
        self.mode = "battle"
        self.auto_chase = False
        self.auto_next_tick = None
        self.intro_phase = None
        self.last_path = []
        self.battle_controller = BattleController(self)
        self.status_text = "Drone cukup dekat -- battle dimulai!"
        self.status_caught = False
        self.drone_state_text = "BATTLE -- adversarial search (Minimax/Alpha-Beta)"

    def end_battle(self):
        """Battle selesai (menang/kalah) -> balik ke mode eksplorasi dengan
        posisi tank/drone direset agak menjauh supaya tidak langsung battle lagi."""
        self.mode = "explore"
        self.battle_controller = None
        self.initial_drone_pos, self.initial_tank_pos = pick_random_start_positions()
        self.tank = self.initial_tank_pos
        self.drone = self.initial_drone_pos
        self.tank_dir = (0, -1)
        self.grid = generate_map(self.drone, self.tank)
        self.last_path = []
        self.game_over = False
        self.reset_drone_knowledge()
        self.start_intro_sequence()

    # ------------------------------------------------------------------
    # SEKUENS PEMINDAIAN AWAL
    # ------------------------------------------------------------------
    def reset_drone_knowledge(self):
        self.drone_state = "searching"
        self.search_pattern = generate_search_pattern(self.grid)
        self.search_index = 0
        self.visited_cells = set()
        self.last_known_pos = None
        self.drone_state_text = "MENYISIR (belum tahu posisi tank)"

    def start_intro_sequence(self):
        self.scan_next_tick = None
        self.scan_pause_deadline = None
        self.intro_move_next_tick = None
        self.auto_next_tick = pygame.time.get_ticks() + AUTO_CHASE_TICK_MS if self.auto_chase else None

        self.intro_tank_pos = self.tank
        self.last_path = []
        self.last_known_pos = None

        scan_result = search(self.grid, self.drone, self.intro_tank_pos, self.algo, self.heuristic)
        self.scan_expanded_order = scan_result.expanded_order
        self.scan_revealed_count = 0
        self.scan_max_g = max((g for _, g in self.scan_expanded_order), default=0)
        self.intro_phase = "scanning"
        self.drone_state = "scanning"

        algo_label = "UCS" if self.algo == "ucs" else f"A* ({self.heuristic})"
        self.drone_state_text = "MEMINDAI -- drone menyapu seluruh medan untuk menemukan tank"
        self.status_text = f"Pemindaian awal dimulai: drone mencari posisi awal tank ke seluruh medan ({algo_label})..."
        self.status_caught = False
        self.render_stats(scan_result)

        if not self.scan_expanded_order:
            self.finish_scan_stage(scan_result)
        else:
            self.scan_next_tick = pygame.time.get_ticks() + SCAN_TICK_MS
        self._pending_scan_result = scan_result

    def finish_scan_stage(self, scan_result):
        self.scan_expanded_order = []
        self.scan_revealed_count = 0
        self.last_path = scan_result.path if scan_result.found else []
        self.drone_state_text = "TANK DITEMUKAN (posisi awal) -- menyusun jalur pergerakan"
        self.status_text = "Posisi awal tank ditemukan. Drone menghitung jalur menuju sana..."
        self.scan_pause_deadline = pygame.time.get_ticks() + SCAN_PATH_PAUSE_MS

    def start_intro_move(self):
        move_result = search(self.grid, self.drone, self.intro_tank_pos, self.algo, self.heuristic)
        self.render_stats(move_result)
        self.intro_path = move_result.path
        self.intro_path_index = 0
        self.intro_phase = "moving"
        self.drone_state = "moving"
        self.last_path = self.intro_path

        self.drone_state_text = "MENUJU POSISI AWAL TANK -- mengabaikan pergerakan tank untuk sementara"
        if move_result.found:
            self.status_text = (
                "Drone bergerak menuju posisi awal tank hasil pemindaian, mencari jalur alternatif "
                "kalau ada halangan (mengabaikan pergerakan tank saat ini)..."
            )
        else:
            self.status_text = "Drone tidak menemukan jalur menuju posisi awal tank (terkurung terrain) -- pemindaian awal dibatalkan."

        if not move_result.found or len(self.intro_path) <= 1:
            self.finish_intro_sequence()
        else:
            self.intro_move_next_tick = pygame.time.get_ticks() + INTRO_MOVE_TICK_MS

    def step_intro_move(self):
        self.intro_path_index += 1
        if self.intro_path_index >= len(self.intro_path):
            self.intro_move_next_tick = None
            self.finish_intro_sequence()
            return
        self.drone = self.intro_path[self.intro_path_index]
        self.visited_cells.add(self.drone)
        self.intro_move_next_tick = pygame.time.get_ticks() + INTRO_MOVE_TICK_MS

    def finish_intro_sequence(self):
        self.intro_phase = None
        self.last_path = []

        if not self.game_over and self.check_battle_trigger():
            return

        self.last_known_pos = None
        self.search_index = find_nearest_search_index(self.search_pattern, self.drone)
        self.drone_state = "searching"
        self.drone_state_text = "MENYISIR -- tank tidak ada di posisi awal, radar drone kini aktif"
        self.status_text = "Tank sudah berpindah dari posisi awal. Radar jarak pendek drone kini aktif dan ia mulai menyisir dari sini..."
        self.status_caught = False

    # ------------------------------------------------------------------
    # GILIRAN TANK -> DRONE
    # ------------------------------------------------------------------
    def try_move_tank(self, dr, dc):
        if self.game_over or self.intro_phase == "scanning" or self.mode == "battle":
            return
        nr, nc = self.tank[0] + dr, self.tank[1] + dc
        if not (0 <= nr < ROWS and 0 <= nc < COLS):
            return
        if is_blocked(self.grid[nr][nc]):
            return
        self.tank = (nr, nc)
        self.tank_dir = (dr, dc)

        if self.intro_phase == "moving":
            self.check_battle_trigger()
            return  # drone mengabaikan pergerakan tank selama fase ini

        self.drone_turn()

    def continue_search_pattern(self):
        if not self.search_pattern:
            return SearchResult([], math.inf, 0, [], 0.0, False)

        attempts = 0
        while attempts < len(self.search_pattern):
            target = self.search_pattern[self.search_index]
            if self.drone == target:
                self.visited_cells.add(self.drone)
                self.search_index = (self.search_index + 1) % len(self.search_pattern)
                attempts += 1
                continue

            result = search(self.grid, self.drone, target, self.algo, self.heuristic)
            if result.found:
                return result

            self.search_index = (self.search_index + 1) % len(self.search_pattern)
            attempts += 1

        return SearchResult([], math.inf, 0, [], 0.0, False)

    def drone_turn(self):
        if self.game_over or self.intro_phase or self.mode != "explore":
            return

        if self.check_battle_trigger():
            return

        detected = can_detect_tank(self.grid, self.drone, self.tank, self.vision_radius)

        if detected:
            self.drone_state = "chasing"
            self.last_known_pos = self.tank
            result = search(self.grid, self.drone, self.tank, self.algo, self.heuristic)
        elif self.last_known_pos and self.drone != self.last_known_pos:
            self.drone_state = "investigating"
            result = search(self.grid, self.drone, self.last_known_pos, self.algo, self.heuristic)
            if not result.found:
                self.last_known_pos = None
                self.drone_state = "searching"
                result = self.continue_search_pattern()
        else:
            if self.last_known_pos:
                self.last_known_pos = None
                self.search_index = find_nearest_search_index(self.search_pattern, self.drone)
            self.drone_state = "searching"
            result = self.continue_search_pattern()

        self.last_path = result.path
        self.render_stats(result)

        if detected:
            self.drone_state_text = "MENGEJAR -- tank terdeteksi dalam radius pandang"
        elif self.drone_state == "investigating":
            self.drone_state_text = "MENYELIDIKI -- menuju posisi terakhir tank yang diketahui"
        else:
            self.drone_state_text = "MENYISIR -- tank belum terdeteksi"

        if not result.found:
            if detected:
                self.status_text = "Drone tidak menemukan jalur menuju tank saat ini."
            elif self.drone_state == "investigating":
                self.status_text = "Drone tidak menemukan jalur menuju posisi terakhir tank."
            else:
                self.status_text = "Drone tidak menemukan jalur menuju titik sisiran berikutnya."
            return

        if len(result.path) > 1:
            self.drone = result.path[1]
        self.visited_cells.add(self.drone)

        if self.check_battle_trigger():
            return
        elif detected:
            self.status_text = "Tank terdeteksi! Drone mengejar dan terus menghitung ulang jalurnya..."
            self.status_caught = False
        elif self.drone_state == "investigating":
            self.status_text = "Kontak terputus. Drone mengingat posisi terakhir tank dan menuju ke sana..."
            self.status_caught = False
        else:
            self.status_text = "Tidak ada tanda-tanda di posisi terakhir. Drone melanjutkan pola pencarian..."
            self.status_caught = False

    def render_stats(self, result):
        algo_label = "UCS" if self.algo == "ucs" else f"A* ({self.heuristic})"
        cost_txt = str(result.cost) if result.found else "-"
        self.stats_lines = [
            f"Algoritma       : {algo_label}",
            f"Jalur ditemukan : {'ya' if result.found else 'tidak'}",
            f"Biaya jalur     : {cost_txt}",
            f"Node diekspansi : {result.expanded_count}",
            f"Waktu komputasi : {result.time_ms:.3f} ms",
        ]

    # ------------------------------------------------------------------
    # UPDATE PER FRAME (mengganti setInterval/setTimeout JS)
    # ------------------------------------------------------------------
    def update(self, now_ms):
        if self.mode == "battle":
            self.battle_controller.update(now_ms)
            return

        if self.intro_phase == "scanning" and self.scan_next_tick is not None and now_ms >= self.scan_next_tick:
            self.scan_revealed_count = min(len(self.scan_expanded_order), self.scan_revealed_count + SCAN_REVEAL_PER_TICK)
            if self.scan_revealed_count >= len(self.scan_expanded_order):
                self.scan_next_tick = None
                self.finish_scan_stage(self._pending_scan_result)
            else:
                self.scan_next_tick = now_ms + SCAN_TICK_MS

        if self.scan_pause_deadline is not None and now_ms >= self.scan_pause_deadline:
            self.scan_pause_deadline = None
            self.start_intro_move()

        if self.intro_phase == "moving" and self.intro_move_next_tick is not None and now_ms >= self.intro_move_next_tick:
            self.step_intro_move()

        if self.auto_chase and not self.intro_phase and self.auto_next_tick is not None and now_ms >= self.auto_next_tick:
            self.drone_turn()
            self.auto_next_tick = now_ms + AUTO_CHASE_TICK_MS

    # ------------------------------------------------------------------
    # EVENT
    # ------------------------------------------------------------------
    def handle_keydown(self, key):
        if self.mode == "battle":
            if key == pygame.K_b:
                self.battle_controller.toggle_alpha_beta()
            elif key == pygame.K_COMMA:
                self.battle_controller.change_depth(-1)
            elif key == pygame.K_PERIOD:
                self.battle_controller.change_depth(1)
            elif key == pygame.K_t:
                self.battle_controller.toggle_tree()
            elif key == pygame.K_LEFTBRACKET:
                self.battle_controller.change_tree_depth(-1)
            elif key == pygame.K_RIGHTBRACKET:
                self.battle_controller.change_tree_depth(1)
            else:
                self.battle_controller.handle_keydown(key)
            return

        mapping = {
            pygame.K_UP: (-1, 0), pygame.K_w: (-1, 0),
            pygame.K_DOWN: (1, 0), pygame.K_s: (1, 0),
            pygame.K_LEFT: (0, -1), pygame.K_a: (0, -1),
            pygame.K_RIGHT: (0, 1), pygame.K_d: (0, 1),
        }
        if key in mapping:
            dr, dc = mapping[key]
            self.try_move_tank(dr, dc)

    def handle_click(self, pos):
        if self.mode == "battle":
            return
        buttons = [
            self.btn_algo_ucs, self.btn_algo_astar,
            self.btn_heur_manhattan, self.btn_heur_euclidean, self.btn_heur_chebyshev,
            *self.btn_visions,
            self.btn_auto_chase,
            self.btn_random_map, self.btn_reset_pos,
            self.btn_experiment,
        ]
        for btn in buttons:
            if btn.handle_click(pos):
                return

    # ------------------------------------------------------------------
    # RENDER
    # ------------------------------------------------------------------
    def draw(self):
        screen = self.screen
        screen.fill(COLORS["bg_deep"])

        title = self.font_title.render("Tank vs Drone -- UCS/A* + Battle Minimax -- Simulasi Ukraina", True, COLORS["text_cream"])
        screen.blit(title, (BOARD_X, 18))

        if self.mode == "battle":
            self.battle_controller.draw(screen, self.fonts)
        else:
            self.draw_board()
            self.draw_legend()
            self.draw_status()
            self.draw_panel()

        pygame.display.flip()

    def draw_board(self):
        screen = self.screen
        board_rect = pygame.Rect(BOARD_X, BOARD_Y, BOARD_W, BOARD_H)
        pygame.draw.rect(screen, COLORS["dirt"], board_rect)

        for r in range(ROWS):
            for c in range(COLS):
                self.draw_cell(self.grid[r][c], r, c)

        # jejak area tersisir
        trail_surf = pygame.Surface((BOARD_W, BOARD_H), pygame.SRCALPHA)
        for (r, c) in self.visited_cells:
            pygame.draw.rect(
                trail_surf, (*COLORS["search_trail"], 56),
                (c * CELL + 6, r * CELL + 6, CELL - 12, CELL - 12),
            )
        screen.blit(trail_surf, (BOARD_X, BOARD_Y))

        # gelombang ekspansi UCS/A* saat pemindaian awal
        if self.drone_state == "scanning" and self.scan_expanded_order and self.scan_revealed_count > 0:
            max_g = self.scan_max_g if self.scan_max_g > 0 else 1
            wave_surf = pygame.Surface((BOARD_W, BOARD_H), pygame.SRCALPHA)
            for i in range(self.scan_revealed_count):
                (r, c), g = self.scan_expanded_order[i]
                t = max(0.0, min(1.0, g / max_g))
                color = lerp_color(COLORS["scan_wave_near"], COLORS["scan_wave_far"], t)
                pygame.draw.rect(wave_surf, (*color, 128), (c * CELL + 3, r * CELL + 3, CELL - 6, CELL - 6))
            screen.blit(wave_surf, (BOARD_X, BOARD_Y))

            frontier_start = max(0, self.scan_revealed_count - SCAN_REVEAL_PER_TICK)
            for i in range(frontier_start, self.scan_revealed_count):
                (r, c), g = self.scan_expanded_order[i]
                pygame.draw.rect(
                    screen, COLORS["scan_frontier"],
                    (BOARD_X + c * CELL + 2, BOARD_Y + r * CELL + 2, CELL - 4, CELL - 4),
                    width=2,
                )

        # lingkaran radius pandang
        if self.drone_state in ("searching", "investigating"):
            vis_surf = pygame.Surface((BOARD_W, BOARD_H), pygame.SRCALPHA)
            dcx = self.drone[1] * CELL + CELL // 2
            dcy = self.drone[0] * CELL + CELL // 2
            pygame.draw.circle(vis_surf, (*COLORS["vision_ring"], 90), (dcx, dcy), self.vision_radius * CELL, width=2)
            screen.blit(vis_surf, (BOARD_X, BOARD_Y))

        # penanda posisi terakhir tank diketahui
        if self.last_known_pos:
            r, c = self.last_known_pos
            cx = BOARD_X + c * CELL + CELL // 2
            cy = BOARD_Y + r * CELL + CELL // 2
            self._draw_dashed_circle(screen, (cx, cy), int(CELL * 0.4), COLORS["lkp_marker"])

        # jalur aktif
        path_color_key = "search_active"
        if self.drone_state in ("chasing", "moving"):
            path_color_key = "path"
        elif self.drone_state == "investigating":
            path_color_key = "investigate_path"
        path_surf = pygame.Surface((BOARD_W, BOARD_H), pygame.SRCALPHA)
        for (r, c) in self.last_path:
            pygame.draw.rect(path_surf, (*COLORS[path_color_key], 153), (c * CELL + 6, r * CELL + 6, CELL - 12, CELL - 12))
        screen.blit(path_surf, (BOARD_X, BOARD_Y))

        self.draw_tank()
        self.draw_drone()

        pygame.draw.rect(screen, COLORS["panel_border"], board_rect, width=1, border_radius=10)

    def _draw_dashed_circle(self, screen, center, radius, color, dash_len=4, gap_len=3, width=2):
        circumference = 2 * math.pi * radius
        n = max(8, int(circumference / (dash_len + gap_len)))
        for i in range(n):
            a0 = (i / n) * 2 * math.pi
            a1 = a0 + (dash_len / (dash_len + gap_len)) * (2 * math.pi / n)
            p0 = (center[0] + radius * math.cos(a0), center[1] + radius * math.sin(a0))
            p1 = (center[0] + radius * math.cos(a1), center[1] + radius * math.sin(a1))
            pygame.draw.line(screen, color, p0, p1, width)

    def draw_cell(self, terrain, r, c):
        screen = self.screen
        x = BOARD_X + c * CELL
        y = BOARD_Y + r * CELL
        checker = (r + c) % 2 == 0
        base = COLORS["dirt"] if checker else COLORS["dirt_alt"]
        pygame.draw.rect(screen, base, (x, y, CELL, CELL))

        if terrain == TREE:
            pygame.draw.rect(screen, COLORS["tree_trunk"], (x + CELL // 2 - 2, y + int(CELL * 0.55), 4, int(CELL * 0.4)))
            pygame.draw.circle(screen, COLORS["tree_canopy"], (x + CELL // 2, y + int(CELL * 0.42)), int(CELL * 0.36))
        elif terrain == ROCK:
            pygame.draw.ellipse(screen, COLORS["rock"], (x + int(CELL * 0.12), y + int(CELL * 0.27), int(CELL * 0.76), int(CELL * 0.56)))
            pygame.draw.ellipse(screen, COLORS["rock_dark"], (x + int(CELL * 0.44), y + int(CELL * 0.48), int(CELL * 0.32), int(CELL * 0.24)))
        elif terrain == RIVER:
            pygame.draw.rect(screen, COLORS["river"] if checker else COLORS["river_alt"], (x, y, CELL, CELL))
        elif terrain == BRIDGE:
            pygame.draw.rect(screen, COLORS["bridge"], (x, y, CELL, CELL))
            self.draw_bridge_ties(x, y, r, c)
        elif terrain == ARTILLERY:
            pygame.draw.circle(screen, COLORS["artillery_body"], (x + int(CELL * 0.5), y + int(CELL * 0.6)), int(CELL * 0.24))
            pygame.draw.line(screen, COLORS["artillery_barrel"], (x + int(CELL * 0.5), y + int(CELL * 0.55)), (x + int(CELL * 0.9), y + int(CELL * 0.2)), 4)
        elif terrain == CAMP:
            pygame.draw.rect(screen, COLORS["camp_tent_dark"], (x + 2, y + int(CELL * 0.55), CELL - 4, int(CELL * 0.4)))
            pygame.draw.polygon(screen, COLORS["camp_tent"], [
                (x + 2, y + int(CELL * 0.58)),
                (x + CELL // 2, y + int(CELL * 0.1)),
                (x + CELL - 2, y + int(CELL * 0.58)),
            ])
        elif terrain == RUIN:
            pygame.draw.rect(screen, COLORS["ruin_wall"], (x + 3, y + int(CELL * 0.3), CELL - 6, int(CELL * 0.65)))
            pygame.draw.lines(screen, COLORS["ruin_crack"], False, [
                (x + int(CELL * 0.3), y + int(CELL * 0.3)),
                (x + int(CELL * 0.45), y + int(CELL * 0.6)),
                (x + int(CELL * 0.35), y + int(CELL * 0.95)),
            ], 1)
            pygame.draw.line(screen, COLORS["ruin_crack"], (x + 3, y + int(CELL * 0.3)), (x + int(CELL * 0.6), y + int(CELL * 0.08)), 1)

        self.draw_cost_label(terrain, x, y)

    def draw_bridge_ties(self, x, y, r, c):
        axis = self.get_river_axis(r, c)
        col = (0, 0, 0)
        surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        if axis == "vertical":
            for i in range(1, 4):
                pygame.draw.line(surf, (*col, 76), (i * CELL / 4, 0), (i * CELL / 4, CELL), 1)
        elif axis == "horizontal":
            for i in range(1, 4):
                pygame.draw.line(surf, (*col, 76), (0, i * CELL / 4), (CELL, i * CELL / 4), 1)
        else:
            sign = 1 if axis == "diag-main" else -1
            for i in (-1, 0, 1):
                off = i * CELL * 0.6
                if sign == 1:
                    p0 = (-CELL + off, -CELL)
                    p1 = (2 * CELL + off, 2 * CELL)
                else:
                    p0 = (2 * CELL + off, -CELL)
                    p1 = (-CELL + off, 2 * CELL)
                pygame.draw.line(surf, (*col, 76), p0, p1, 1)
        self.screen.blit(surf, (x, y))

    def is_riverish(self, t):
        return t in (RIVER, BRIDGE)

    def get_river_axis(self, r, c):
        grid = self.grid
        north = r > 0 and self.is_riverish(grid[r - 1][c])
        south = r < ROWS - 1 and self.is_riverish(grid[r + 1][c])
        west = c > 0 and self.is_riverish(grid[r][c - 1])
        east = c < COLS - 1 and self.is_riverish(grid[r][c + 1])
        nw = r > 0 and c > 0 and self.is_riverish(grid[r - 1][c - 1])
        se = r < ROWS - 1 and c < COLS - 1 and self.is_riverish(grid[r + 1][c + 1])
        ne = r > 0 and c < COLS - 1 and self.is_riverish(grid[r - 1][c + 1])
        sw = r < ROWS - 1 and c > 0 and self.is_riverish(grid[r + 1][c - 1])
        if north or south:
            return "vertical"
        if west or east:
            return "horizontal"
        if nw or se:
            return "diag-main"
        if ne or sw:
            return "diag-anti"
        return "horizontal"

    def draw_cost_label(self, terrain, x, y):
        blocked = terrain == ARTILLERY
        cost = DRONE_TERRAIN_COST.get(terrain)
        if not blocked and cost == 1:
            return
        label = "X" if blocked else str(cost)
        color = COLORS["cost_label_blocked"] if blocked else COLORS["cost_label"]
        txt = self.font_cost.render(label, True, color)
        rect = txt.get_rect()
        rect.bottomright = (x + CELL - 3, y + CELL - 3)
        outline = self.font_cost.render(label, True, (0, 0, 0))
        for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            self.screen.blit(outline, (rect.x + ox, rect.y + oy))
        self.screen.blit(txt, rect)

    def draw_tank(self):
        screen = self.screen
        r, c = self.tank
        cx = BOARD_X + c * CELL + CELL // 2
        cy = BOARD_Y + r * CELL + CELL // 2
        dr, dc = self.tank_dir

        body_w, body_h = int(CELL * 0.64), int(CELL * 0.48)
        pygame.draw.rect(screen, COLORS["tank"], (cx - body_w // 2, cy - body_h // 2, body_w, body_h))
        track_h = int(CELL * 0.1)
        track_w = int(CELL * 0.68)
        pygame.draw.rect(screen, COLORS["tank_dark"], (cx - track_w // 2, cy - int(CELL * 0.3), track_w, track_h))
        pygame.draw.rect(screen, COLORS["tank_dark"], (cx - track_w // 2, cy + int(CELL * 0.2), track_w, track_h))
        pygame.draw.circle(screen, COLORS["tank_dark"], (cx, cy), int(CELL * 0.18))
        barrel_len = CELL * 0.36
        pygame.draw.line(screen, COLORS["tank_dark"], (cx, cy), (cx + dc * barrel_len, cy + dr * barrel_len), 4)

    def draw_drone(self):
        screen = self.screen
        r, c = self.drone
        cx = BOARD_X + c * CELL + CELL // 2
        cy = BOARD_Y + r * CELL + CELL // 2
        radius = CELL * 0.32

        for ox, oy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            arm_x, arm_y = cx + ox * radius, cy + oy * radius
            pygame.draw.line(screen, COLORS["drone_dark"], (cx, cy), (arm_x, arm_y), 2)
            pygame.draw.circle(screen, COLORS["drone_dark"], (arm_x, arm_y), int(CELL * 0.09))
        pygame.draw.circle(screen, COLORS["drone"], (cx, cy), int(CELL * 0.14))

    def draw_legend(self):
        screen = self.screen
        note = self.font_legend.render(
            'Angka di pojok tiap tile = biaya tempuh DRONE (biaya 1/tanah tidak ditulis); "X" = terhalang total.',
            True, COLORS["text_cream"],
        )
        screen.blit(note, (BOARD_X, LEGEND_Y))

        col_w = (BOARD_W // 2)
        line_h = 16
        start_y = LEGEND_Y + 20
        for i, (color_key, label) in enumerate(LEGEND_ITEMS):
            col = i // 9
            row = i % 9
            lx = BOARD_X + col * col_w
            ly = start_y + row * line_h
            pygame.draw.rect(screen, COLORS[color_key], (lx, ly + 2, 12, 12), border_radius=3)
            txt = self.font_legend.render(label, True, COLORS["text_dim"])
            screen.blit(txt, (lx + 18, ly))

    def draw_status(self):
        screen = self.screen
        rect = pygame.Rect(BOARD_X, STATUS_Y, BOARD_W, 34)
        color = COLORS["danger"] if self.status_caught else COLORS["panel_border"]
        pygame.draw.rect(screen, (22, 21, 16), rect, border_radius=7)
        pygame.draw.rect(screen, color, rect, width=1, border_radius=7)
        text_color = COLORS["danger"] if self.status_caught else COLORS["text_dim"]
        lines = wrap_text(self.status_text, self.font_ui, BOARD_W - 20)
        for i, line in enumerate(lines[:2]):
            txt = self.font_ui.render(line, True, text_color)
            screen.blit(txt, (BOARD_X + 10, STATUS_Y + 6 + i * 15))

    def draw_panel(self):
        screen = self.screen
        panel_rect = pygame.Rect(PANEL_X, PANEL_Y, PANEL_W, SCREEN_H - PANEL_Y - 20)
        pygame.draw.rect(screen, COLORS["panel"], panel_rect, border_radius=12)
        pygame.draw.rect(screen, COLORS["panel_border"], panel_rect, width=1, border_radius=12)

        x = self.panel_x
        header = self.font_ui_bold.render("Kontrol Algoritma Drone", True, COLORS["text_dim"])
        screen.blit(header, (x, PANEL_Y + 16))

        # highlight tombol algoritma & heuristik & vision aktif
        self.btn_algo_ucs.primary = self.algo == "ucs"
        self.btn_algo_astar.primary = self.algo == "astar"
        self.btn_algo_ucs.draw(screen, self.font_ui)
        self.btn_algo_astar.draw(screen, self.font_ui)

        for btn, key in (
            (self.btn_heur_manhattan, "manhattan"),
            (self.btn_heur_euclidean, "euclidean"),
            (self.btn_heur_chebyshev, "chebyshev"),
        ):
            btn.enabled = self.algo == "astar"
            btn.primary = self.algo == "astar" and self.heuristic == key
            btn.draw(screen, self.font_ui)

        for btn, v in zip(self.btn_visions, self.vision_options):
            btn.primary = v == self.vision_radius
            btn.draw(screen, self.font_ui)

        self.btn_auto_chase.label = ("[x] " if self.auto_chase else "[ ] ") + "Drone bergerak otomatis tiap 450ms"
        self.btn_auto_chase.primary = False
        self.btn_auto_chase.draw(screen, self.font_ui)

        state_y = self.btn_auto_chase.rect.bottom + 10
        lbl = self.font_ui.render("Status drone saat ini", True, COLORS["text_cream"])
        screen.blit(lbl, (x, state_y))
        state_box = pygame.Rect(x, state_y + 18, self.panel_w, 36)
        pygame.draw.rect(screen, (22, 21, 16), state_box, border_radius=6)
        pygame.draw.rect(screen, COLORS["panel_border"], state_box, width=1, border_radius=6)
        for i, line in enumerate(wrap_text(self.drone_state_text, self.font_small, self.panel_w - 16)[:2]):
            t = self.font_small.render(line, True, COLORS["text_cream"])
            screen.blit(t, (x + 8, state_y + 22 + i * 14))

        self.btn_random_map.draw(screen, self.font_ui)
        self.btn_reset_pos.draw(screen, self.font_ui)

        stats_hdr = self.font_ui_bold.render("Statistik Pencarian Terakhir", True, COLORS["text_dim"])
        screen.blit(stats_hdr, (x, self.stats_panel_y))
        stats_box = pygame.Rect(x, self.stats_panel_y + 18, self.panel_w, 78)
        pygame.draw.rect(screen, (22, 21, 16), stats_box, border_radius=6)
        pygame.draw.rect(screen, COLORS["panel_border"], stats_box, width=1, border_radius=6)
        for i, line in enumerate(self.stats_lines):
            t = self.font_small.render(line, True, COLORS["text_cream"])
            screen.blit(t, (x + 8, self.stats_panel_y + 24 + i * 14))

        exp_hdr = self.font_ui_bold.render("Eksperimen Perbandingan Heuristik", True, COLORS["text_dim"])
        screen.blit(exp_hdr, (x, self.experiment_y - 30))
        self.btn_experiment.draw(screen, self.font_ui)

        table_y = self.btn_experiment.rect.bottom + 10
        if self.experiment_lines:
            headers = ["Kombinasi", "Node", "Biaya", "ms"]
            col_x = [x, x + 150, x + 210, x + 260]
            for i, h in enumerate(headers):
                t = self.font_small.render(h, True, COLORS["text_dim"])
                screen.blit(t, (col_x[i], table_y))
            table_y += 16
            for label, expanded, cost_txt, ms_txt, best in self.experiment_lines:
                color = COLORS["accent_hover"] if best else COLORS["text_cream"]
                vals = [label, str(expanded), cost_txt, ms_txt]
                for i, v in enumerate(vals):
                    t = self.font_small.render(v, True, color)
                    screen.blit(t, (col_x[i], table_y))
                table_y += 15
            table_y += 6
            for line in wrap_text(self.experiment_note, self.font_legend, self.panel_w):
                t = self.font_legend.render(line, True, COLORS["text_dim"])
                screen.blit(t, (x, table_y))
                table_y += 14


# =============================================================================
# BAGIAN 7 -- MAIN LOOP
# =============================================================================
def main():
    pygame.init()
    pygame.display.set_caption("Tank vs Drone -- UCS/A* + Battle Minimax -- Simulasi Ukraina")
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()

    game = Game(screen)

    running = True
    while running:
        now_ms = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    game.handle_keydown(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                game.handle_click(event.pos)

        game.update(now_ms)
        game.draw()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()