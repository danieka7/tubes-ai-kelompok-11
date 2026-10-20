import math
import random
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache

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


# -----------------------------------------------------------------------------
# CSP pembangkit peta: backtracking + forward checking + MRV, degree, LCV
#   Variabel   : tiap objek (camp, ruin, artileri, pohon, batu)
#   Domain     : anchor (baris, kolom) pojok kiri-atas footprint objek
#   Constraint : unary (sel DIRT, di luar bridge_zone, bukan start/goal),
#                blok 2x2 (footprint menutup 4 sel sekaligus -> keempatnya sama),
#                tidak tumpang tindih (forward checking),
#                kardinalitas (jumlah variabel per jenis tetap),
#                global (tank harus bisa mencapai goal, dicek tiap penempatan)
# -----------------------------------------------------------------------------
TREE_RATIO = 0.08
ROCK_RATIO = 0.06
ARTILLERY_COUNT = 5
CAMP_COUNT = 2
RUIN_COUNT = 2
BRIDGE_COUNT = 2
SINGLE_SIZE = 1
BLOCK_SIZE = 2
MAX_MAP_ATTEMPTS = 25        # batas pembangkitan ulang sungai
MAX_SEARCH_NODES = 20000     # batas penempatan per percobaan agar tidak berputar tanpa akhir
LCV_JITTER = 0.6             # derau skor LCV supaya blok tidak selalu terdorong ke pojok peta
MAP_DEBUG_LINE_H = 16

TERRAIN_LABELS = {
    TREE: "Pohon",
    ROCK: "Batu",
    ARTILLERY: "Artileri",
    CAMP: "Camp",
    RUIN: "Ruin",
}


class MapGenerationError(RuntimeError):
    """Peta yang valid tidak berhasil dibangkitkan dalam batas percobaan."""


@dataclass(frozen=True)
class MapVariable:
    name: str
    terrain: int
    size: int


@dataclass
class MapGenStats:
    """Angka proses CSP untuk panel debug (diakumulasi dari semua percobaan)."""
    attempts: int = 0
    base_unreachable: int = 0
    variable_count: int = 0
    single_domain_size: int = 0
    block_domain_size: int = 0
    nodes_visited: int = 0
    max_depth: int = 0
    backtracks: int = 0
    values_pruned: int = 0
    wipeouts: int = 0
    reachability_checks: int = 0
    reachability_prunes: int = 0
    min_domain_at_select: int = ROWS * COLS
    elapsed_ms: float = 0.0
    placed: dict = field(default_factory=dict)
    targets: dict = field(default_factory=dict)


@lru_cache(maxsize=None)
def footprint(size, row, col):
    """Sel yang ditutup objek berukuran size x size dengan anchor (row, col)."""
    return tuple((row + dr, col + dc) for dr in range(size) for dc in range(size))


def overlapping_anchors(other_size, size, anchor):
    """Semua anchor objek berukuran other_size yang footprint-nya beririsan
    dengan objek (size, anchor). Waktu/memori: O((size + other_size - 1)^2)."""
    row, col = anchor
    return [
        (r, c)
        for r in range(row - other_size + 1, row + size)
        for c in range(col - other_size + 1, col + size)
    ]


def build_map_variables():
    """Satu variabel per objek. Blok dipasang lebih dulu oleh MRV karena domainnya lebih kecil."""
    total_cells = ROWS * COLS
    specs = (
        ("camp", CAMP, BLOCK_SIZE, CAMP_COUNT),
        ("ruin", RUIN, BLOCK_SIZE, RUIN_COUNT),
        ("artileri", ARTILLERY, SINGLE_SIZE, ARTILLERY_COUNT),
        ("pohon", TREE, SINGLE_SIZE, int(total_cells * TREE_RATIO)),
        ("batu", ROCK, SINGLE_SIZE, int(total_cells * ROCK_RATIO)),
    )
    return [
        MapVariable(f"{label}_{index}", terrain, size)
        for label, terrain, size, count in specs
        for index in range(count)
    ]


def build_initial_domain(grid, size, forbidden):
    """Constraint unary: seluruh footprint harus DIRT dan di luar sel terlarang.
    Waktu: O(ROWS * COLS * size^2). Memori: O(ROWS * COLS)."""
    domain = set()
    for row in range(ROWS - size + 1):
        for col in range(COLS - size + 1):
            cells = footprint(size, row, col)
            if all(grid[r][c] == DIRT and (r, c) not in forbidden for r, c in cells):
                domain.add((row, col))
    return domain


def build_base_grid(start_pos, goal_pos):
    """Sungai + jembatan, start/goal dipaksa DIRT (constraint unary)."""
    grid = [[DIRT] * COLS for _ in range(ROWS)]
    river_cells = generate_river_path()
    for r, c in river_cells:
        grid[r][c] = RIVER
    bridge_cells = pick_bridge_cells(river_cells, BRIDGE_COUNT)
    for r, c in bridge_cells:
        grid[r][c] = BRIDGE
    for r, c in (start_pos, goal_pos):
        grid[r][c] = DIRT
    return grid, build_bridge_exclusion_zone(bridge_cells)


def eliminated_count(size, anchor, groups):
    """Skor LCV: jumlah nilai di domain variabel lain yang hilang bila (size, anchor) dipilih.
    Variabel sejenis berbagi domain identik, jadi dihitung sekali per ukuran lalu dikalikan jumlahnya."""
    total = 0
    for other_size, (domain, count) in groups.items():
        hits = sum(1 for a in overlapping_anchors(other_size, size, anchor) if a in domain)
        total += hits * count
    return total


class MapCspSolver:
    """Backtracking dengan forward checking, MRV, degree, LCV, dan pemangkasan keterjangkauan.
    Waktu terburuk eksponensial, dibatasi MAX_SEARCH_NODES; memori O(variabel * domain)."""

    def __init__(self, grid, variables, domains_by_size, start, goal, stats):
        self.grid = grid
        self.variables = variables
        self.start = start
        self.goal = goal
        self.stats = stats
        self.domains = {v.name: set(domains_by_size[v.size]) for v in variables}
        self.assignment = {}

    def solve(self):
        """True bila semua variabel terpasang tanpa melanggar constraint."""
        return self._backtrack()

    def _backtrack(self):
        if len(self.assignment) == len(self.variables):
            return True
        var = self._select_variable()
        for anchor in self._order_values(var):
            removed = self._assign(var, anchor)
            if removed is None:
                continue
            if self._backtrack():
                return True
            self._unassign(var, anchor, removed)
        self.stats.backtracks += 1
        return False

    def _unassigned(self):
        return [v for v in self.variables if v.name not in self.assignment]

    def _degree(self, var, unassigned):
        """Degree: jumlah nilai variabel lain yang bentrok dengan SATU nilai var."""
        return sum(
            (var.size + other.size - 1) ** 2
            for other in unassigned
            if other.name != var.name
        )

    def _select_variable(self):
        """MRV (domain terkecil); seri dipecah dengan degree terbesar."""
        unassigned = self._unassigned()
        chosen = min(
            unassigned,
            key=lambda v: (len(self.domains[v.name]), -self._degree(v, unassigned)),
        )
        smallest = len(self.domains[chosen.name])
        self.stats.min_domain_at_select = min(self.stats.min_domain_at_select, smallest)
        return chosen

    def _group_domains(self, var):
        """ukuran -> (domain wakil, jumlah variabel lain yang belum terpasang)."""
        groups = {}
        for other in self._unassigned():
            if other.name == var.name:
                continue
            domain, count = groups.get(other.size, (self.domains[other.name], 0))
            groups[other.size] = (domain, count + 1)
        return groups

    def _order_values(self, var):
        """LCV: nilai yang paling sedikit membuang domain lain dicoba lebih dulu."""
        groups = self._group_domains(var)
        scored = [
            (eliminated_count(var.size, a, groups) * random.uniform(1 - LCV_JITTER, 1 + LCV_JITTER), a)
            for a in self.domains[var.name]
        ]
        scored.sort()
        return [anchor for _, anchor in scored]

    def _assign(self, var, anchor):
        """Pasang nilai. Mengembalikan catatan nilai terpangkas, atau None bila ditolak
        (batas node, domain kosong, atau tank tak bisa mencapai goal) setelah dibatalkan."""
        if self.stats.nodes_visited >= MAX_SEARCH_NODES:
            return None
        self.stats.nodes_visited += 1
        self._paint(var, anchor, var.terrain)
        self.assignment[var.name] = anchor
        self.stats.max_depth = max(self.stats.max_depth, len(self.assignment))
        removed, consistent = self._forward_check(var, anchor)
        if consistent and self._goal_still_reachable():
            return removed
        self._unassign(var, anchor, removed)
        return None

    def _unassign(self, var, anchor, removed):
        for name, anchors in removed.items():
            self.domains[name].update(anchors)
        del self.assignment[var.name]
        self._paint(var, anchor, DIRT)

    def _paint(self, var, anchor, terrain):
        for r, c in footprint(var.size, *anchor):
            self.grid[r][c] = terrain

    def _forward_check(self, var, anchor):
        """Constraint tidak-tumpang-tindih: buang nilai variabel lain yang beririsan.
        Mengembalikan (catatan untuk undo, apakah tidak ada domain yang kosong)."""
        removed = {}
        for other in self.variables:
            if other.name in self.assignment:
                continue
            domain = self.domains[other.name]
            hits = [a for a in overlapping_anchors(other.size, var.size, anchor) if a in domain]
            domain.difference_update(hits)
            removed[other.name] = hits
            self.stats.values_pruned += len(hits)
            if not domain:
                self.stats.wipeouts += 1
                return removed, False
        return removed, True

    def _goal_still_reachable(self):
        """Constraint global. Menambah penghalang tidak pernah memulihkan jalur,
        jadi cabang yang sudah terputus aman dipangkas. Waktu: O(ROWS * COLS)."""
        self.stats.reachability_checks += 1
        if is_reachable(self.grid, self.start, self.goal):
            return True
        self.stats.reachability_prunes += 1
        return False


def try_generate_grid(start_pos, goal_pos, stats):
    """Satu percobaan: sungai baru lalu CSP. Mengembalikan grid, atau None bila gagal."""
    grid, bridge_zone = build_base_grid(start_pos, goal_pos)
    if not is_reachable(grid, start_pos, goal_pos):
        stats.base_unreachable += 1
        return None

    forbidden = bridge_zone | {start_pos, goal_pos}
    domains_by_size = {
        size: build_initial_domain(grid, size, forbidden)
        for size in (SINGLE_SIZE, BLOCK_SIZE)
    }
    variables = build_map_variables()
    stats.variable_count = len(variables)
    stats.single_domain_size = len(domains_by_size[SINGLE_SIZE])
    stats.block_domain_size = len(domains_by_size[BLOCK_SIZE])
    stats.targets = {}
    for var in variables:
        stats.targets[var.terrain] = stats.targets.get(var.terrain, 0) + 1

    solver = MapCspSolver(grid, variables, domains_by_size, start_pos, goal_pos, stats)
    if not solver.solve():
        return None
    stats.placed = {}
    for var in variables:
        stats.placed[var.terrain] = stats.placed.get(var.terrain, 0) + 1
    return grid


def validate_position(name, pos):
    if not (0 <= pos[0] < ROWS and 0 <= pos[1] < COLS):
        raise ValueError(f"{name} {pos} berada di luar peta {ROWS}x{COLS}")


def generate_map(start_pos, goal_pos):
    """Bangkitkan peta lewat CSP. Mengembalikan (grid, MapGenStats).
    Raises MapGenerationError bila MAX_MAP_ATTEMPTS percobaan semuanya gagal."""
    validate_position("start_pos", start_pos)
    validate_position("goal_pos", goal_pos)
    stats = MapGenStats()
    started = time.perf_counter()
    grid = None
    while grid is None and stats.attempts < MAX_MAP_ATTEMPTS:
        stats.attempts += 1
        grid = try_generate_grid(start_pos, goal_pos, stats)
    stats.elapsed_ms = (time.perf_counter() - started) * 1000.0
    if grid is None:
        raise MapGenerationError(
            f"Peta valid tidak ditemukan setelah {MAX_MAP_ATTEMPTS} percobaan."
        )
    return grid, stats


def format_map_debug_lines(stats):
    """Baris teks panel debug CSP pembangkit peta."""
    placed = " | ".join(
        f"{TERRAIN_LABELS[t]} {stats.placed.get(t, 0)}/{n}" for t, n in stats.targets.items()
    )
    return [
        f"[Debug CSP Peta]  waktu: {stats.elapsed_ms:.1f} ms",
        f"Percobaan peta    : {stats.attempts}/{MAX_MAP_ATTEMPTS}  (sungai tak terjangkau: {stats.base_unreachable})",
        f"Variabel          : {stats.variable_count}  | domain sel: {stats.single_domain_size}  | domain blok 2x2: {stats.block_domain_size}",
        f"Node dikunjungi   : {stats.nodes_visited}  (batas {MAX_SEARCH_NODES})  | kedalaman maks: {stats.max_depth}",
        f"Backtrack         : {stats.backtracks}",
        f"Forward checking  : nilai dipangkas {stats.values_pruned}  | domain kosong: {stats.wipeouts}",
        f"Reachability      : dicek {stats.reachability_checks}  | cabang dipangkas: {stats.reachability_prunes}",
        f"MRV               : domain terkecil saat pemilihan: {stats.min_domain_at_select}",
        f"Terpasang         : {placed}",
    ]


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
    "bg_deep": (26, 36, 22),
    "panel": (36, 50, 30),
    "panel_border": (84, 104, 66),
    "text_cream": (220, 234, 196),
    "text_dim": (160, 182, 128),
    "accent": (163, 199, 98),
    "accent_hover": (200, 230, 140),

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

# Mode battle memakai layar penuh (board+panel eksplorasi digantikan sepenuhnya
# oleh HUD battle: HP/log/tombol aksi di kiri, panel kontrol algoritma +
# pohon pencarian / tabel eksperimen di kanan).
BATTLE_AREA_MARGIN = 20

# Ukuran desain awal dipakai sebagai batas minimum jendela, supaya UI
# tidak pernah lebih sempit dari yang dirancang (kalau lebih sempit,
# tombol dan teks akan bertumpuk).
MIN_SCREEN_W = SCREEN_W
MIN_SCREEN_H = SCREEN_H
BASE_CELL = 30          # ukuran petak minimum (= nilai CELL awal)
PANEL_MIN_W = 320       # lebar panel kanan minimum
LEGEND_BLOCK_H = 190    # tinggi ruang legenda di bawah papan
STATUS_BLOCK_H = 60     # tinggi ruang kotak status di bawah legenda


def apply_layout(win_w, win_h):
    """Hitung ulang semua konstanta tata letak mode eksplorasi
    berdasarkan ukuran jendela (win_w x win_h).

    Aturannya:
      - Papan dibesarkan dengan menaikkan CELL, dibatasi oleh tinggi DAN
        lebar jendela (dipilih yang lebih kecil) supaya tidak ada yang
        keluar layar.
      - Sisa lebar diberikan ke panel kanan, jadi panel ikut melebar.
    """
    # Variabel-variabel ini global karena dibaca oleh banyak fungsi gambar.
    global CELL, BOARD_W, BOARD_H, PANEL_X, PANEL_W
    global LEGEND_Y, STATUS_Y, SCREEN_W, SCREEN_H

    # Jendela tidak boleh lebih kecil dari ukuran desain awal.
    SCREEN_W = max(MIN_SCREEN_W, win_w)
    SCREEN_H = max(MIN_SCREEN_H, win_h)

    # Petak terbesar yang masih muat secara vertikal
    # (papan + legenda + status harus muat di tinggi jendela).
    cell_by_h = (SCREEN_H - BOARD_Y - 12 - LEGEND_BLOCK_H - STATUS_BLOCK_H) // ROWS
    # Petak terbesar yang masih muat secara horizontal
    # (papan + jarak 24 + panel minimum + margin kanan 20).
    cell_by_w = (SCREEN_W - BOARD_X - 24 - PANEL_MIN_W - 20) // COLS

    # Ambil yang lebih kecil, tetapi tidak boleh di bawah ukuran dasar.
    CELL = max(BASE_CELL, min(cell_by_h, cell_by_w))

    BOARD_W, BOARD_H = COLS * CELL, ROWS * CELL

    # Panel menempati seluruh sisa lebar di kanan papan.
    PANEL_X = BOARD_X + BOARD_W + 24
    PANEL_W = SCREEN_W - PANEL_X - 20

    LEGEND_Y = BOARD_Y + BOARD_H + 12
    STATUS_Y = LEGEND_Y + LEGEND_BLOCK_H


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
            bg = (50, 68, 42)
            fg = COLORS["text_dim"]
        elif self.primary:
            bg = COLORS["accent"]
            fg = (24, 34, 20)
        else:
            bg = (50, 68, 42)
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


# =============================================================================
# BAGIAN 5B.1 -- BERBAGAI FUNGSI EVALUASI (untuk eksperimen "tingkah laku NPC")
# =============================================================================
# Semua Eval(s) di bawah dipakai HANYA pada cutoff non-terminal (depth limit),
# bukan sebagai utility. Bobot berbeda -> "kepribadian" NPC berbeda meski
# algoritma pencariannya (minimax/alpha-beta) sama persis.

def battle_eval_seimbang(state):
    """Seimbang: HP jadi prioritas utama, kit reparasi & status bertahan sekunder."""
    score = 1.0 * (state.npcHP - state.playerHP)
    score += 6.0 * (state.npcRepairKits - state.playerRepairKits)
    score += 8.0 * (1 if state.npcDefending else 0)
    score -= 8.0 * (1 if state.playerDefending else 0)
    return score


def battle_eval_agresif(state):
    """Agresif: selisih HP dibobot sangat tinggi, kit reparasi/bertahan nyaris
    diabaikan -> NPC cenderung terus menyerang, jarang bertahan/reparasi diri."""
    score = 4.0 * (state.npcHP - state.playerHP)
    score += 1.0 * (state.npcRepairKits - state.playerRepairKits)
    score += 2.0 * (1 if state.npcDefending else 0)
    score -= 2.0 * (1 if state.playerDefending else 0)
    return score


def battle_eval_defensif(state):
    """Defensif: status bertahan & HP sendiri dibobot sangat tinggi ->
    NPC cenderung bertahan/reparasi diri lebih sering, menghindari HP rendah."""
    score = 1.0 * (state.npcHP - state.playerHP)
    score += 5.0 * (state.npcRepairKits - state.playerRepairKits)
    score += 20.0 * (1 if state.npcDefending else 0)
    score -= 20.0 * (1 if state.playerDefending else 0)
    if state.npcHP < 40:
        score += 25.0  # panik kalau HP sendiri kritis -> makin ingin bertahan/reparasi
    return score


def battle_eval_hemat_reparasi(state):
    """Hemat reparasi: selisih jumlah kit reparasi dibobot paling berat -> NPC
    berusaha menimbun/tidak boros kit reparasi dibanding menyerang."""
    score = 0.5 * (state.npcHP - state.playerHP)
    score += 15.0 * (state.npcRepairKits - state.playerRepairKits)
    score += 4.0 * (1 if state.npcDefending else 0)
    score -= 4.0 * (1 if state.playerDefending else 0)
    return score


# key -> (fungsi, label tampilan)
EVAL_FUNCTIONS = {
    "seimbang": (battle_eval_seimbang, "Seimbang"),
    "agresif": (battle_eval_agresif, "Agresif"),
    "defensif": (battle_eval_defensif, "Defensif"),
    "hemat": (battle_eval_hemat_reparasi, "Hemat Reparasi"),
}

# Dipertahankan agar nama lama tetap bisa dipakai di tempat lain / kompatibel.
battle_eval_default = battle_eval_seimbang

# key -> (urutan aksi, label tampilan) -- dipakai utk eksperimen "urutan aksi"
# terhadap efisiensi alpha-beta pruning (move ordering).
ACTION_ORDERS = {
    "default": ([ACT_ATTACK, ACT_DEFEND, ACT_REPAIR], "Serang -> Bertahan -> Reparasi"),
    "bertahan_dulu": ([ACT_DEFEND, ACT_ATTACK, ACT_REPAIR], "Bertahan -> Serang -> Reparasi"),
    "reparasi_dulu": ([ACT_REPAIR, ACT_DEFEND, ACT_ATTACK], "Reparasi -> Bertahan -> Serang"),
}


# =============================================================================
# BAGIAN 5B.2 -- LOADOUT DRONE, GELOMBANG MUSUH & PARAMETER KESEIMBANGAN
# =============================================================================
# Semua angka keseimbangan (HP tank, jumlah drone, statistik tiap loadout,
# aturan bantuan musuh) dikumpulkan di BalanceParams supaya bisa dicari
# otomatis oleh local search (BAGIAN 5B.4), bukan ditebak manual.

MAX_ACTIVE_DRONES = 6           # batas drone hidup bersamaan (muat di daftar HUD)
MAX_REINFORCEMENTS = 6          # batas total drone bantuan per pertempuran
PERCENT_BASE = 100

LOADOUT_BOM = "bom"
LOADOUT_MISIL = "misil"
LOADOUT_KAMIKAZE = "kamikaze"

# Sifat tetap tiap loadout (bukan hasil tuning): yang di-tuning hanya HP & damage.
BOM_ACT_INTERVAL = 2            # lambat: bertindak tiap 2 ronde
BOM_REPAIR_KITS = 1
BOM_PIERCE = 0.0
MISIL_ACT_INTERVAL = 1
MISIL_REPAIR_KITS = 2
MISIL_PIERCE = 0.5              # misil menembus separuh efek bertahan tank
KAMIKAZE_ACT_INTERVAL = 1
KAMIKAZE_REPAIR_KITS = 0
KAMIKAZE_PIERCE = 0.0

# Bobot kemunculan loadout saat drone awal / bantuan dipilih acak.
LOADOUT_SPAWN_WEIGHTS = {LOADOUT_BOM: 3, LOADOUT_MISIL: 4, LOADOUT_KAMIKAZE: 3}


@dataclass(frozen=True)
class Loadout:
    """Statistik satu jenis drone. frozen=True agar aman dipakai sebagai kunci cache."""
    key: str
    name: str
    max_hp: int
    damage: int
    act_interval: int      # bertindak sekali tiap N ronde (beda "turn")
    repair_kits: int
    pierce: float          # 0..1: porsi efek bertahan lawan yang diabaikan
    suicide: bool          # True: hancur setelah menyerang (kamikaze)


@dataclass(frozen=True)
class BalanceParams:
    """Vektor parameter yang dicari oleh local search."""
    tank_hp: int = 160
    initial_drones: int = 3
    bom_hp: int = 70
    bom_damage: int = 26
    misil_hp: int = 50
    misil_damage: int = 17
    kamikaze_hp: int = 24
    kamikaze_damage: int = 38
    reinforce_after: int = 6          # bantuan baru boleh muncul setelah ronde ke-N
    reinforce_chance_pct: int = 25    # peluang (%) bantuan muncul tiap ronde setelahnya

    def __post_init__(self):
        if not 1 <= self.initial_drones <= MAX_ACTIVE_DRONES:
            raise ValueError(f"initial_drones harus 1..{MAX_ACTIVE_DRONES}, dapat {self.initial_drones}")
        if not 0 <= self.reinforce_chance_pct <= PERCENT_BASE:
            raise ValueError(f"reinforce_chance_pct harus 0..{PERCENT_BASE}, dapat {self.reinforce_chance_pct}")
        positive_fields = (self.tank_hp, self.bom_hp, self.bom_damage, self.misil_hp,
                           self.misil_damage, self.kamikaze_hp, self.kamikaze_damage)
        if min(positive_fields) <= 0:
            raise ValueError("HP dan damage harus > 0")


# Hasil local search terakhir (lihat `python <file> --tune`). Dipakai game.
TUNED_PARAMS = BalanceParams(tank_hp=225, initial_drones=3, bom_hp=40, bom_damage=18, misil_hp=35, misil_damage=14, kamikaze_hp=24, kamikaze_damage=47, reinforce_after=9, reinforce_chance_pct=15)

def build_loadouts(params):
    """Bangun tabel loadout dari parameter keseimbangan.

    Args:
        params: BalanceParams.
    Returns:
        dict key loadout -> Loadout.
    Kompleksitas: O(1) waktu & memori.
    """
    return {
        LOADOUT_BOM: Loadout(LOADOUT_BOM, "Bom", params.bom_hp, params.bom_damage,
                             BOM_ACT_INTERVAL, BOM_REPAIR_KITS, BOM_PIERCE, False),
        LOADOUT_MISIL: Loadout(LOADOUT_MISIL, "Misil", params.misil_hp, params.misil_damage,
                               MISIL_ACT_INTERVAL, MISIL_REPAIR_KITS, MISIL_PIERCE, False),
        LOADOUT_KAMIKAZE: Loadout(LOADOUT_KAMIKAZE, "Kamikaze", params.kamikaze_hp, params.kamikaze_damage,
                                  KAMIKAZE_ACT_INTERVAL, KAMIKAZE_REPAIR_KITS, KAMIKAZE_PIERCE, True),
    }


DEFAULT_NPC_LOADOUT = build_loadouts(TUNED_PARAMS)[LOADOUT_MISIL]


class BattleState:
    """State duel 1 lawan 1 (tank vs SATU drone) yang dipakai pohon minimax.

    s = (playerHP, npcHP, playerRepairKits, npcRepairKits,
         playerDefending, npcDefending, turnOwner, turnIndex)
    ditambah playerMaxHP (batas reparasi tank) dan npcSpec (Loadout drone).
    Pertempuran banyak drone dipecah jadi beberapa duel semacam ini oleh Encounter.
    """
    __slots__ = (
        "playerHP", "npcHP", "playerRepairKits", "npcRepairKits",
        "playerDefending", "npcDefending", "turnOwner", "turnIndex",
        "playerMaxHP", "npcSpec",
    )

    def __init__(self, playerHP, npcHP, playerRepairKits, npcRepairKits,
                 playerDefending, npcDefending, turnOwner, turnIndex,
                 playerMaxHP=None, npcSpec=None):
        self.playerHP = playerHP
        self.npcHP = npcHP
        self.playerRepairKits = playerRepairKits
        self.npcRepairKits = npcRepairKits
        self.playerDefending = playerDefending
        self.npcDefending = npcDefending
        self.turnOwner = turnOwner
        self.turnIndex = turnIndex
        self.playerMaxHP = TUNED_PARAMS.tank_hp if playerMaxHP is None else playerMaxHP
        self.npcSpec = DEFAULT_NPC_LOADOUT if npcSpec is None else npcSpec

    @staticmethod
    def initial():
        return BattleState(
            playerHP=TUNED_PARAMS.tank_hp, npcHP=DEFAULT_NPC_LOADOUT.max_hp,
            playerRepairKits=BATTLE_START_REPAIR_KITS, npcRepairKits=DEFAULT_NPC_LOADOUT.repair_kits,
            playerDefending=False, npcDefending=False,
            turnOwner=PLAYER, turnIndex=0,
        )

    def is_terminal(self):
        """Terminal(s) = (playerHP <= 0) or (npcHP <= 0)."""
        return self.playerHP <= 0 or self.npcHP <= 0

    def legal_actions(self, order=None):
        """Branching <=3: ATTACK, DEFEND selalu ada; REPAIR hilang kalau stok habis.

        `order` (opsional): urutan pengecekan aksi (list of ACT_*), dipakai
        untuk eksperimen "urutan aksi" (move ordering) terhadap efisiensi
        alpha-beta pruning. Default: ATTACK, DEFEND, REPAIR."""
        order = order or ACTION_ORDERS["default"][0]
        repair_kits = self.playerRepairKits if self.turnOwner == PLAYER else self.npcRepairKits
        actions = []
        for a in order:
            if a == ACT_REPAIR and repair_kits <= 0:
                continue
            actions.append(a)
        return actions

    def _attack_damage(self, is_player_turn, opp_defending):
        """Damage serangan giliran ini setelah memperhitungkan bertahan lawan.

        Serangan drone memakai damage loadout-nya dan bisa menembus sebagian
        efek bertahan tank (pierce); serangan tank memakai ATTACK_DAMAGE."""
        if is_player_turn:
            base_damage, pierce = ATTACK_DAMAGE, 0.0
        else:
            base_damage, pierce = self.npcSpec.damage, self.npcSpec.pierce
        if not opp_defending:
            return base_damage
        return base_damage * (1.0 - DEFEND_REDUCTION * (1.0 - pierce))

    def apply_action(self, action):
        """Kembalikan BattleState BARU (tidak memutasi state ini) supaya aman
        dipakai berulang kali di pohon pencarian minimax."""
        is_player = self.turnOwner == PLAYER
        own_hp, opp_hp = (self.playerHP, self.npcHP) if is_player else (self.npcHP, self.playerHP)
        own_kits = self.playerRepairKits if is_player else self.npcRepairKits
        opp_defending = self.npcDefending if is_player else self.playerDefending
        own_max_hp = self.playerMaxHP if is_player else self.npcSpec.max_hp

        if action == ACT_ATTACK:
            opp_hp -= self._attack_damage(is_player, opp_defending)
            if not is_player and self.npcSpec.suicide:
                own_hp = 0  # kamikaze hancur bersama ledakannya
        elif action == ACT_REPAIR:
            own_hp = min(own_max_hp, own_hp + REPAIR_HEAL_AMOUNT)
            own_kits -= 1
        # ACT_DEFEND: tidak menyerang, tidak memulihkan -- hanya set flag di bawah

        own_defending = (action == ACT_DEFEND)
        next_owner = NPC if is_player else PLAYER

        if is_player:
            return BattleState(
                playerHP=own_hp, npcHP=opp_hp,
                playerRepairKits=own_kits, npcRepairKits=self.npcRepairKits,
                playerDefending=own_defending, npcDefending=opp_defending,
                turnOwner=next_owner, turnIndex=self.turnIndex + 1,
                playerMaxHP=self.playerMaxHP, npcSpec=self.npcSpec,
            )
        return BattleState(
            playerHP=opp_hp, npcHP=own_hp,
            playerRepairKits=self.playerRepairKits, npcRepairKits=own_kits,
            playerDefending=opp_defending, npcDefending=own_defending,
            turnOwner=next_owner, turnIndex=self.turnIndex + 1,
            playerMaxHP=self.playerMaxHP, npcSpec=self.npcSpec,
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


class TreeNode:
    """Satu node di pohon pencarian minimax, dicatat untuk visualisasi debug.

    `action`  -- aksi yang membawa dari parent ke node ini (None untuk root).
    `owner`   -- 'MAX' (giliran NPC) atau 'MIN' (giliran player) di node ini.
    `kind`    -- 'internal' | 'terminal' | 'cutoff' (lihat Cutoff(s, depth) di desain).
    `pruned`  -- True kalau node ini sendiri TIDAK PERNAH dieksplorasi karena
                 alpha-beta memangkas cabang saudaranya (dipakai untuk menggambar
                 cabang yang terpotong, contoh klasik "move ordering").
    `prob`    -- probabilitas cabang ini di node CHANCE (mode Expectimax saja).
    `snapshot`-- salinan ringkas state di node ini (HP/kit reparasi/bertahan/
                 turnIndex), dipakai HANYA untuk panel debug "arahkan mouse
                 ke node".
    """
    __slots__ = (
        "action", "depth", "owner", "value", "kind", "pruned", "prob",
        "snapshot", "children", "x", "y",
    )

    def __init__(self, action, depth, owner=None):
        self.action = action
        self.depth = depth
        self.owner = owner
        self.value = None
        self.kind = None
        self.pruned = False
        self.prob = None
        self.snapshot = None
        self.children = []
        self.x = 0.0
        self.y = 0.0


def _snapshot_of(state):
    return (
        state.playerHP, state.npcHP, state.playerRepairKits, state.npcRepairKits,
        state.playerDefending, state.npcDefending, state.turnIndex,
    )


class MinimaxAgent:
    """Agen NPC: memilih aksi lewat Minimax / Alpha-Beta / Expectimax.

    Mencatat nodes_expanded, terminal_count, cutoff_count, pruned_count &
    root_action_scores -- semuanya dibutuhkan untuk panel debug di BattleHUD --
    dan opsional membangun `last_tree` (TreeNode) untuk divisualisasikan.
    """

    def __init__(self):
        self.nodes_expanded = 0
        self.terminal_count = 0
        self.cutoff_count = 0
        self.pruned_count = 0
        self.root_action_scores = []  # [(action, score), ...] milik keputusan terakhir
        self.last_tree = None         # TreeNode root milik keputusan terakhir (untuk visualisasi)

    def _search(self, state, depth, alpha, beta, max_depth, mode, eval_fn, action_order, node):
        self.nodes_expanded += 1
        if node is not None:
            if state.turnOwner == NPC:
                node.owner = "MAX"
            elif mode == MODE_EXPECTIMAX:
                node.owner = "CHANCE"
            else:
                node.owner = "MIN"
            node.snapshot = _snapshot_of(state)

        # Cutoff(s, depth) = Terminal(s) or (depth >= MAX_DEPTH)
        if state.is_terminal():
            val = battle_utility(state)            # terminal sejati -> utility
            self.terminal_count += 1
            if node is not None:
                node.kind = "terminal"
                node.value = val
            return val
        if depth >= max_depth:
            val = eval_fn(state)                    # cutoff karena depth -> evaluation
            self.cutoff_count += 1
            if node is not None:
                node.kind = "cutoff"
                node.value = val
            return val

        if node is not None:
            node.kind = "internal"

        actions = state.legal_actions(action_order)

        if state.turnOwner == NPC:  # node MAX -- giliran NPC, selalu deterministik
            best = -math.inf
            for i, a in enumerate(actions):
                child = TreeNode(a, depth + 1) if node is not None else None
                val = self._search(state.apply_action(a), depth + 1, alpha, beta, max_depth, mode, eval_fn, action_order, child)
                if child is not None:
                    child.value = val
                    node.children.append(child)
                if val > best:
                    best = val
                if mode == MODE_ALPHABETA:
                    alpha = max(alpha, best)
                    if alpha >= beta:
                        self.pruned_count += len(actions[i + 1:])
                        if node is not None:
                            for remaining in actions[i + 1:]:
                                stub = TreeNode(remaining, depth + 1)
                                stub.pruned = True
                                node.children.append(stub)
                        break  # beta cutoff
            if node is not None:
                node.value = best
            return best

        elif mode == MODE_EXPECTIMAX:  # node CHANCE -- giliran player dimodelkan acak
            # Asumsi desain: NPC TIDAK tahu persis aksi player berikutnya, jadi
            # menganggap semua aksi legal player sama mungkin (uniform) dan
            # memilih aksi yang memaksimalkan NILAI HARAPAN (expected value),
            # bukan skenario terburuk seperti Minimax/Alpha-Beta murni.
            total = 0.0
            prob = 1.0 / len(actions) if actions else 0.0
            for a in actions:
                child = TreeNode(a, depth + 1) if node is not None else None
                val = self._search(state.apply_action(a), depth + 1, -math.inf, math.inf, max_depth, mode, eval_fn, action_order, child)
                if child is not None:
                    child.value = val
                    child.prob = prob
                    node.children.append(child)
                total += prob * val
            if node is not None:
                node.value = total
            return total

        else:  # node MIN -- Minimax biasa atau Alpha-Beta
            best = math.inf
            for i, a in enumerate(actions):
                child = TreeNode(a, depth + 1) if node is not None else None
                val = self._search(state.apply_action(a), depth + 1, alpha, beta, max_depth, mode, eval_fn, action_order, child)
                if child is not None:
                    child.value = val
                    node.children.append(child)
                if val < best:
                    best = val
                if mode == MODE_ALPHABETA:
                    beta = min(beta, best)
                    if beta <= alpha:
                        self.pruned_count += len(actions[i + 1:])
                        if node is not None:
                            for remaining in actions[i + 1:]:
                                stub = TreeNode(remaining, depth + 1)
                                stub.pruned = True
                                node.children.append(stub)
                        break  # alpha cutoff
            if node is not None:
                node.value = best
            return best

    def choose_action(self, state, depth=BATTLE_DEFAULT_DEPTH, mode=MODE_ALPHABETA,
                       eval_fn=battle_eval_default, action_order=None, build_tree=True):
        self.nodes_expanded = 0
        self.terminal_count = 0
        self.cutoff_count = 0
        self.pruned_count = 0
        self.root_action_scores = []
        self.last_tree = None

        alpha, beta = -math.inf, math.inf
        best_action = None
        best_score = -math.inf

        root = TreeNode(None, 0, "MAX" if state.turnOwner == NPC else "MIN") if build_tree else None
        if root is not None:
            root.snapshot = _snapshot_of(state)

        # Catatan: di root sendiri SEMUA aksi tetap dicoba (tidak dipangkas) supaya
        # root_action_scores lengkap untuk overlay debug -- pruning hanya terjadi
        # di level yang lebih dalam.
        for a in state.legal_actions(action_order):
            child_state = state.apply_action(a)
            child_node = TreeNode(a, 1) if build_tree else None
            score = self._search(child_state, 1, alpha, beta, depth, mode, eval_fn, action_order, child_node)
            if child_node is not None:
                child_node.value = score
                root.children.append(child_node)
            self.root_action_scores.append((a, score))
            if score > best_score:
                best_score = score
                best_action = a
            if mode == MODE_ALPHABETA:
                alpha = max(alpha, best_score)

        if build_tree:
            root.value = best_score
            root.kind = "internal"
            self.last_tree = root

        return best_action


# =============================================================================
# BAGIAN 5B.3 -- ENCOUNTER: TANK MELAWAN BANYAK DRONE + BANTUAN MUSUH
# =============================================================================
# Model murni (tanpa pygame) supaya bisa dipakai game DAN simulasi tuning.
# Tiap drone memilih aksi lewat duel 1-lawan-1 (BattleState) melawan tank,
# sehingga seluruh agen minimax/alpha-beta/expectimax lama tetap dipakai apa adanya.

class Drone:
    """Satu drone musuh di medan battle."""
    __slots__ = ("drone_id", "loadout", "hp", "repair_kits", "defending", "charge")

    def __init__(self, drone_id, loadout):
        self.drone_id = drone_id
        self.loadout = loadout
        self.hp = float(loadout.max_hp)
        self.repair_kits = loadout.repair_kits
        self.defending = False
        self.charge = 0

    def is_alive(self):
        return self.hp > 0

    def is_ready(self):
        """True bila drone ini akan bertindak pada fase drone berikutnya."""
        return self.charge + 1 >= self.loadout.act_interval


class Encounter:
    """Satu pertempuran: satu tank vs banyak drone dengan loadout berbeda.

    Urutan satu ronde: tank bertindak -> tiap drone yang sudah siap bertindak
    -> cek bantuan musuh. Bantuan hanya mungkin muncul setelah ronde
    `reinforce_after` (supaya pertempuran yang berlarut-larut makin berat).
    """

    def __init__(self, params, rng):
        self.params = params
        self.rng = rng
        self.loadouts = build_loadouts(params)
        self.tank_max_hp = params.tank_hp
        self.tank_hp = float(params.tank_hp)
        self.tank_repair_kits = BATTLE_START_REPAIR_KITS
        self.tank_defending = False
        self.drones = []
        self.next_drone_id = 1
        self.round_no = 1
        self.reinforcement_count = 0
        self.damage_by_loadout = {key: 0.0 for key in self.loadouts}
        for _ in range(params.initial_drones):
            self._spawn_drone()

    # ---- kueri ----------------------------------------------------------
    def alive_drones(self):
        return [d for d in self.drones if d.is_alive()]

    def is_over(self):
        return self.tank_hp <= 0 or not self.alive_drones()

    def winner(self):
        """PLAYER / NPC, atau None bila pertempuran belum selesai."""
        if not self.is_over():
            return None
        return NPC if self.tank_hp <= 0 else PLAYER

    def incoming_damage_estimate(self):
        """Total damage maksimal yang akan masuk pada fase drone berikutnya."""
        return sum(d.loadout.damage for d in self.alive_drones() if d.is_ready())

    def projection(self, drone, turn_owner):
        """Duel 1-lawan-1 antara tank dan `drone` (dipakai agen minimax)."""
        return BattleState(
            playerHP=self.tank_hp, npcHP=drone.hp,
            playerRepairKits=self.tank_repair_kits, npcRepairKits=drone.repair_kits,
            playerDefending=self.tank_defending, npcDefending=drone.defending,
            turnOwner=turn_owner, turnIndex=self.round_no,
            playerMaxHP=self.tank_max_hp, npcSpec=drone.loadout,
        )

    # ---- aksi -----------------------------------------------------------
    def play_round(self, action, target_id, chooser):
        """Mainkan satu ronde penuh.

        Args:
            action: ACT_ATTACK / ACT_DEFEND / ACT_REPAIR untuk tank.
            target_id: drone_id yang diserang (diabaikan untuk non-serang).
            chooser: fungsi(BattleState)->aksi, memilih aksi drone non-kamikaze.
        Returns:
            list teks log kejadian ronde ini.
        Raises:
            ValueError: pertempuran sudah selesai, kit habis, atau target tidak valid.
        Kompleksitas: O(D * biaya chooser), D = jumlah drone hidup.
        """
        lines = self.player_turn(action, target_id)
        if self.is_over():
            return lines
        lines += self.drones_turn(chooser)
        if self.is_over():
            return lines
        return lines + self.end_round()

    def player_turn(self, action, target_id):
        if self.is_over():
            raise ValueError("Pertempuran sudah berakhir.")
        if action == ACT_REPAIR and self.tank_repair_kits <= 0:
            raise ValueError("Kit reparasi tank habis.")
        target = self._find_target(target_id) if action == ACT_ATTACK else self.alive_drones()[0]
        new_state = self.projection(target, PLAYER).apply_action(action)
        self._absorb(target, new_state)
        return [self._player_line(action, target)]

    def drones_turn(self, chooser):
        lines = []
        queue = self.alive_drones()
        index = 0
        # while (bukan for+break): berhenti begitu tank hancur, sisa drone tidak perlu bertindak.
        while index < len(queue) and self.tank_hp > 0:
            lines += self._drone_act(queue[index], chooser)
            index += 1
        return lines

    def end_round(self):
        """Naikkan nomor ronde dan putuskan apakah bantuan musuh tiba."""
        self.round_no += 1
        if not self._should_reinforce():
            return []
        drone = self._spawn_drone()
        self.reinforcement_count += 1
        return [f"BANTUAN! Drone {drone.loadout.name} #{drone.drone_id} tiba."]

    # ---- internal -------------------------------------------------------
    def _spawn_drone(self):
        keys = list(LOADOUT_SPAWN_WEIGHTS)
        weights = [LOADOUT_SPAWN_WEIGHTS[k] for k in keys]
        key = self.rng.choices(keys, weights=weights, k=1)[0]
        drone = Drone(self.next_drone_id, self.loadouts[key])
        self.next_drone_id += 1
        self.drones.append(drone)
        return drone

    def _should_reinforce(self):
        if self.round_no <= self.params.reinforce_after:
            return False
        if len(self.alive_drones()) >= MAX_ACTIVE_DRONES:
            return False
        if self.reinforcement_count >= MAX_REINFORCEMENTS:
            return False
        return self.rng.random() * PERCENT_BASE < self.params.reinforce_chance_pct

    def _find_target(self, target_id):
        matches = [d for d in self.alive_drones() if d.drone_id == target_id]
        if not matches:
            raise ValueError(f"Target drone #{target_id} tidak ada atau sudah hancur.")
        return matches[0]

    def _absorb(self, drone, new_state):
        """Salin hasil duel 1-lawan-1 kembali ke tank & drone sungguhan."""
        tank_damage = self.tank_hp - new_state.playerHP
        if tank_damage > 0:
            self.damage_by_loadout[drone.loadout.key] += tank_damage
        self.tank_hp = new_state.playerHP
        self.tank_repair_kits = new_state.playerRepairKits
        self.tank_defending = new_state.playerDefending
        drone.hp = new_state.npcHP
        drone.repair_kits = new_state.npcRepairKits
        drone.defending = new_state.npcDefending

    def _drone_act(self, drone, chooser):
        drone.charge += 1
        if drone.charge < drone.loadout.act_interval:
            return []  # masih mengisi daya; tidak ada kejadian untuk dilaporkan
        drone.charge = 0
        state = self.projection(drone, NPC)
        action = ACT_ATTACK if drone.loadout.suicide else chooser(state)
        if action not in (ACT_ATTACK, ACT_DEFEND, ACT_REPAIR):
            raise ValueError(f"Aksi drone tidak dikenal: {action!r}")
        self._absorb(drone, state.apply_action(action))
        return [self._drone_line(drone, action)]

    def _player_line(self, action, target):
        if action == ACT_ATTACK:
            return f"Tank: Serang #{target.drone_id} {target.loadout.name} (HP {max(0, target.hp):.0f})"
        return f"Tank: {ACTION_LABELS[action]} (HP {max(0, self.tank_hp):.0f})"

    def _drone_line(self, drone, action):
        label = "MELEDAK" if drone.loadout.suicide and action == ACT_ATTACK else ACTION_LABELS[action]
        return f"#{drone.drone_id} {drone.loadout.name}: {label} (HP tank {max(0, self.tank_hp):.0f})"


# =============================================================================
# BAGIAN 5B.4 -- LOCAL SEARCH UNTUK KESEIMBANGAN (tidak terlalu mudah/sulit)
# =============================================================================
# Ruang pencarian: vektor BalanceParams. Fungsi objektif (loss) mengukur seberapa
# jauh hasil simulasi dari "pita nyaman": pemain biasa menang cukup sering
# (tidak frustasi) tapi tidak selalu (tidak bosan), pemain mahir jelas lebih
# unggul, pertempuran tidak terlalu singkat/panjang, bantuan musuh sesekali
# muncul, dan tiap loadout benar-benar berkontribusi.

CASUAL_WIN_BAND = (0.35, 0.55)
SKILLED_WIN_BAND = (0.65, 0.85)
ROUNDS_BAND = (10.0, 22.0)
ROUNDS_SCALE = 10.0               # normalisasi pelanggaran ronde ke skala win-rate
REINFORCE_RATE_BAND = (0.25, 0.60)
MIN_DAMAGE_SHARE = 0.20           # tiap loadout minimal menyumbang 20% damage ke tank

WEIGHT_WIN = 4.0
WEIGHT_ROUNDS = 1.0
WEIGHT_REINFORCE = 1.0
WEIGHT_SHARE = 2.0
WEIGHT_TIMEOUT = 4.0

SIM_ROUND_LIMIT = 60              # pengaman simulasi: lewat ini dianggap macet
CASUAL_REPAIR_HP_RATIO = 0.3
CASUAL_REPAIR_PROB = 0.6
CASUAL_DEFEND_PROB = 0.2
SKILLED_DANGER_FACTOR = 1.5       # waspada bila HP <= 1.5x damage masuk yang diperkirakan

LS_BATTLES_PER_POLICY = 60
LS_MAX_FAILURES = 25              # berhenti bila 25 tetangga berturut-turut tidak lebih baik
LS_RESTARTS = 3
LS_SEED = 2026
LS_EVAL_SEED = 7000
VALIDATION_BATTLES = 400
VALIDATION_SEED = 99000
TUNE_FLAG = "--tune"
NEIGHBOR_MAX_STEPS = 2
NEIGHBOR_RETRY_LIMIT = 20

# nama field -> (min, max, langkah)
PARAM_SPACE = {
    "tank_hp": (120, 300, 10),
    "initial_drones": (2, 5, 1),
    "bom_hp": (40, 120, 5),
    "bom_damage": (15, 45, 2),
    "misil_hp": (30, 90, 5),
    "misil_damage": (10, 30, 1),
    "kamikaze_hp": (10, 50, 5),
    "kamikaze_damage": (20, 60, 3),
    "reinforce_after": (3, 15, 1),
    "reinforce_chance_pct": (5, 60, 5),
}


@dataclass(frozen=True)
class BattleOutcome:
    player_won: bool
    timed_out: bool
    rounds: int
    reinforced: bool
    damage_by_loadout: dict


@dataclass(frozen=True)
class BalanceStats:
    casual_win_rate: float
    skilled_win_rate: float
    avg_rounds: float
    reinforce_rate: float
    timeout_rate: float
    min_damage_share: float


class CachedChooser:
    """Pemilih aksi drone via minimax/alpha-beta dengan memo per state.

    Memo membuat ribuan simulasi tuning jauh lebih cepat tanpa mengubah
    keputusan (agen yang sama, state yang sama -> aksi yang sama)."""

    def __init__(self, depth=BATTLE_DEFAULT_DEPTH, mode=MODE_ALPHABETA, eval_key="seimbang"):
        self.depth = depth
        self.mode = mode
        self.eval_fn = EVAL_FUNCTIONS[eval_key][0]
        self.agent = MinimaxAgent()
        self.cache = {}

    def __call__(self, state):
        key = (state.playerHP, state.npcHP, state.playerRepairKits, state.npcRepairKits,
               state.playerDefending, state.npcDefending, state.playerMaxHP, state.npcSpec)
        if key not in self.cache:
            self.cache[key] = self.agent.choose_action(
                state, depth=self.depth, mode=self.mode, eval_fn=self.eval_fn, build_tree=False)
        return self.cache[key]


def casual_policy(enc, rng):
    """Pemain biasa: kadang reparasi saat HP rendah, kadang bertahan, sisanya
    menyerang drone acak (tidak memprioritaskan target).

    Returns: (aksi, target_id). Kompleksitas O(D)."""
    target = rng.choice(enc.alive_drones())
    low_hp = enc.tank_hp < CASUAL_REPAIR_HP_RATIO * enc.tank_max_hp
    if low_hp and enc.tank_repair_kits > 0 and rng.random() < CASUAL_REPAIR_PROB:
        return ACT_REPAIR, target.drone_id
    if rng.random() < CASUAL_DEFEND_PROB:
        return ACT_DEFEND, target.drone_id
    return ACT_ATTACK, target.drone_id


def skilled_policy(enc, rng):
    """Pemain mahir: baca ancaman, reparasi/bertahan tepat waktu, dan
    memprioritaskan kamikaze lalu drone dengan HP terendah.

    `rng` tidak dipakai (deterministik) -- ada demi antarmuka yang sama dengan
    casual_policy. Returns: (aksi, target_id). Kompleksitas O(D)."""
    alive = enc.alive_drones()
    target = min(alive, key=lambda d: (not d.loadout.suicide, d.hp))
    in_danger = enc.tank_hp <= enc.incoming_damage_estimate() * SKILLED_DANGER_FACTOR
    can_heal_fully = enc.tank_hp <= enc.tank_max_hp - REPAIR_HEAL_AMOUNT
    if in_danger and enc.tank_repair_kits > 0 and can_heal_fully:
        return ACT_REPAIR, target.drone_id
    if in_danger:
        return ACT_DEFEND, target.drone_id
    return ACT_ATTACK, target.drone_id


def simulate_battle(params, policy, rng, chooser):
    """Simulasikan satu pertempuran sampai selesai (atau macet).

    Returns: BattleOutcome. Kompleksitas O(R * D * biaya chooser)."""
    enc = Encounter(params, rng)
    while not enc.is_over() and enc.round_no <= SIM_ROUND_LIMIT:
        action, target_id = policy(enc, rng)
        enc.play_round(action, target_id, chooser)
    return BattleOutcome(
        player_won=enc.winner() == PLAYER,
        timed_out=not enc.is_over(),
        rounds=enc.round_no,
        reinforced=enc.reinforcement_count > 0,
        damage_by_loadout=dict(enc.damage_by_loadout),
    )


def _run_policy(params, policy, battles, base_seed, chooser):
    return [
        simulate_battle(params, policy, random.Random(base_seed + i), chooser)
        for i in range(battles)
    ]


def _rate(outcomes, predicate):
    return sum(1 for o in outcomes if predicate(o)) / len(outcomes)


def _min_damage_share(outcomes):
    totals = {}
    for o in outcomes:
        for key, dmg in o.damage_by_loadout.items():
            totals[key] = totals.get(key, 0.0) + dmg
    grand_total = sum(totals.values())
    if grand_total <= 0:
        return 0.0
    return min(totals.values()) / grand_total


def evaluate_params(params, battles, base_seed, chooser):
    """Ukur BalanceStats dari `battles` simulasi per kebijakan pemain.

    Seed tetap (base_seed + i) -> parameter yang sama selalu menghasilkan
    loss yang sama, sehingga local search tidak tertipu noise.
    Kompleksitas O(battles * biaya simulate_battle)."""
    if battles < 1:
        raise ValueError("battles harus >= 1")
    casual = _run_policy(params, casual_policy, battles, base_seed, chooser)
    skilled = _run_policy(params, skilled_policy, battles, base_seed, chooser)
    both = casual + skilled
    return BalanceStats(
        casual_win_rate=_rate(casual, lambda o: o.player_won),
        skilled_win_rate=_rate(skilled, lambda o: o.player_won),
        avg_rounds=sum(o.rounds for o in both) / len(both),
        reinforce_rate=_rate(both, lambda o: o.reinforced),
        timeout_rate=_rate(both, lambda o: o.timed_out),
        min_damage_share=_min_damage_share(both),
    )


def band_violation(value, low, high):
    """Jarak `value` ke luar pita [low, high]; 0 bila di dalam pita."""
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def balance_loss(stats):
    """Fungsi objektif local search: 0 = semua sasaran keseimbangan terpenuhi."""
    win_gap = (band_violation(stats.casual_win_rate, *CASUAL_WIN_BAND) ** 2
               + band_violation(stats.skilled_win_rate, *SKILLED_WIN_BAND) ** 2)
    rounds_gap = (band_violation(stats.avg_rounds, *ROUNDS_BAND) / ROUNDS_SCALE) ** 2
    reinforce_gap = band_violation(stats.reinforce_rate, *REINFORCE_RATE_BAND) ** 2
    share_gap = band_violation(stats.min_damage_share, MIN_DAMAGE_SHARE, 1.0) ** 2
    return (WEIGHT_WIN * win_gap + WEIGHT_ROUNDS * rounds_gap + WEIGHT_REINFORCE * reinforce_gap
            + WEIGHT_SHARE * share_gap + WEIGHT_TIMEOUT * stats.timeout_rate ** 2)


def random_params(rng):
    """Titik awal acak (untuk random restart) pada grid PARAM_SPACE."""
    values = {}
    for name, (low, high, step) in PARAM_SPACE.items():
        values[name] = low + step * rng.randint(0, (high - low) // step)
    return BalanceParams(**values)


def random_neighbor(params, rng):
    """Tetangga: ubah SATU parameter sebesar +-1..NEIGHBOR_MAX_STEPS langkah (terpotong ke batas).

    Returns: BalanceParams berbeda dari `params` (atau sama bila semua percobaan
    terpotong batas -- sangat jarang)."""
    neighbor = params
    attempts = 0
    while neighbor == params and attempts < NEIGHBOR_RETRY_LIMIT:
        attempts += 1
        name = rng.choice(list(PARAM_SPACE))
        low, high, step = PARAM_SPACE[name]
        shift = rng.choice([-1, 1]) * rng.randint(1, NEIGHBOR_MAX_STEPS) * step
        new_value = max(low, min(high, getattr(params, name) + shift))
        neighbor = BalanceParams(**{**params.__dict__, name: new_value})
    return neighbor


def hill_climb(start, rng, evaluate, max_failures=LS_MAX_FAILURES, log=print):
    """Stochastic first-improvement hill climbing.

    Args:
        start: BalanceParams awal.
        evaluate: fungsi(params)->loss (makin kecil makin baik).
        max_failures: berhenti setelah sekian tetangga berturut-turut tidak membaik.
    Returns: (params_terbaik, loss_terbaik).
    Kompleksitas: O(langkah * biaya evaluate); berhenti di optimum lokal."""
    current, current_loss = start, evaluate(start)
    failures = 0
    while failures < max_failures and current_loss > 0:
        candidate = random_neighbor(current, rng)
        candidate_loss = evaluate(candidate)
        if candidate_loss < current_loss:
            current, current_loss, failures = candidate, candidate_loss, 0
            log(f"    loss={current_loss:.4f}  {current}")
        else:
            failures += 1
    return current, current_loss


def run_local_search(restarts=LS_RESTARTS, seed=LS_SEED, log=print):
    """Hill climbing dengan random restart; mengembalikan (params, loss) terbaik.

    Restart pertama mulai dari TUNED_PARAMS, sisanya dari titik acak, agar
    tidak terjebak satu optimum lokal. Kompleksitas O(restarts * hill_climb)."""
    if restarts < 1:
        raise ValueError("restarts harus >= 1")
    rng = random.Random(seed)
    chooser = CachedChooser()

    def evaluate(params):
        return balance_loss(evaluate_params(params, LS_BATTLES_PER_POLICY, LS_EVAL_SEED, chooser))

    starts = [TUNED_PARAMS] + [random_params(rng) for _ in range(restarts - 1)]
    results = []
    for index, start in enumerate(starts, start=1):
        log(f"[Restart {index}/{restarts}] mulai: {start}")
        results.append(hill_climb(start, rng, evaluate, log=log))
    return min(results, key=lambda r: r[1])


def format_stats(stats):
    return (f"menang pemain biasa={stats.casual_win_rate:.0%}  pemain mahir={stats.skilled_win_rate:.0%}  "
            f"rata-rata ronde={stats.avg_rounds:.1f}  bantuan muncul={stats.reinforce_rate:.0%}  "
            f"macet={stats.timeout_rate:.0%}  porsi damage terkecil={stats.min_damage_share:.0%}")


def run_tuning_cli(args):
    """Entry point `python <file> --tune [jumlah_restart]`."""
    flag_index = args.index(TUNE_FLAG)
    has_count = flag_index + 1 < len(args)
    restarts = int(args[flag_index + 1]) if has_count else LS_RESTARTS
    best, loss = run_local_search(restarts)
    stats = evaluate_params(best, VALIDATION_BATTLES, VALIDATION_SEED, CachedChooser())
    print(f"\nParameter terbaik (loss pencarian={loss:.4f}):\n  {best}")
    print(f"Validasi {VALIDATION_BATTLES} pertempuran/kebijakan (seed baru):\n  {format_stats(stats)}")
    print("\nSalin ke TUNED_PARAMS:\n  TUNED_PARAMS = " + repr(best))


# =============================================================================
# BAGIAN 5B.5 -- LOCAL SEARCH NPC (HILL CLIMBING & SIMULATED ANNEALING)
# =============================================================================
# Beda dengan local search keseimbangan (BAGIAN 5B.4) yang menyetel angka
# permainan secara offline, di sini local search dipakai SAAT BERTEMPUR untuk
# memilih aksi drone. Ruang pencariannya adalah sebuah RENCANA: satu jenis
# aksi (ATTACK / DEFEND / REPAIR) untuk tiap ronde selama `horizon` ronde.
#   - State awal   : rencana "semua serang" (restart pertama) atau acak.
#   - Neighbor     : ubah SATU slot rencana ke jenis aksi lain.
#   - Fungsi skor  : simulasikan rencana melawan model tank tetap, lalu nilai
#                    dengan utility (terminal) atau fungsi evaluasi (cutoff).
# Hanya aksi pertama rencana yang dijalankan; rencana dihitung ulang di tiap
# giliran drone (receding horizon), sehingga perubahan HP/kit akibat drone lain
# atau bantuan musuh yang datang acak otomatis ikut diperhitungkan.
PLAN_KINDS = (ACT_ATTACK, ACT_DEFEND, ACT_REPAIR)

LOCAL_DEFAULT_HORIZON = 3       # jumlah ronde ke depan yang direncanakan
LOCAL_EVAL_BUDGET = 150         # total evaluasi rencana per keputusan (sama untuk HC & SA)
HILL_CLIMB_RESTARTS = 3         # jumlah restart Hill Climbing (anggaran dibagi rata)
ANNEAL_START_TEMP = 60.0        # suhu awal SA, sepadan dengan selisih skor antar-aksi
ANNEAL_COOLING_RATE = 0.97      # faktor pendinginan geometrik per iterasi
TANK_MODEL_REPAIR_RATIO = 0.35  # model tank: reparasi bila HP <= rasio ini dari HP maksimum
EXPERIMENT_SEED = 7             # seed RNG eksperimen supaya perbandingan local search adil
ADVERSARIAL_EXPERIMENT_HEADERS = ("Kombinasi", "Aksi", "Skor", "Node", "Terminal/Cutoff", "Dipangkas", "ms")
LOCAL_EXPERIMENT_HEADERS = ("Kombinasi", "Aksi", "Skor", "Evaluasi", "Restart", "Diterima", "ms")


def tank_model_action(state):
    """Model lawan tetap untuk simulasi rencana.

    Reparasi saat HP tank rendah dan kit masih ada, selain itu menyerang.
    Ini asumsi desain, bukan prediksi pasti perilaku pemain.

    Parameter: state (BattleState giliran tank).
    Kembalian: jenis aksi (ACT_*).
    Kompleksitas: waktu O(1), memori O(1).
    """
    low_hp = state.playerHP <= state.playerMaxHP * TANK_MODEL_REPAIR_RATIO
    if low_hp and state.playerRepairKits > 0:
        return ACT_REPAIR
    return ACT_ATTACK


def legalize_npc_action(state, kind):
    """Ubah jenis aksi dari rencana menjadi aksi legal untuk drone.

    Aksi yang tidak legal (kit habis) diganti DEFEND, yang selalu legal,
    supaya rencana apa pun tetap bisa disimulasikan tanpa error.

    Parameter: state (BattleState giliran NPC), kind (salah satu PLAN_KINDS).
    Kembalian: jenis aksi legal (ACT_*).
    Kompleksitas: waktu O(1), memori O(1).
    """
    return kind if kind in state.legal_actions() else ACT_DEFEND


def _play_planned_round(state, kind):
    """Satu ronde simulasi: aksi drone dari rencana, lalu balasan model tank.

    Tank tidak membalas bila drone sudah memenangkan duel di aksinya sendiri,
    sebab aksi pada state terminal tidak bermakna.
    """
    after_drone = state.apply_action(legalize_npc_action(state, kind))
    if after_drone.is_terminal():
        return after_drone
    return after_drone.apply_action(tank_model_action(after_drone))


def evaluate_plan(state, plan, eval_fn):
    """Skor sebuah rencana: simulasi len(plan) ronde, lalu utility / evaluasi.

    Parameter:
        state   : BattleState giliran NPC (titik awal simulasi).
        plan    : tuple jenis aksi, indeks slot = ronde ke depan.
        eval_fn : fungsi evaluasi untuk state non-terminal di ujung horizon.
    Kembalian: skor float dari sudut pandang NPC (tinggi = drone unggul).
    Kompleksitas: waktu O(len(plan)), memori O(1).
    """
    current = state
    offset = 0
    while not current.is_terminal() and offset < len(plan):
        current = _play_planned_round(current, plan[offset])
        offset += 1
    return battle_utility(current) if current.is_terminal() else eval_fn(current)


class LocalSearchAgent:
    """Agen NPC berbasis local search: Hill Climbing (random restart) atau Simulated Annealing.

    Kedua algoritma memakai ruang rencana, neighbor, dan fungsi skor yang sama, dengan
    anggaran evaluasi yang sama (LOCAL_EVAL_BUDGET) supaya adil dibandingkan.

    Mencatat evaluations, accepted_moves, restarts_done, best_score, root_action_scores,
    serta best_trace / current_trace (untuk grafik konvergensi di HUD).
    """

    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self._reset(None, 0)

    def _reset(self, state, horizon):
        self.root_state = state
        self.horizon = horizon
        self.evaluations = 0
        self.accepted_moves = 0
        self.restarts_done = 0
        self.best_plan = ()
        self.best_score = -math.inf
        self.best_trace = []          # skor terbaik sejauh ini, per iterasi
        self.current_trace = []       # skor rencana yang sedang dipegang, per iterasi
        self.root_action_scores = []  # [(aksi, skor terbaik), ...] urut menurun
        self._action_best = {}

    def _score(self, state, plan, eval_fn):
        """Evaluasi satu rencana dan catat skor terbaik untuk aksi pertamanya."""
        score = evaluate_plan(state, plan, eval_fn)
        self.evaluations += 1
        first_action = legalize_npc_action(state, plan[0])
        self._action_best[first_action] = max(score, self._action_best.get(first_action, -math.inf))
        return score

    def _track(self, held_plan, held_score):
        """Perbarui rencana terbaik dan rekam satu titik grafik konvergensi."""
        if held_score > self.best_score:
            self.best_score = held_score
            self.best_plan = held_plan
        self.best_trace.append(self.best_score)
        self.current_trace.append(held_score)

    def _initial_plan(self, restart):
        """Restart pertama = baseline 'semua serang' (deterministik), sisanya acak."""
        if restart == 0:
            return (ACT_ATTACK,) * self.horizon
        return tuple(self.rng.choice(PLAN_KINDS) for _ in range(self.horizon))

    def _neighbor(self, plan):
        """Neighbor: satu slot diganti ke jenis aksi lain."""
        slot = self.rng.randrange(len(plan))
        kind = self.rng.choice([candidate for candidate in PLAN_KINDS if candidate != plan[slot]])
        return plan[:slot] + (kind,) + plan[slot + 1:]

    def _accept_annealing(self, delta, temperature):
        """Aturan Metropolis: selalu terima yang tidak lebih buruk, yang lebih buruk dengan peluang e^(delta/T)."""
        return delta >= 0 or self.rng.random() < math.exp(delta / temperature)

    def _hill_climb(self, state, eval_fn):
        """Stochastic hill climbing + random restart.

        Neighbor hanya diterima kalau skornya LEBIH baik, jadi terjebak di optimum
        lokal; restart acak memberi kesempatan keluar dari jebakan itu.
        Kompleksitas: waktu O(LOCAL_EVAL_BUDGET * horizon), memori O(horizon).
        """
        steps_per_restart = LOCAL_EVAL_BUDGET // HILL_CLIMB_RESTARTS
        for restart in range(HILL_CLIMB_RESTARTS):
            plan = self._initial_plan(restart)
            score = self._score(state, plan, eval_fn)
            self._track(plan, score)
            step = 1
            while step < steps_per_restart:
                candidate = self._neighbor(plan)
                candidate_score = self._score(state, candidate, eval_fn)
                if candidate_score > score:
                    plan, score = candidate, candidate_score
                    self.accepted_moves += 1
                self._track(plan, score)
                step += 1
            self.restarts_done += 1

    def _anneal(self, state, eval_fn):
        """Simulated annealing dengan pendinginan geometrik.

        Suhu tinggi di awal menerima neighbor yang lebih buruk (menjelajah);
        suhu turun sehingga di akhir perilakunya mendekati hill climbing.
        Kompleksitas: waktu O(LOCAL_EVAL_BUDGET * horizon), memori O(horizon).
        """
        plan = self._initial_plan(0)
        score = self._score(state, plan, eval_fn)
        self._track(plan, score)
        temperature = ANNEAL_START_TEMP
        step = 1
        while step < LOCAL_EVAL_BUDGET:
            candidate = self._neighbor(plan)
            candidate_score = self._score(state, candidate, eval_fn)
            if self._accept_annealing(candidate_score - score, temperature):
                plan, score = candidate, candidate_score
                self.accepted_moves += 1
            self._track(plan, score)
            temperature *= ANNEAL_COOLING_RATE
            step += 1
        self.restarts_done = 1

    def choose_action(self, state, horizon=LOCAL_DEFAULT_HORIZON, mode=MODE_HILL,
                      eval_fn=battle_eval_default):
        """Pilih aksi drone lewat local search.

        Parameter:
            state   : BattleState giliran NPC (duel belum selesai).
            horizon : jumlah ronde rencana (>= 1).
            mode    : MODE_HILL atau MODE_ANNEAL.
            eval_fn : fungsi evaluasi (kepribadian NPC).
        Kembalian: jenis aksi (ACT_*) dari rencana terbaik yang ditemukan.
        Raises: ValueError jika mode bukan local search, horizon < 1, atau state
                bukan giliran NPC / sudah terminal.
        Kompleksitas: waktu O(LOCAL_EVAL_BUDGET * horizon), memori O(horizon).
        """
        if mode not in LOCAL_MODES:
            raise ValueError(f"Mode local search tidak dikenal: {mode}")
        if horizon < 1:
            raise ValueError("Horizon rencana minimal 1 ronde.")
        if state.turnOwner != NPC or state.is_terminal():
            raise ValueError("Local search hanya dipanggil saat giliran drone dan duel belum selesai.")

        self._reset(state, horizon)
        run_search = self._hill_climb if mode == MODE_HILL else self._anneal
        run_search(state, eval_fn)

        self.root_action_scores = sorted(self._action_best.items(), key=lambda item: -item[1])
        return legalize_npc_action(state, self.best_plan[0])


# -----------------------------------------------------------------------
# Visualisasi pohon pencarian (debug): layout + gambar TreeNode sebagai
# kotak-kotak yang terhubung garis, mirip diagram pohon minimax di textbook.
# Kedalaman yang digambar (vis_depth) sengaja dibatasi terpisah dari
# kedalaman pencarian sesungguhnya (bisa sampai 6) supaya tetap terbaca --
# node di batas vis_depth yang masih punya anak diberi label "+N cabang...".
# Ukuran node dan jarak antar-baris menyesuaikan tinggi area, dan kedalaman
# tampil diturunkan otomatis bila lebar/tinggi area tidak cukup, sehingga
# pohon selalu utuh walau jendela tidak full screen.
# -----------------------------------------------------------------------
TREE_ROW_GAP = 60              # jarak antar-baris maksimum (saat area lega)
TREE_NODE_H = 34               # tinggi node maksimum
TREE_MIN_NODE_H = 30           # tinggi node minimum; dua baris teks font 12 masih muat
TREE_MIN_ROW_SPACING = 6       # celah minimum antar-baris node agar garis tetap terlihat
TREE_MARGIN_X = 40
TREE_TOP_PADDING = 20
TREE_BOTTOM_RESERVED = 36      # ruang bawah: label "+N cabang" (14) + hint/catatan (18) + cadangan
TREE_MIN_NODE_W = 30
TREE_NODE_GAP_X = 12


def _tree_visible_children(node, vis_depth):
    if node.depth >= vis_depth:
        return []
    return node.children


def _tree_count_leaves(node, vis_depth):
    children = _tree_visible_children(node, vis_depth)
    if not children:
        return 1
    return sum(_tree_count_leaves(c, vis_depth) for c in children)


def _tree_fit_vertical(depth, area_h):
    """Cari ukuran node & jarak baris agar pohon `depth` tingkat muat di tinggi `area_h`.

    Node dicoba dari ukuran terbesar dulu, lalu diperkecil ke minimum. Ini
    menghindari pengurangan kedalaman tampilan kalau cukup dengan memperkecil
    jarak antar-baris.

    Args:
        depth: jumlah tingkat di bawah root yang akan digambar (>= 1).
        area_h: tinggi area gambar pohon dalam piksel.

    Returns:
        (node_h, row_gap) jika muat, atau None jika tidak muat.

    Kompleksitas: O(1) waktu dan memori.
    """
    usable = area_h - TREE_TOP_PADDING - TREE_BOTTOM_RESERVED
    for node_h in (TREE_NODE_H, TREE_MIN_NODE_H):
        row_gap = (usable - node_h) / depth
        if row_gap >= node_h + TREE_MIN_ROW_SPACING:
            return node_h, min(TREE_ROW_GAP, row_gap)
    return None


def _tree_best_fit_depth(root, requested_depth, max_leaves, area_h):
    """Turunkan kedalaman tampilan sampai pohon muat DI LEBAR maupun DI TINGGI area.

    Args:
        root: TreeNode akar pohon.
        requested_depth: kedalaman tampilan yang diminta pengguna.
        max_leaves: jumlah daun maksimum yang muat di lebar area.
        area_h: tinggi area gambar pohon dalam piksel.

    Returns:
        (depth, node_h, row_gap) -- kedalaman efektif beserta ukuran vertikalnya.

    Kompleksitas: O(D * N) waktu (D = kedalaman diminta, N = jumlah node
    tercatat, karena tiap percobaan menghitung daun), O(D) memori rekursi.
    """
    depth = max(1, min(requested_depth, root.depth + 6))
    while depth > 1 and (
        _tree_count_leaves(root, depth) > max_leaves
        or _tree_fit_vertical(depth, area_h) is None
    ):
        depth -= 1

    # Fallback untuk area yang sangat pendek: depth 1 dengan ukuran terkecil,
    # agar tetap ada gambar alih-alih error.
    layout = _tree_fit_vertical(depth, area_h) or (TREE_MIN_NODE_H, TREE_MIN_NODE_H + TREE_MIN_ROW_SPACING)
    return depth, layout[0], layout[1]


def _tree_assign_positions(node, vis_depth, node_w, gap, leaf_counter,
                            origin_x, origin_y, node_h, row_gap):
    node.y = origin_y + node_h / 2.0 + node.depth * row_gap
    children = _tree_visible_children(node, vis_depth)
    if not children:
        node.x = origin_x + leaf_counter[0] * (node_w + gap) + node_w / 2.0
        leaf_counter[0] += 1
        return node.x
    xs = [
        _tree_assign_positions(c, vis_depth, node_w, gap, leaf_counter, origin_x, origin_y, node_h, row_gap)
        for c in children
    ]
    node.x = sum(xs) / len(xs)
    return node.x


ACTION_LABELS = {ACT_ATTACK: "Serang", ACT_DEFEND: "Bertahan", ACT_REPAIR: "Reparasi", None: "ROOT"}
MODE_LABELS = {
    MODE_MINIMAX: "Minimax", MODE_ALPHABETA: "Alpha-Beta", MODE_EXPECTIMAX: "Expectimax",
    MODE_HILL: "Hill Climbing", MODE_ANNEAL: "Simulated Annealing",
}
KIND_LABELS = {"terminal": "terminal (utility)", "cutoff": "cutoff (evaluasi)", "internal": "internal"}


def _draw_node_tooltip(screen, fonts, node, mouse_pos):
    """Panel debug: detail satu node pohon saat mouse diarahkan ke sana --
    action, pemilik (MAX/MIN/CHANCE), kedalaman, jenis node, nilai, DAN
    snapshot state (HP/kit reparasi/status bertahan) di titik itu."""
    font = fonts["small"]
    font_bold = fonts["ui_bold"]
    lines = []
    label = ACTION_LABELS.get(node.action, node.action or "ROOT")
    if node.pruned:
        lines.append((f"{label}  (DIPANGKAS alpha-beta)", COLORS["danger"]))
    else:
        owner_txt = {"MAX": "MAX (giliran NPC)", "MIN": "MIN (giliran Player)", "CHANCE": "CHANCE (Expectimax)"}.get(node.owner, "?")
        lines.append((f"Aksi: {label}", COLORS["text_cream"]))
        lines.append((f"Pemilik node: {owner_txt}", COLORS["text_dim"]))
        lines.append((f"Kedalaman: {node.depth}   Jenis: {KIND_LABELS.get(node.kind, node.kind)}", COLORS["text_dim"]))
        val_txt = f"{node.value:.2f}" if node.value is not None else "?"
        if node.prob is not None:
            val_txt += f"   (p={node.prob:.2f})"
        lines.append((f"Nilai: {val_txt}", COLORS["accent_hover"]))
        if node.snapshot:
            php, nhp, prep, nrep, pdef, ndef, tidx = node.snapshot
            lines.append((f"HP Player/NPC: {int(php)} / {int(nhp)}   Giliran ke-{tidx}", COLORS["text_dim"]))
            lines.append((
                f"Reparasi P/N: {prep}/{nrep}   Bertahan P:{'Y' if pdef else 'n'} N:{'Y' if ndef else 'n'}",
                COLORS["text_dim"],
            ))

    pad = 8
    w = max(font.size(t)[0] for t, _ in lines) + pad * 2
    h = len(lines) * 15 + pad * 2
    x, y = mouse_pos[0] + 14, mouse_pos[1] + 14
    x = min(x, screen.get_width() - w - 4)
    y = min(y, screen.get_height() - h - 4)
    box = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, (16, 24, 13), box, border_radius=6)
    pygame.draw.rect(screen, COLORS["accent"], box, width=1, border_radius=6)
    for i, (text, color) in enumerate(lines):
        t = font.render(text, True, color)
        screen.blit(t, (box.x + pad, box.y + pad + i * 15))


def draw_search_tree(screen, fonts, root, vis_depth, area, mouse_pos=None):
    """Gambar pohon pencarian NPC (root = TreeNode) di dalam `area` (pygame.Rect).

    Kedalaman tampilan, tinggi node, dan jarak antar-baris disesuaikan otomatis
    dengan lebar DAN tinggi `area`, sehingga pohon selalu utuh walau jendela
    tidak full screen. Kalau `mouse_pos` berada di atas sebuah node, node itu
    disorot dan detailnya ditampilkan lewat tooltip (panel debug pohon).
    """
    font_small = fonts["small"]

    if root is None:
        t = fonts["ui"].render("Belum ada pohon pencarian -- tunggu NPC mengambil keputusan.", True, COLORS["text_dim"])
        screen.blit(t, (area.x + 20, area.y + 20))
        return

    avail_w = max(200, area.w - 2 * TREE_MARGIN_X)
    max_leaves = max(1, avail_w // (TREE_MIN_NODE_W + TREE_NODE_GAP_X))
    eff_depth, node_h, row_gap = _tree_best_fit_depth(root, vis_depth, max_leaves, area.h)

    leaves = _tree_count_leaves(root, eff_depth)
    node_w = int(max(TREE_MIN_NODE_W, min(96, avail_w / leaves - TREE_NODE_GAP_X)))

    leaf_counter = [0]
    _tree_assign_positions(
        root, eff_depth, node_w, TREE_NODE_GAP_X, leaf_counter,
        area.x + TREE_MARGIN_X, area.y + TREE_TOP_PADDING, node_h, row_gap,
    )

    show_text = node_w >= 40
    hover_node = [None]

    def draw_edges(node):
        for c in _tree_visible_children(node, eff_depth):
            col = (80, 100, 64) if not c.pruned else (66, 58, 52)
            pygame.draw.line(
                screen, col,
                (node.x, node.y + node_h / 2), (c.x, c.y - node_h / 2),
                1,
            )
            draw_edges(c)

    draw_edges(root)

    def draw_node(node):
        rect = pygame.Rect(0, 0, node_w, node_h)
        rect.center = (int(node.x), int(node.y))
        is_hover = mouse_pos is not None and rect.collidepoint(mouse_pos)
        if is_hover:
            hover_node[0] = node

        if node.pruned:
            bg, border = (38, 44, 34), (110, 70, 62)
        elif node.owner == "MAX":
            bg, border = (46, 39, 30), COLORS["drone"]
        elif node.owner == "CHANCE":
            bg, border = (42, 34, 46), (150, 110, 200)
        else:
            bg, border = (32, 52, 30), COLORS["tank"]
        pygame.draw.rect(screen, bg, rect, border_radius=6)
        width = (3 if is_hover else 2) if node.kind == "terminal" else (2 if is_hover else 1)
        pygame.draw.rect(screen, (COLORS["accent_hover"] if is_hover else border), rect, width=width, border_radius=6)

        if show_text:
            label = ACTION_LABELS.get(node.action, node.action or "ROOT")
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
    else:
        hint = font_small.render("Arahkan mouse ke sebuah node untuk lihat detail (debug).", True, COLORS["text_dim"])
        screen.blit(hint, (area.x + TREE_MARGIN_X, area.bottom - 18))

    if hover_node[0] is not None:
        _draw_node_tooltip(screen, fonts, hover_node[0], mouse_pos)


# -----------------------------------------------------------------------
# Visualisasi local search: grafik konvergensi (skor rencana yang dipegang vs
# skor terbaik per iterasi) + rencana terbaik per ronde. Menggantikan pohon
# pencarian saat mode NPC adalah Hill Climbing / Simulated Annealing.
# -----------------------------------------------------------------------
LOCAL_CHART_HEIGHT_RATIO = 0.45   # porsi tinggi area untuk grafik
LOCAL_CHART_MIN_HEIGHT = 70
LOCAL_CHART_LEFT_MARGIN = 52      # ruang label sumbu-Y
LOCAL_CHART_MIN_SPAN = 1.0        # cegah pembagian nol saat semua skor sama
LOCAL_CHART_TOP_OFFSET = 24       # ruang legenda di atas grafik
LOCAL_CHART_INSET = 1             # garis tepi grafik 1 px, supaya polyline tidak menimpanya
LOCAL_CHART_LINE_WIDTH = 2
LOCAL_CAPTION_GAP = 2
LOCAL_VIEW_LINE_H = 16
LOCAL_VIEW_SECTION_GAP = 6
LOCAL_VIEW_MESSAGE_OFFSET = 20
LOCAL_LEGEND_SWATCH = 10
LOCAL_LEGEND_SWATCH_TOP = 3
LOCAL_LEGEND_TEXT_GAP = 16
LOCAL_LEGEND_ITEM_GAP = 14
LOCAL_VIEW_BG = (20, 29, 17)


def _chart_points(trace, rect, low, high):
    """Ubah deret skor menjadi titik layar polyline di dalam `rect`.

    Kompleksitas: waktu O(n), memori O(n).
    """
    span = max(high - low, LOCAL_CHART_MIN_SPAN)
    last = max(1, len(trace) - 1)
    usable_w = rect.w - LOCAL_CHART_INSET
    usable_h = rect.h - 2 * LOCAL_CHART_INSET
    return [
        (rect.x + usable_w * index / last, rect.bottom - LOCAL_CHART_INSET - usable_h * (value - low) / span)
        for index, value in enumerate(trace)
    ]


def _plan_lines(agent):
    """Teks rencana terbaik per ronde, mis. 'Putaran +0: Serang'.

    Kompleksitas: waktu O(horizon), memori O(horizon).
    """
    return [f"Putaran +{offset}: {ACTION_LABELS[kind]}" for offset, kind in enumerate(agent.best_plan)]


def _draw_chart_legend(screen, fonts, area):
    legend_x = area.x
    for label, color in (("skor rencana yang dipegang", COLORS["drone"]), ("skor terbaik", COLORS["accent_hover"])):
        pygame.draw.rect(
            screen, color,
            (legend_x, area.y + LOCAL_LEGEND_SWATCH_TOP, LOCAL_LEGEND_SWATCH, LOCAL_LEGEND_SWATCH),
            border_radius=2,
        )
        text = fonts["small"].render(label, True, COLORS["text_dim"])
        screen.blit(text, (legend_x + LOCAL_LEGEND_TEXT_GAP, area.y))
        legend_x += LOCAL_LEGEND_TEXT_GAP + text.get_width() + LOCAL_LEGEND_ITEM_GAP


def _draw_convergence_chart(screen, fonts, agent, area):
    """Gambar grafik konvergensi; kembalikan koordinat Y tepat di bawah grafik."""
    font_small = fonts["small"]
    _draw_chart_legend(screen, fonts, area)

    chart = pygame.Rect(
        area.x + LOCAL_CHART_LEFT_MARGIN, area.y + LOCAL_CHART_TOP_OFFSET,
        area.w - LOCAL_CHART_LEFT_MARGIN, max(LOCAL_CHART_MIN_HEIGHT, int(area.h * LOCAL_CHART_HEIGHT_RATIO)),
    )
    pygame.draw.rect(screen, LOCAL_VIEW_BG, chart, border_radius=4)
    pygame.draw.rect(screen, COLORS["panel_border"], chart, width=1, border_radius=4)

    low = min(min(agent.current_trace), min(agent.best_trace))
    high = max(max(agent.current_trace), max(agent.best_trace))
    screen.blit(font_small.render(f"{high:.0f}", True, COLORS["text_dim"]), (area.x, chart.y))
    screen.blit(font_small.render(f"{low:.0f}", True, COLORS["text_dim"]), (area.x, chart.bottom - LOCAL_VIEW_LINE_H))

    for trace, color in ((agent.current_trace, COLORS["drone"]), (agent.best_trace, COLORS["accent_hover"])):
        points = _chart_points(trace, chart, low, high)
        if len(points) >= 2:
            pygame.draw.lines(screen, color, False, points, LOCAL_CHART_LINE_WIDTH)

    caption = f"iterasi 1 .. {len(agent.best_trace)}"
    screen.blit(font_small.render(caption, True, COLORS["text_dim"]), (chart.x, chart.bottom + LOCAL_CAPTION_GAP))
    return chart.bottom + LOCAL_CAPTION_GAP + LOCAL_VIEW_LINE_H + LOCAL_VIEW_SECTION_GAP


def draw_local_search_view(screen, fonts, agent, area):
    """Gambar hasil keputusan local search terakhir di dalam `area` (pygame.Rect).

    Kompleksitas: waktu O(n + horizon), n = jumlah iterasi tercatat.
    """
    if not agent.best_trace:
        text = fonts["ui"].render(
            "Belum ada rencana local search -- tunggu NPC mengambil keputusan.", True, COLORS["text_dim"],
        )
        screen.blit(text, (area.x + LOCAL_VIEW_MESSAGE_OFFSET, area.y + LOCAL_VIEW_MESSAGE_OFFSET))
        return

    y = _draw_convergence_chart(screen, fonts, agent, area)
    title = f"Rencana terbaik {agent.root_state.npcSpec.name} (aksi per putaran):"
    screen.blit(fonts["ui_bold"].render(title, True, COLORS["text_dim"]), (area.x, y))
    y += LOCAL_VIEW_LINE_H + LOCAL_VIEW_SECTION_GAP

    max_lines = max(0, (area.bottom - y) // LOCAL_VIEW_LINE_H)
    for line in _plan_lines(agent)[:max_lines]:
        screen.blit(fonts["small"].render(line, True, COLORS["text_cream"]), (area.x, y))
        y += LOCAL_VIEW_LINE_H


# Jarak vertikal panel kontrol kanan (mode Battle). Sebelumnya label tiap
# baris tombol digambar cuma 16px di atas barisnya sendiri, padahal jarak
# antar-grup tombol cuma 6px -- akibatnya label baris berikutnya tertindih
# tombol baris sebelumnya ("tulisan menyatu dengan tombol"). Konstanta di
# bawah memberi ruang yang cukup supaya tiap label selalu berada di celah
# kosong, bukan di atas tombol lain.
ROW_LABEL_OFFSET = 16    # jarak label digambar di atas baris tombolnya sendiri
ROW_GROUP_GAP = 16       # jarak ekstra antar-grup tombol, di luar gap internal baris
FIRST_ROW_TOP_PADDING = 60  # jarak baris tombol pertama dari atas area (di bawah judul)


def _make_option_row(x, y, width, items, height=26, gap=6, cols=None):
    """items: list of (label, callback). Bikin satu baris tombol pilihan
    (BUKAN toggle -- masing-masing tombol memilih SATU nilai spesifik,
    disorot lewat .primary saat itu yang aktif). Kalau jumlah item > cols,
    dibungkus ke baris berikutnya. Mengembalikan (list_of_Button, y_berikutnya)."""
    buttons = []
    n = len(items)
    if n == 0:
        return buttons, y
    if cols is None or n <= cols:
        w = (width - gap * (n - 1)) / n
        for i, (label, cb) in enumerate(items):
            buttons.append(Button((x + i * (w + gap), y, w, height), label, cb))
        return buttons, y + height + gap
    rows = math.ceil(n / cols)
    idx = 0
    cur_y = y
    for _ in range(rows):
        row_items = items[idx: idx + cols]
        m = len(row_items)
        w = (width - gap * (m - 1)) / m
        for i, (label, cb) in enumerate(row_items):
            buttons.append(Button((x + i * (w + gap), cur_y, w, height), label, cb))
        idx += cols
        cur_y += height + gap
    return buttons, cur_y


class BattleController:
    """Mesin giliran mode battle:
    input player (lewat tombol) -> apply -> cek terminal ->
    giliran NPC lewat MinimaxAgent (Minimax/Alpha-Beta/Expectimax, dipilih
    lewat tombol) -> apply -> cek terminal -> ulang.

    Semua parameter (algoritma, fungsi evaluasi, urutan aksi, kedalaman,
    kedalaman pohon yang digambar, tampilan pohon/tabel eksperimen) dipilih
    lewat tombol khusus -- tidak ada toggle keyboard."""

    def __init__(self, game):
        self.game = game
        self.enc = Encounter(TUNED_PARAMS, random.Random())
        self.target_id = None
        self.agent = MinimaxAgent()
        self.local_agent = LocalSearchAgent()

        self.depth = BATTLE_DEFAULT_DEPTH
        self.horizon = LOCAL_DEFAULT_HORIZON
        self.mode = MODE_ALPHABETA
        self.eval_key = "seimbang"
        self.order_key = "default"
        self.tree_vis_depth = 2
        self.view = "tree"  # 'tree' | 'experiment'

        self.awaiting_player = True
        self.log = [f"Battle dimulai! {len(self.enc.drones)} drone menyerang. Klik drone untuk memilih target."]
        self.result_text = None
        self.end_deadline = None
        self.last_npc_action = None

        self.experiment_rows = []   # [(label, res_tuple_or_None), ...]
        self.experiment_headers = ADVERSARIAL_EXPERIMENT_HEADERS
        self.experiment_note = ""

        # Papan skor ulangan: tetap ada selama sesi battle ini (tidak direset
        # oleh restart_battle), hanya hilang kalau kembali ke eksplorasi.
        self.round_no = 1
        self.player_wins = 0
        self.npc_wins = 0

        # Dev Mode: ubah HP / kit reparasi kedua pihak secara instan, untuk
        # menguji skenario tertentu (mis. HP kritis, kit reparasi habis)
        # tanpa harus memainkannya sampai ke sana secara manual.
        self.dev_mode = False

        self._build_buttons()

    # ------------------------------------------------------------------
    # SETUP TOMBOL
    # ------------------------------------------------------------------
    def _build_buttons(self):
        area = pygame.Rect(BATTLE_AREA_MARGIN, BOARD_Y, SCREEN_W - 2 * BATTLE_AREA_MARGIN,
                            SCREEN_H - BOARD_Y - BATTLE_AREA_MARGIN)
        self.area = area

        left_w = 300
        self.left_x = area.x + 20
        self.left_w = left_w
        self.right_x = self.left_x + left_w + 20
        self.right_w = area.w - left_w - 60

        # ---- Tombol aksi player (kiri) ----
        act_y = area.y + BATTLE_ACTION_TOP
        act_h = BATTLE_ACTION_H
        self.btn_attack = Button((self.left_x, act_y, self.left_w, act_h),
                                  "[Serang]", lambda: self.on_player_action(ACT_ATTACK))
        self.btn_defend = Button((self.left_x, act_y + act_h + 8, self.left_w, act_h),
                                  "[Bertahan]", lambda: self.on_player_action(ACT_DEFEND))
        self.btn_repair = Button((self.left_x, act_y + 2 * (act_h + 8), self.left_w, act_h),
                                  "[Reparasi]", lambda: self.on_player_action(ACT_REPAIR))
        self.player_buttons = [self.btn_attack, self.btn_defend, self.btn_repair]
        # Tombol konfirmasi kembali ke mode eksplorasi. Posisinya sama dengan
        # tombol [Serang] karena saat battle selesai, tombol aksi disembunyikan
        # dan digantikan tombol ini. enabled=False supaya tidak bisa diklik
        # sebelum pertempuran berakhir.
        self.btn_return = Button(
            (self.left_x, act_y + act_h + 8, self.left_w, act_h),
            "Kembali ke Mode Eksplorasi",
            self.on_return_to_explore,
            enabled=False,
        )
        # Tombol ulang setelah battle selesai (slot paling atas, menonjol).
        self.btn_rematch = Button(
            (self.left_x, act_y, self.left_w, act_h),
            "Ulangi Pertarungan",
            self.restart_battle,
            primary=True,
            enabled=False,
        )
        # Tombol kecil yang selalu tersedia, juga di tengah battle.
        self.btn_restart_small = Button(
            (self.left_x + self.left_w - 110, self.btn_repair.rect.bottom + 14, 110, 22),
            "Mulai Ulang",
            self.restart_battle,
        )

        # ---- Dev Mode (kiri-bawah): ubah HP & kit reparasi secara instan ----
        # Diberi jangkar dari area.bottom (bukan dari bawah log) supaya
        # posisinya tetap stabil walau jumlah baris log berubah-ubah.
        dev_h = 176
        dev_y = area.bottom - dev_h
        self.dev_y = dev_y
        self.btn_dev_toggle = Button((self.left_x, dev_y, self.left_w, 24),
                                      "Dev Mode: NONAKTIF", self.toggle_dev_mode)
        row_y = dev_y + 24 + 20  # +20 utk baris label "MODE DEV -- ..."
        btn_w = (self.left_w - 3 * 6) // 4

        def hp_row(y, owner):
            btns = []
            for i, delta in enumerate((-10, -1, 1, 10)):
                sign = "+" if delta > 0 else ""
                bx = self.left_x + i * (btn_w + 6)
                btns.append(Button((bx, y, btn_w, 22), f"{sign}{delta} HP",
                                    lambda d=delta, o=owner: self.dev_adjust_hp(o, d)))
            return btns

        def kit_row(y, owner):
            half_w = (self.left_w - 6) // 2
            return [
                Button((self.left_x, y, half_w, 22), "-1 Kit", lambda o=owner: self.dev_adjust_kit(o, -1)),
                Button((self.left_x + half_w + 6, y, half_w, 22), "+1 Kit", lambda o=owner: self.dev_adjust_kit(o, 1)),
            ]

        self.dev_btns_tank_hp = hp_row(row_y, PLAYER)
        row_y += 22 + 6
        self.dev_btns_tank_kit = kit_row(row_y, PLAYER)
        row_y += 22 + 14
        self.dev_btns_drone_hp = hp_row(row_y, NPC)
        row_y += 22 + 6
        self.dev_btns_drone_kit = kit_row(row_y, NPC)

        self.dev_adjust_buttons = (
            self.dev_btns_tank_hp + self.dev_btns_tank_kit +
            self.dev_btns_drone_hp + self.dev_btns_drone_kit
        )

        # ---- Panel kontrol kanan: algoritma / eval / urutan / depth / view ----
        # Tiap baris tombol diberi ROW_GROUP_GAP ekstra setelahnya supaya
        # label baris berikutnya (digambar ROW_LABEL_OFFSET di atas tombol,
        # lihat draw_combined) selalu jatuh di ruang kosong, bukan menindih
        # tombol baris sebelumnya.
        x, w = self.right_x, self.right_w
        y = area.y + FIRST_ROW_TOP_PADDING

        self.row_mode, y = _make_option_row(x, y, w, [
            ("Hill Climbing", lambda: self.set_mode(MODE_HILL)),
            ("Simulated Annealing", lambda: self.set_mode(MODE_ANNEAL)),
            ("Minimax", lambda: self.set_mode(MODE_MINIMAX)),
            ("Alpha-Beta", lambda: self.set_mode(MODE_ALPHABETA)),
            ("Expectimax", lambda: self.set_mode(MODE_EXPECTIMAX)),
        ], cols=3)
        y += ROW_GROUP_GAP

        self.row_eval, y = _make_option_row(x, y, w, [
            (label, (lambda k=key: self.set_eval(k))) for key, (_, label) in EVAL_FUNCTIONS.items()
        ], cols=4)
        y += ROW_GROUP_GAP

        self.row_order, y = _make_option_row(x, y, w, [
            ("Serang dulu", lambda: self.set_order("default")),
            ("Bertahan dulu", lambda: self.set_order("bertahan_dulu")),
            ("Reparasi dulu", lambda: self.set_order("reparasi_dulu")),
        ])
        y += ROW_GROUP_GAP

        self.row_depth, y = _make_option_row(x, y, w, [
            (str(d), (lambda dd=d: self.set_depth(dd))) for d in range(1, 7)
        ], height=24, cols=6)
        y += ROW_GROUP_GAP

        self.row_tree_depth, y = _make_option_row(x, y, w, [
            (str(d), (lambda dd=d: self.set_tree_depth(dd))) for d in range(1, 7)
        ], height=24, cols=6)
        y += ROW_GROUP_GAP

        self.row_view, y = _make_option_row(x, y, w, [
            ("Pohon Pencarian", lambda: self.set_view("tree")),
            ("Tabel Eksperimen", lambda: self.set_view("experiment")),
        ])
        y += ROW_GROUP_GAP

        self.btn_experiment = Button((x, y, w, 30), "Jalankan Eksperimen", self.on_run_experiment, primary=True)
        y += 30 + 10

        self.debug_y = y
        self.debug_h = 92
        self.canvas_y = self.debug_y + self.debug_h + 10  # area pohon / tabel mulai di sini
        self._sync_view_labels()

    def all_buttons(self):
        return (
            self.player_buttons + [self.btn_rematch, self.btn_return, self.btn_restart_small] +
            self.row_mode + self.row_eval + self.row_order +
            self.row_depth + self.row_tree_depth + self.row_view + [self.btn_experiment] +
            [self.btn_dev_toggle] + self.dev_adjust_buttons
        )

    # ------------------------------------------------------------------
    # SETTER (dipanggil tombol -- memilih SATU nilai, bukan toggle)
    # ------------------------------------------------------------------
    def is_local_mode(self):
        """True kalau algoritma NPC yang dipilih adalah local search."""
        return self.mode in LOCAL_MODES

    def search_depth(self):
        """Nilai yang diatur tombol 1-6: horizon rencana (local search) atau MAX_DEPTH (adversarial)."""
        return self.horizon if self.is_local_mode() else self.depth

    def _sync_view_labels(self):
        """Label tombol tampilan pertama mengikuti algoritma: grafik konvergensi atau pohon."""
        self.row_view[0].label = "Konvergensi" if self.is_local_mode() else "Pohon Pencarian"

    def set_mode(self, mode):
        self.mode = mode
        self._sync_view_labels()

    def set_eval(self, key):
        self.eval_key = key

    def set_order(self, key):
        self.order_key = key

    def set_depth(self, d):
        if self.is_local_mode():
            self.horizon = d
            return
        self.depth = d
        self.tree_vis_depth = min(self.tree_vis_depth, d)

    def set_tree_depth(self, d):
        self.tree_vis_depth = min(d, self.depth)

    def set_view(self, v):
        self.view = v

    # ------------------------------------------------------------------
    # DEV MODE -- ubah HP & kit reparasi kedua pihak secara instan (testing).
    # Tidak lewat apply_action(); mengubah state.playerHP/npcHP dkk langsung.
    # ------------------------------------------------------------------
    def toggle_dev_mode(self):
        self.dev_mode = not self.dev_mode

    def dev_adjust_hp(self, owner, delta):
        if not self.dev_mode or self.result_text is not None:
            return
        if owner == PLAYER:
            self.enc.tank_hp = max(0, min(self.enc.tank_max_hp, self.enc.tank_hp + delta))
            self._log(f"[Dev] Tank HP -> {int(self.enc.tank_hp)}")
        else:
            target = self.selected_target()
            target.hp = max(0, min(target.loadout.max_hp, target.hp + delta))
            self._log(f"[Dev] Drone #{target.drone_id} HP -> {int(target.hp)}")
        if self.enc.is_over():
            self._finish()

    def dev_adjust_kit(self, owner, delta):
        if not self.dev_mode or self.result_text is not None:
            return
        if owner == PLAYER:
            self.enc.tank_repair_kits = max(0, min(9, self.enc.tank_repair_kits + delta))
            self._log(f"[Dev] Kit Tank -> {self.enc.tank_repair_kits}")
        else:
            target = self.selected_target()
            target.repair_kits = max(0, min(9, target.repair_kits + delta))
            self._log(f"[Dev] Kit Drone #{target.drone_id} -> {target.repair_kits}")

    # ------------------------------------------------------------------
    # ALUR BATTLE
    # ------------------------------------------------------------------
    def _log(self, text):
        self.log.append(text)
        if len(self.log) > BATTLE_LOG_MAX:
            self.log.pop(0)

    def selected_target(self):
        """Drone yang sedang ditargetkan; otomatis pindah ke drone hidup pertama bila sudah hancur."""
        alive = self.enc.alive_drones()
        matches = [d for d in alive if d.drone_id == self.target_id]
        return matches[0] if matches else alive[0]

    def focus_state(self):
        """Duel 1-lawan-1 tank vs target saat ini (untuk eksperimen & debug)."""
        return self.enc.projection(self.selected_target(), NPC)

    def on_player_action(self, action):
        if self.result_text is not None or not self.awaiting_player:
            return
        if action == ACT_REPAIR and self.enc.tank_repair_kits <= 0:
            self._log("Kit reparasi habis!")
            return
        self.player_act(action)

    def _run_selected_agent(self, state):
        """Minta aksi ke agen yang sesuai dengan algoritma NPC terpilih."""
        eval_fn = EVAL_FUNCTIONS[self.eval_key][0]
        if self.is_local_mode():
            return self.local_agent.choose_action(state, horizon=self.horizon, mode=self.mode, eval_fn=eval_fn)
        return self.agent.choose_action(
            state, depth=self.depth, mode=self.mode, eval_fn=eval_fn,
            action_order=ACTION_ORDERS[self.order_key][0],
        )

    def _choose_drone_action(self, state):
        """Chooser untuk Encounter: jalankan agen terpilih & catat untuk panel debug."""
        action = self._run_selected_agent(state)
        self.last_npc_action = action
        return action

    def player_act(self, action):
        target_id = self.selected_target().drone_id
        self.awaiting_player = False
        for line in self.enc.play_round(action, target_id, self._choose_drone_action):
            self._log(line)
        if self.enc.is_over():
            self._finish()
            return
        self.awaiting_player = True

    def _finish(self):
        """Battle berakhir. Hanya menyimpan hasil dan menampilkan tombol
        konfirmasi. TIDAK lagi memanggil end_battle() otomatis; pemain
        sendiri yang memutuskan kapan kembali ke mode eksplorasi."""
        self.awaiting_player = False
        if self.enc.winner() == NPC:
            self.result_text = "DRONE MENANG -- tank hancur!"
            self.npc_wins += 1
        else:
            self.result_text = "PLAYER MENANG -- semua drone dilumpuhkan!"
            self.player_wins += 1
        self._log(self.result_text)

    def update(self, now_ms):
        """Tidak ada lagi timer otomatis di mode battle. Method ini
        dipertahankan (kosong) karena Game.update() masih memanggilnya."""
        pass

    def restart_battle(self):
        """Ulangi pertarungan dari awal (HP & kit reparasi penuh, giliran
        player) TANPA keluar dari mode battle. Pengaturan algoritma, fungsi
        evaluasi, urutan aksi, kedalaman, dan tampilan tetap dipertahankan.
        Peta, posisi tank, dan drone di mode eksplorasi tidak disentuh."""
        self.round_no += 1
        self.enc = Encounter(TUNED_PARAMS, random.Random())
        self.target_id = None
        self.agent = MinimaxAgent()   # bersihkan skor debug & pohon lama
        self.local_agent = LocalSearchAgent()   # bersihkan grafik & rencana lama
        self.awaiting_player = True
        self.result_text = None
        self.end_deadline = None
        self.last_npc_action = None
        self.experiment_rows = []
        self.experiment_note = ""
        self.log = [f"Ronde {self.round_no} dimulai! Pilih aksi lewat tombol di bawah."]

    def on_return_to_explore(self):
        """Dipanggil saat tombol konfirmasi diklik. Baru di sini Game
        diminta kembali ke mode eksplorasi."""
        if self.result_text is not None:
            self.game.end_battle()

    def roster_row_rect(self, index):
        return pygame.Rect(self.left_x, self.area.y + ROSTER_TOP + index * ROSTER_ROW_H,
                           self.left_w, ROSTER_ROW_H - 2)

    def handle_click(self, pos):
        for btn in self.all_buttons():
            if btn.handle_click(pos):
                return
        self._select_target_at(pos)

    def _select_target_at(self, pos):
        if self.result_text is not None:
            return
        for index, drone in enumerate(self.enc.alive_drones()):
            if self.roster_row_rect(index).collidepoint(pos):
                self.target_id = drone.drone_id


    def draw(self, screen, fonts):
        BattleHUD.draw_combined(screen, fonts, self)

    # ------------------------------------------------------------------
    # EKSPERIMEN (poin 1-6 laporan): dijalankan HANYA saat tombol ditekan.
    # Semua kombinasi dihitung dari state SAAT INI, tapi "dipaksa" seolah
    # giliran NPC (turnOwner=NPC, status bertahan direset) supaya semua
    # kombinasi dibandingkan dari titik tolak yang sama & adil.
    # ------------------------------------------------------------------
    def on_run_experiment(self):
        """Jalankan eksperimen sesuai keluarga algoritma NPC yang sedang dipilih."""
        if self.is_local_mode():
            self._run_local_experiment()
        else:
            self._run_adversarial_experiment()

    def _run_local_experiment(self):
        """Bandingkan algoritma local search, fungsi evaluasi, dan horizon dari state yang sama.

        Tiap kombinasi memakai RNG ber-seed sama (EXPERIMENT_SEED) supaya selisih hasil berasal
        dari konfigurasinya, bukan dari keberuntungan acak.
        """
        if self.result_text is not None:
            self._log("Eksperimen tidak tersedia: battle sudah selesai.")
            return
        base = self.focus_state()
        base.playerDefending = False
        base.npcDefending = False

        def run_one(mode, eval_key, horizon):
            agent = LocalSearchAgent(random.Random(EXPERIMENT_SEED))
            t0 = time.perf_counter()
            action = agent.choose_action(
                base, horizon=horizon, mode=mode, eval_fn=EVAL_FUNCTIONS[eval_key][0],
            )
            ms = (time.perf_counter() - t0) * 1000.0
            return {
                "action_label": ACTION_LABELS[action], "score": agent.best_score, "ms": ms,
                "extra": (str(agent.evaluations), str(agent.restarts_done), str(agent.accepted_moves)),
            }

        cm, ce, ch = self.mode, self.eval_key, self.horizon
        mlabel = MODE_LABELS[cm]
        rows = [(f"(1) Algoritma -- eval={EVAL_FUNCTIONS[ce][1]}, horizon={ch}", None)]
        for mkey in LOCAL_MODES:
            rows.append((MODE_LABELS[mkey], run_one(mkey, ce, ch)))

        rows.append((f"(2) Fungsi evaluasi -- {mlabel}, horizon={ch}", None))
        for ekey, (_, elabel) in EVAL_FUNCTIONS.items():
            rows.append((elabel, run_one(cm, ekey, ch)))

        rows.append((f"(3) Horizon -- {mlabel}, eval={EVAL_FUNCTIONS[ce][1]}", None))
        for horizon in range(1, 7):
            rows.append((f"horizon = {horizon}", run_one(cm, ce, horizon)))

        self.experiment_rows = rows
        self.experiment_headers = LOCAL_EXPERIMENT_HEADERS
        self.experiment_note = (
            "(1) HC dan SA memakai anggaran evaluasi sama; HC hanya menerima neighbor yang lebih baik "
            "(bisa terjebak di optimum lokal, dibantu restart), SA kadang menerima yang lebih buruk "
            "saat suhu tinggi. Hasilnya acak (seed tetap di eksperimen ini), bukan jaminan optimal. "
            "(2) Fungsi evaluasi mengubah aksi pilihan NPC -- 'kepribadian' NPC. "
            "(3) Horizon makin panjang = ruang rencana membesar (satu slot per putaran) sementara anggaran "
            "evaluasi tetap, jadi peluang menemukan rencana terbaik bisa turun walau waktu per evaluasi naik."
        )
        self.view = "experiment"

    def _run_adversarial_experiment(self):
        base = self.focus_state()
        base.playerDefending = False
        base.npcDefending = False

        def run_one(mode, eval_key, order_key, depth):
            agent = MinimaxAgent()
            t0 = time.perf_counter()
            action = agent.choose_action(
                base, depth=depth, mode=mode,
                eval_fn=EVAL_FUNCTIONS[eval_key][0],
                action_order=ACTION_ORDERS[order_key][0],
                build_tree=False,
            )
            ms = (time.perf_counter() - t0) * 1000.0
            best_score = max((s for _, s in agent.root_action_scores), default=0.0)
            return {
                "action_label": ACTION_LABELS[action], "score": best_score, "ms": ms,
                "extra": (
                    str(agent.nodes_expanded), f"{agent.terminal_count}/{agent.cutoff_count}",
                    str(agent.pruned_count),
                ),
            }

        cm, ce, co, cd = self.mode, self.eval_key, self.order_key, self.depth
        rows = []

        rows.append((f"(1) Algoritma pencarian -- eval={EVAL_FUNCTIONS[ce][1]}, depth={cd}", None))
        for mkey, mlabel in ((MODE_MINIMAX, "Minimax (tanpa pruning)"),
                             (MODE_ALPHABETA, "Alpha-Beta"),
                             (MODE_EXPECTIMAX, "Expectimax")):
            rows.append((mlabel, run_one(mkey, ce, co, cd)))

        # Blok (2)-(4) memakai algoritma yang SEDANG KAMU PILIH (cm) sebagai
        # baseline, bukan selalu dipaksa Alpha-Beta -- supaya kalau kamu ganti
        # ke Minimax/Expectimax di panel kiri, seluruh tabel eksperimen (selain
        # blok 1, yang memang membandingkan algoritma) ikut memakainya.
        mlabel = MODE_LABELS[cm]

        rows.append((f"(2) Fungsi evaluasi -- {mlabel}, depth={cd}", None))
        for ekey, (_, elabel) in EVAL_FUNCTIONS.items():
            rows.append((elabel, run_one(cm, ekey, co, cd)))

        rows.append((f"(3) Urutan aksi -- {mlabel}, eval={EVAL_FUNCTIONS[ce][1]}, depth={cd}", None))
        for okey, (_, olabel) in ACTION_ORDERS.items():
            rows.append((olabel, run_one(cm, ce, okey, cd)))

        rows.append((f"(4) Kedalaman -- {mlabel}, eval={EVAL_FUNCTIONS[ce][1]}", None))
        for d in range(1, 7):
            rows.append((f"depth = {d}", run_one(cm, ce, co, d)))

        self.experiment_rows = rows
        self.experiment_headers = ADVERSARIAL_EXPERIMENT_HEADERS
        pruning_note = (
            "Dipangkas hanya akan >0 kalau algoritmanya Alpha-Beta -- Minimax murni & Expectimax "
            "tidak pernah memangkas cabang (nilainya akan selalu 0 di blok 2-4)."
            if cm != MODE_ALPHABETA else
            "Urutan yang menaruh aksi terbaik lebih dulu memangkas lebih banyak (move ordering)."
        )
        self.experiment_note = (
            "(1) Alpha-Beta selalu mengekspansi node <= Minimax murni (nilai/aksi terpilih sama; "
            "Expectimax beda karena giliran player dianggap ACAK, bukan lawan optimal, jadi bisa "
            "memilih aksi lebih 'berani'). "
            f"(2) Fungsi evaluasi diuji memakai algoritma yang sedang aktif ({mlabel}) -- mengubah AKSI "
            "yang dipilih NPC pada HP/kit reparasi sama menunjukkan 'kepribadian' NPC (Agresif condong "
            "Serang, Defensif condong Bertahan/Reparasi). "
            f"(3) Urutan aksi (dgn {mlabel}) TIDAK mengubah aksi/skor akhir karena pencarian tetap "
            f"exhaustive/optimal, tapi bisa mengubah jumlah node dipangkas. {pruning_note} "
            f"(4) Dengan {mlabel}, semakin dalam depth, node & waktu makin naik cepat (branching <=3), "
            "tapi keputusan biasanya menstabil setelah depth tertentu."
        )
        self.view = "experiment"


class BattleHUD:
    """Render seluruh layar mode battle: HP bar, log, tombol aksi player,
    panel kontrol (algoritma/eval/urutan/depth/tampilan -- semua tombol,
    tanpa toggle keyboard), panel debug (skor aksi & node count), DAN
    pohon pencarian / tabel eksperimen (dipilih lewat tombol tampilan)."""

    @staticmethod
    def _hp_bar(screen, x, y, w, h, hp, max_hp, color_fg, font):
        pygame.draw.rect(screen, (20, 29, 17), (x, y, w, h), border_radius=5)
        ratio = max(0.0, min(1.0, hp / max_hp))
        fill_w = int((w - 4) * ratio)
        if fill_w > 0:
            pygame.draw.rect(screen, color_fg, (x + 2, y + 2, fill_w, h - 4), border_radius=4)
        pygame.draw.rect(screen, COLORS["panel_border"], (x, y, w, h), width=1, border_radius=5)
        txt = font.render(f"{max(0, int(hp))} / {int(max_hp)}", True, COLORS["text_cream"])
        screen.blit(txt, txt.get_rect(center=(x + w // 2, y + h // 2)))

    @staticmethod
    def _draw_row_label(screen, font, text, x, y):
        t = font.render(text, True, COLORS["text_dim"])
        screen.blit(t, (x, y))

    @staticmethod
    def _row_label_y(row):
        """Posisi Y label sebuah baris tombol -- ROW_LABEL_OFFSET di atas
        tombol pertamanya. `row` harus punya gap sisa dari ROW_GROUP_GAP
        (lihat BattleController._build_buttons) supaya tidak menindih
        baris tombol sebelumnya."""
        return row[0].rect.y - ROW_LABEL_OFFSET

    @staticmethod
    def _draw_button_row(screen, font, buttons, active_predicate):
        for btn in buttons:
            btn.primary = active_predicate(btn)
            btn.draw(screen, font)

    @staticmethod
    def _draw_roster(screen, fonts, controller):
        """Daftar drone hidup: nama loadout, bar HP, status siap/isi daya, bertahan.
        Baris yang dipilih (target serangan tank) diberi garis tepi aksen."""
        font_small = fonts["small"]
        alive = controller.enc.alive_drones()
        selected_id = None if controller.result_text else controller.selected_target().drone_id
        header = font_small.render(f"Drone hidup: {len(alive)} (klik untuk menargetkan)", True, COLORS["text_dim"])
        screen.blit(header, (controller.left_x, controller.area.y + ROSTER_TOP - 16))
        for index, drone in enumerate(alive):
            row = controller.roster_row_rect(index)
            is_selected = drone.drone_id == selected_id
            pygame.draw.rect(screen, (20, 29, 17), row, border_radius=4)
            border = COLORS["accent"] if is_selected else COLORS["panel_border"]
            pygame.draw.rect(screen, border, row, width=2 if is_selected else 1, border_radius=4)
            label = font_small.render(f"#{drone.drone_id} {drone.loadout.name}", True, COLORS["text_cream"])
            screen.blit(label, (row.x + 6, row.y + 2))
            bar = pygame.Rect(row.x + ROSTER_BAR_X, row.y + 5, ROSTER_BAR_W, ROSTER_BAR_H)
            pygame.draw.rect(screen, COLORS["panel_border"], bar)
            fill_w = int(ROSTER_BAR_W * max(0.0, drone.hp) / drone.loadout.max_hp)
            pygame.draw.rect(screen, COLORS["drone"], (bar.x, bar.y, fill_w, bar.h))
            hp_text = font_small.render(f"{int(drone.hp)}", True, COLORS["text_dim"])
            screen.blit(hp_text, (bar.right + 4, row.y + 2))
            tag = ("SIAP" if drone.is_ready() else "ISI") + (" DEF" if drone.defending else "")
            tag_text = font_small.render(tag, True, COLORS["text_dim"])
            screen.blit(tag_text, (row.x + ROSTER_TAG_X, row.y + 2))

    @staticmethod
    def draw_combined(screen, fonts, controller):
        font_ui = fonts["ui"]
        font_ui_bold = fonts["ui_bold"]
        font_small = fonts["small"]
        font_title = fonts["title"]

        area = controller.area
        pygame.draw.rect(screen, COLORS["panel"], area, border_radius=12)
        pygame.draw.rect(screen, COLORS["panel_border"], area, width=1, border_radius=12)

        title = font_title.render("MODE BATTLE -- Local Search (Hill Climbing / SA) + Adversarial Search", True, COLORS["text_cream"])
        screen.blit(title, (area.x + 20, area.y + 12))

        enc = controller.enc
        left_x = controller.left_x
        left_w = controller.left_w
        bars_y = area.y + 44

        lbl_p = font_ui_bold.render("TANK (Player)", True, COLORS["tank"])
        screen.blit(lbl_p, (left_x, bars_y))
        BattleHUD._hp_bar(screen, left_x, bars_y + 20, left_w, 26, enc.tank_hp, enc.tank_max_hp, COLORS["tank"], font_small)
        repair_p = font_small.render(
            f"Reparasi: {enc.tank_repair_kits}  {'| BERTAHAN' if enc.tank_defending else ''}",
            True, COLORS["text_dim"],
        )
        screen.blit(repair_p, (left_x, bars_y + 50))

        BattleHUD._draw_roster(screen, fonts, controller)

        turn_y = area.y + ROSTER_TOP + MAX_ACTIVE_DRONES * ROSTER_ROW_H + 6
        if controller.result_text:
            txt = font_ui_bold.render(controller.result_text, True, COLORS["accent"])
        elif controller.awaiting_player:
            txt = font_ui_bold.render("Giliranmu -- pilih target & aksi:", True, COLORS["text_cream"])
        else:
            txt = font_ui_bold.render("Drone berpikir...", True, COLORS["text_dim"])
        screen.blit(txt, (left_x, turn_y))

        score = font_small.render(
            f"Percobaan {controller.round_no} | Giliran {enc.round_no} | Bantuan: {enc.reinforcement_count} | "
            f"Menang T:{controller.player_wins} D:{controller.npc_wins}",
            True, COLORS["text_dim"],
        )
        screen.blit(score, (left_x, turn_y + 22))

        # ---- Tombol aksi player ATAU tombol konfirmasi ----
        battle_over = controller.result_text is not None

        if battle_over:
            # Pertempuran selesai: tombol aksi disembunyikan (juga dinonaktifkan
            # supaya tidak bisa terklik), diganti tombol konfirmasi.
            for btn in controller.player_buttons:
                btn.enabled = False
            controller.btn_rematch.enabled = True
            controller.btn_return.enabled = True
            controller.btn_restart_small.enabled = False
            controller.btn_rematch.draw(screen, font_ui_bold)
            controller.btn_return.draw(screen, font_ui_bold)
        else:
            controller.btn_rematch.enabled = False
            controller.btn_return.enabled = False
            controller.btn_restart_small.enabled = False
            can_act = controller.awaiting_player
            controller.btn_attack.enabled = can_act
            controller.btn_defend.enabled = can_act
            controller.btn_repair.enabled = can_act and enc.tank_repair_kits > 0
            target = controller.selected_target()
            controller.btn_attack.label = f"[Serang #{target.drone_id} {target.loadout.name}]"
            for btn in controller.player_buttons:
                btn.primary = False
                btn.draw(screen, font_ui_bold)

        if not battle_over:
            controller.btn_restart_small.enabled = True
            controller.btn_restart_small.draw(screen, font_small)

        log_y = controller.btn_repair.rect.bottom + 16
        log_hdr = font_ui_bold.render("Log Pertarungan", True, COLORS["text_dim"])
        screen.blit(log_hdr, (left_x, log_y))
        log_y += 20
        # Log dibatasi supaya tidak menabrak panel Dev Mode di bawahnya
        # (posisi dev panel dijangkar dari area.bottom, lihat _build_buttons).
        max_log_y = controller.dev_y - 18
        for line in controller.log[-BATTLE_LOG_MAX:]:
            if log_y > max_log_y:
                break
            t = font_small.render(line, True, COLORS["text_cream"])
            screen.blit(t, (left_x, log_y))
            log_y += 16

        # ---- Dev Mode: ubah HP & kit reparasi secara instan ----
        battle_active = controller.result_text is None
        controller.btn_dev_toggle.primary = controller.dev_mode
        controller.btn_dev_toggle.label = "Dev Mode: AKTIF" if controller.dev_mode else "Dev Mode: NONAKTIF"
        controller.btn_dev_toggle.draw(screen, font_ui_bold)

        for btn in controller.dev_adjust_buttons:
            btn.enabled = controller.dev_mode and battle_active
        if controller.dev_mode:
            hint = font_small.render("Tank:", True, COLORS["text_dim"])
            screen.blit(hint, (left_x, controller.dev_btns_tank_hp[0].rect.y - 14))
            hint2 = font_small.render("Drone:", True, COLORS["text_dim"])
            screen.blit(hint2, (left_x, controller.dev_btns_drone_hp[0].rect.y - 14))
        for btn in controller.dev_adjust_buttons:
            btn.draw(screen, font_small)

        # ==================================================================
        # KOLOM KANAN -- panel kontrol (semua tombol) + debug + pohon/tabel
        # ==================================================================
        rx, rw = controller.right_x, controller.right_w

        BattleHUD._draw_row_label(screen, font_small, "Algoritma NPC:", rx, BattleHUD._row_label_y(controller.row_mode))
        BattleHUD._draw_button_row(screen, font_ui, controller.row_mode, lambda b: _btn_matches_mode(b, controller.mode))

        BattleHUD._draw_row_label(screen, font_small, "Fungsi Evaluasi (kepribadian NPC):", rx, BattleHUD._row_label_y(controller.row_eval))
        BattleHUD._draw_button_row(screen, font_ui, controller.row_eval, lambda b: b.label == EVAL_FUNCTIONS[controller.eval_key][1])

        BattleHUD._draw_row_label(screen, font_small, "Urutan Aksi (move ordering):", rx, BattleHUD._row_label_y(controller.row_order))
        for btn in controller.row_order:
            btn.enabled = not controller.is_local_mode()  # move ordering hanya relevan untuk alpha-beta
        order_label = {"default": "Serang dulu", "bertahan_dulu": "Bertahan dulu", "reparasi_dulu": "Reparasi dulu"}[controller.order_key]
        BattleHUD._draw_button_row(screen, font_ui, controller.row_order, lambda b: b.label == order_label)

        depth_title = "Horizon Rencana (putaran):" if controller.is_local_mode() else "Kedalaman Pencarian (MAX_DEPTH):"
        BattleHUD._draw_row_label(screen, font_small, depth_title, rx, BattleHUD._row_label_y(controller.row_depth))
        BattleHUD._draw_button_row(screen, font_small, controller.row_depth, lambda b: b.label == str(controller.search_depth()))

        BattleHUD._draw_row_label(screen, font_small, "Kedalaman Pohon Digambar:", rx, BattleHUD._row_label_y(controller.row_tree_depth))
        for btn in controller.row_tree_depth:
            btn.enabled = not controller.is_local_mode() and int(btn.label) <= controller.depth
            btn.primary = btn.label == str(controller.tree_vis_depth)
            btn.draw(screen, font_small)

        BattleHUD._draw_row_label(screen, font_small, "Tampilan:", rx, BattleHUD._row_label_y(controller.row_view))
        BattleHUD._draw_button_row(screen, font_ui, controller.row_view,
                                    lambda b: (b is controller.row_view[0]) == (controller.view == "tree"))

        controller.btn_experiment.draw(screen, font_ui_bold)

        BattleHUD._draw_debug_panel(screen, fonts, controller)

        # ---- Area besar: grafik local search / pohon pencarian / tabel eksperimen ----
        canvas_area = pygame.Rect(rx, controller.canvas_y, rw, area.bottom - 16 - controller.canvas_y)
        if controller.view == "experiment":
            BattleHUD._draw_experiment_table(screen, fonts, controller, canvas_area)
        elif controller.is_local_mode():
            draw_local_search_view(screen, fonts, controller.local_agent, canvas_area)
        else:
            BattleHUD._draw_tree_view(screen, fonts, controller, canvas_area)

    @staticmethod
    def _debug_content(controller):
        """(judul, agen, teks hitungan) untuk panel debug sesuai keluarga algoritma terpilih."""
        eval_label = EVAL_FUNCTIONS[controller.eval_key][1]
        if controller.is_local_mode():
            agent = controller.local_agent
            header = f"[Debug] {controller.mode}  eval={eval_label}  horizon={controller.horizon}"
            counts = (
                f"Evaluasi: {agent.evaluations}   Restart: {agent.restarts_done}   "
                f"Diterima: {agent.accepted_moves}   Skor terbaik: {agent.best_score:.1f}"
            )
            return header, agent, counts
        agent = controller.agent
        header = f"[Debug] {controller.mode}  eval={eval_label}  depth={controller.depth}"
        counts = (
            f"Node diekspansi: {agent.nodes_expanded}   Terminal: {agent.terminal_count}   "
            f"Cutoff: {agent.cutoff_count}   Dipangkas: {agent.pruned_count}"
        )
        return header, agent, counts

    @staticmethod
    def _draw_debug_panel(screen, fonts, controller):
        """Panel debug: skor aksi root + hitungan pencarian (berbeda untuk local search / adversarial)."""
        font_small, font_ui_bold = fonts["small"], fonts["ui_bold"]
        rx, rw = controller.right_x, controller.right_w
        header, agent, counts = BattleHUD._debug_content(controller)

        dbg_y = controller.debug_y
        dbg_box = pygame.Rect(rx, dbg_y, rw, controller.debug_h)
        pygame.draw.rect(screen, (20, 29, 17), dbg_box, border_radius=6)
        pygame.draw.rect(screen, COLORS["panel_border"], dbg_box, width=1, border_radius=6)
        ty = dbg_y + 6
        screen.blit(font_ui_bold.render(header, True, COLORS["text_dim"]), (rx + 8, ty))
        ty += 18
        if not agent.root_action_scores:
            screen.blit(font_small.render("(belum ada keputusan NPC)", True, COLORS["text_dim"]), (rx + 8, ty))
            return
        score_txt = "  |  ".join(
            f"{ACTION_LABELS[a]}: {s:.1f}{' <-' if a == controller.last_npc_action else ''}"
            for a, s in agent.root_action_scores
        )
        for line in wrap_text(score_txt, font_small, rw - 16):
            screen.blit(font_small.render(line, True, COLORS["text_cream"]), (rx + 8, ty))
            ty += 15
        screen.blit(font_small.render(counts, True, COLORS["text_dim"]), (rx + 8, ty))

    @staticmethod
    def _draw_tree_view(screen, fonts, controller, canvas_area):
        """Legenda + pohon pencarian adversarial (Minimax / Alpha-Beta / Expectimax)."""
        font_small = fonts["small"]
        legend_x = canvas_area.x
        legend_y = canvas_area.y
        for label, color in (("MAX (NPC)", COLORS["drone"]), ("MIN (Player)", COLORS["tank"]),
                              ("CHANCE (Expectimax)", (150, 110, 200)), ("dipangkas", (110, 70, 62))):
            pygame.draw.rect(screen, color, (legend_x, legend_y + 3, 10, 10), border_radius=2)
            t = font_small.render(label, True, COLORS["text_dim"])
            screen.blit(t, (legend_x + 16, legend_y))
            legend_x += 16 + t.get_width() + 14
        tree_area = pygame.Rect(canvas_area.x, canvas_area.y + 22, canvas_area.w, canvas_area.h - 22)
        draw_search_tree(screen, fonts, controller.agent.last_tree, controller.tree_vis_depth,
                          tree_area, pygame.mouse.get_pos())

    @staticmethod
    def _draw_experiment_table(screen, fonts, controller, area):
        font_small = fonts["small"]
        font_ui_bold = fonts["ui_bold"]
        if not controller.experiment_rows:
            t = font_ui_bold.render(
                'Belum ada hasil -- tekan tombol "Jalankan Eksperimen" di atas.',
                True, COLORS["text_dim"],
            )
            screen.blit(t, (area.x, area.y))
            return

        col_x = [area.x, area.x + int(area.w * 0.34), area.x + int(area.w * 0.46),
                 area.x + int(area.w * 0.58), area.x + int(area.w * 0.68), area.x + int(area.w * 0.78),
                 area.x + int(area.w * 0.88)]
        headers = controller.experiment_headers
        y = area.y
        for i, h in enumerate(headers):
            t = font_small.render(h, True, COLORS["text_dim"])
            screen.blit(t, (col_x[i], y))
        y += 18
        pygame.draw.line(screen, COLORS["panel_border"], (area.x, y), (area.right, y), 1)
        y += 6

        max_rows_y = area.bottom - 60
        for label, res in controller.experiment_rows:
            if y > max_rows_y:
                more = font_small.render("... (lebih banyak baris, perkecil daftar dgn eksperimen ulang)", True, COLORS["text_dim"])
                screen.blit(more, (area.x, y))
                break
            if res is None:
                t = font_ui_bold.render(label, True, COLORS["accent_hover"])
                screen.blit(t, (area.x, y))
                y += 20
                continue
            vals = [
                label, res["action_label"], f'{res["score"]:.1f}',
                *res["extra"], f'{res["ms"]:.2f}',
            ]
            for i, v in enumerate(vals):
                t = font_small.render(v, True, COLORS["text_cream"])
                screen.blit(t, (col_x[i], y))
            y += 15

        y += 8
        for line in wrap_text(controller.experiment_note, fonts["small"], area.w):
            t = font_small.render(line, True, COLORS["text_dim"])
            screen.blit(t, (area.x, y))
            y += 14


def _btn_matches_mode(btn, mode):
    return {
        "Hill Climbing": MODE_HILL, "Simulated Annealing": MODE_ANNEAL,
        "Minimax": MODE_MINIMAX, "Alpha-Beta": MODE_ALPHABETA, "Expectimax": MODE_EXPECTIMAX,
    }.get(btn.label) == mode


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
        self.show_map_debug = False
        self.grid, self.map_stats = generate_map(self.drone, self.tank)
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
        y += 40

        self.btn_map_debug = Button((x, y, w, 28), "", self.toggle_map_debug)
        y += 38

        self.stats_panel_y = y
        y += 100

        self.btn_experiment = Button((x, y, w, 28), "Jalankan Eksperimen", self.on_experiment)
        self.experiment_y = y + 36

        self.panel_x = x
        self.panel_w = w

    def on_resize(self, win_w, win_h):
        """Dipanggil setiap jendela berubah ukuran. Menghitung ulang tata
        letak, lalu membangun ulang tombol supaya posisinya mengikuti."""
        apply_layout(win_w, win_h)

        # Surface layar sudah diganti oleh pygame saat jendela di-resize,
        # jadi ambil referensi terbaru.
        self.screen = pygame.display.get_surface()

        # Tombol panel eksplorasi dibuat dari PANEL_X / PANEL_W, jadi
        # harus dibangun ulang. State game (auto_chase, algoritma, dll.)
        # disimpan di atribut Game, bukan di tombol, sehingga tidak hilang.
        self._build_ui()

        # Kalau sedang di mode Battle, tombolnya juga disusun ulang.
        if self.battle_controller is not None:
            self.battle_controller._build_buttons()

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
        self.grid, self.map_stats = generate_map(self.drone, self.tank)
        self.last_path = []
        self.game_over = False
        self.reset_drone_knowledge()
        self.start_intro_sequence()

    def toggle_map_debug(self):
        self.show_map_debug = not self.show_map_debug

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
        self.grid, self.map_stats = generate_map(self.drone, self.tank)
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
            # Mode battle sepenuhnya berbasis tombol klik (lihat handle_click) --
            # tidak ada lagi kontrol keyboard/toggle di mode ini.
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
            self.battle_controller.handle_click(pos)
            return
        buttons = [
            self.btn_algo_ucs, self.btn_algo_astar,
            self.btn_heur_manhattan, self.btn_heur_euclidean, self.btn_heur_chebyshev,
            *self.btn_visions,
            self.btn_auto_chase,
            self.btn_random_map, self.btn_reset_pos,
            self.btn_map_debug,
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

        title = self.font_title.render("Tank vs Drone Simulasi Ukraina", True, COLORS["text_cream"])
        screen.blit(title, (BOARD_X, 18))

        if self.mode == "battle":
            self.battle_controller.draw(screen, self.fonts)
        else:
            self.draw_board()
            if self.show_map_debug:
                self.draw_map_debug()
            else:
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

    def draw_map_debug(self):
        screen = self.screen
        box = pygame.Rect(BOARD_X, LEGEND_Y, BOARD_W, LEGEND_BLOCK_H - 10)
        pygame.draw.rect(screen, (20, 29, 17), box, border_radius=7)
        pygame.draw.rect(screen, COLORS["panel_border"], box, width=1, border_radius=7)
        for i, line in enumerate(format_map_debug_lines(self.map_stats)):
            color = COLORS["accent_hover"] if i == 0 else COLORS["text_cream"]
            text = self.font_small.render(line, True, color)
            screen.blit(text, (BOARD_X + 10, LEGEND_Y + 8 + i * MAP_DEBUG_LINE_H))

    def draw_status(self):
        screen = self.screen
        rect = pygame.Rect(BOARD_X, STATUS_Y, BOARD_W, 34)
        color = COLORS["danger"] if self.status_caught else COLORS["panel_border"]
        pygame.draw.rect(screen, (20, 29, 17), rect, border_radius=7)
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
        pygame.draw.rect(screen, (20, 29, 17), state_box, border_radius=6)
        pygame.draw.rect(screen, COLORS["panel_border"], state_box, width=1, border_radius=6)
        for i, line in enumerate(wrap_text(self.drone_state_text, self.font_small, self.panel_w - 16)[:2]):
            t = self.font_small.render(line, True, COLORS["text_cream"])
            screen.blit(t, (x + 8, state_y + 22 + i * 14))

        self.btn_random_map.draw(screen, self.font_ui)
        self.btn_reset_pos.draw(screen, self.font_ui)

        self.btn_map_debug.label = "Debug Peta (CSP): " + ("ON" if self.show_map_debug else "OFF")
        self.btn_map_debug.primary = self.show_map_debug
        self.btn_map_debug.draw(screen, self.font_ui)

        stats_hdr = self.font_ui_bold.render("Statistik Pencarian Terakhir", True, COLORS["text_dim"])
        screen.blit(stats_hdr, (x, self.stats_panel_y))
        stats_box = pygame.Rect(x, self.stats_panel_y + 18, self.panel_w, 78)
        pygame.draw.rect(screen, (20, 29, 17), stats_box, border_radius=6)
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
    if TUNE_FLAG in sys.argv[1:]:
        run_tuning_cli(sys.argv[1:])
        return

    pygame.init()
    pygame.display.set_caption("Tank vs Drone Simulasi Ukraina")

    # RESIZABLE: jendela boleh ditarik / di-maximize. Ukuran layar
    # sebenarnya ikut berubah, bukan sekadar gambar yang diperbesar.
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.RESIZABLE)
    clock = pygame.time.Clock()

    game = Game(screen)

    running = True
    while running:
        now_ms = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                # Jendela ditarik / di-maximize / fullscreen.
                # Kalau ditarik lebih kecil dari minimum, paksa kembali.
                if event.w < MIN_SCREEN_W or event.h < MIN_SCREEN_H:
                    pygame.display.set_mode(
                        (max(MIN_SCREEN_W, event.w), max(MIN_SCREEN_H, event.h)),
                        pygame.RESIZABLE,
                    )
                # Hitung ulang tata letak untuk kedua mode (eksplorasi & battle).
                w, h = pygame.display.get_surface().get_size()
                game.on_resize(w, h)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()  # memicu VIDEORESIZE juga
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
