import math
import time

from terrain import COLS, DRONE_TERRAIN_COST, ROWS, is_blocked_for_drone


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
