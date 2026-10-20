import random

from terrain import ARTILLERY, BRIDGE, CAMP, COLS, DIRT, RIVER, ROCK, ROWS, RUIN, TREE, is_blocked


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
