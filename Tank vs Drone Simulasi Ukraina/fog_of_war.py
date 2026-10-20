import math

from terrain import COLS, ROWS, blocks_sight, is_blocked_for_drone


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
