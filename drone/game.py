import math
import pygame

from battle_constants import BATTLE_TRIGGER_DISTANCE
from battle_controller import BattleController
from display_config import (
    AUTO_CHASE_TICK_MS,
    BOARD_H,
    BOARD_W,
    BOARD_X,
    BOARD_Y,
    COLORS,
    INTRO_MOVE_TICK_MS,
    LEGEND_ITEMS,
    LEGEND_Y,
    PANEL_W,
    PANEL_X,
    PANEL_Y,
    SCAN_PATH_PAUSE_MS,
    SCAN_REVEAL_PER_TICK,
    SCAN_TICK_MS,
    SCREEN_H,
    STATUS_Y,
    apply_layout,
    lerp_color,
)
from fog_of_war import can_detect_tank, find_nearest_search_index, generate_search_pattern
from map_generation import generate_map, pick_random_start_positions
from pathfinding import SearchResult, search
from terrain import (
    ARTILLERY,
    BRIDGE,
    CAMP,
    CELL,
    COLS,
    DRONE_TERRAIN_COST,
    RIVER,
    ROCK,
    ROWS,
    RUIN,
    TREE,
    is_blocked,
)
from widgets import Button, wrap_text


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
