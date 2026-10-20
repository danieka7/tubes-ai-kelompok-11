from terrain import CELL, COLS, ROWS


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
