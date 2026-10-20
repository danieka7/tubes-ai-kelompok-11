import math

from widgets import Button


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
