import pygame

from display_config import COLORS
from labels import ACTION_LABELS, KIND_LABELS


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
