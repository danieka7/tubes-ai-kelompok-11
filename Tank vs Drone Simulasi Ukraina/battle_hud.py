import pygame

from battle_constants import (
    BATTLE_LOG_MAX,
    MODE_ALPHABETA,
    MODE_ANNEAL,
    MODE_EXPECTIMAX,
    MODE_HILL,
    MODE_MINIMAX,
    ROSTER_BAR_H,
    ROSTER_BAR_W,
    ROSTER_BAR_X,
    ROSTER_ROW_H,
    ROSTER_TAG_X,
    ROSTER_TOP,
)
from battle_eval import EVAL_FUNCTIONS
from battle_layout import ROW_LABEL_OFFSET
from display_config import COLORS
from labels import ACTION_LABELS
from loadouts import MAX_ACTIVE_DRONES
from local_search_view import draw_local_search_view
from tree_view import draw_search_tree
from widgets import wrap_text


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
