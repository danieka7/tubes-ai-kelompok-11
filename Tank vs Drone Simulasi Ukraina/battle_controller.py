import random
import time
import pygame

from battle_constants import (
    ACT_ATTACK,
    ACT_DEFEND,
    ACT_REPAIR,
    BATTLE_ACTION_H,
    BATTLE_ACTION_TOP,
    BATTLE_DEFAULT_DEPTH,
    BATTLE_LOG_MAX,
    LOCAL_MODES,
    MODE_ALPHABETA,
    MODE_ANNEAL,
    MODE_EXPECTIMAX,
    MODE_HILL,
    MODE_MINIMAX,
    NPC,
    PLAYER,
    ROSTER_ROW_H,
    ROSTER_TOP,
)
from battle_eval import ACTION_ORDERS, EVAL_FUNCTIONS
from battle_hud import BattleHUD
from battle_layout import FIRST_ROW_TOP_PADDING, ROW_GROUP_GAP, _make_option_row
from display_config import BATTLE_AREA_MARGIN, BOARD_Y, SCREEN_H, SCREEN_W
from encounter import Encounter
from labels import ACTION_LABELS, MODE_LABELS
from loadouts import TUNED_PARAMS
from local_search_agent import (
    ADVERSARIAL_EXPERIMENT_HEADERS,
    EXPERIMENT_SEED,
    LOCAL_DEFAULT_HORIZON,
    LOCAL_EXPERIMENT_HEADERS,
    LocalSearchAgent,
)
from minimax_agent import MinimaxAgent
from widgets import Button


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
