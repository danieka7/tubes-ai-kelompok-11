import math
import random

from battle_constants import ACT_ATTACK, ACT_DEFEND, ACT_REPAIR, LOCAL_MODES, MODE_HILL, NPC
from battle_eval import battle_eval_default
from battle_state import battle_utility


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
