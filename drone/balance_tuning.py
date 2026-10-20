import random
from dataclasses import dataclass

from battle_constants import (
    ACT_ATTACK,
    ACT_DEFEND,
    ACT_REPAIR,
    BATTLE_DEFAULT_DEPTH,
    MODE_ALPHABETA,
    PLAYER,
    REPAIR_HEAL_AMOUNT,
)
from battle_eval import EVAL_FUNCTIONS
from encounter import Encounter
from loadouts import BalanceParams, TUNED_PARAMS
from minimax_agent import MinimaxAgent


# =============================================================================
# BAGIAN 5B.4 -- LOCAL SEARCH UNTUK KESEIMBANGAN (tidak terlalu mudah/sulit)
# =============================================================================
# Ruang pencarian: vektor BalanceParams. Fungsi objektif (loss) mengukur seberapa
# jauh hasil simulasi dari "pita nyaman": pemain biasa menang cukup sering
# (tidak frustasi) tapi tidak selalu (tidak bosan), pemain mahir jelas lebih
# unggul, pertempuran tidak terlalu singkat/panjang, bantuan musuh sesekali
# muncul, dan tiap loadout benar-benar berkontribusi.

CASUAL_WIN_BAND = (0.35, 0.55)
SKILLED_WIN_BAND = (0.65, 0.85)
ROUNDS_BAND = (10.0, 22.0)
ROUNDS_SCALE = 10.0               # normalisasi pelanggaran ronde ke skala win-rate
REINFORCE_RATE_BAND = (0.25, 0.60)
MIN_DAMAGE_SHARE = 0.20           # tiap loadout minimal menyumbang 20% damage ke tank

WEIGHT_WIN = 4.0
WEIGHT_ROUNDS = 1.0
WEIGHT_REINFORCE = 1.0
WEIGHT_SHARE = 2.0
WEIGHT_TIMEOUT = 4.0

SIM_ROUND_LIMIT = 60              # pengaman simulasi: lewat ini dianggap macet
CASUAL_REPAIR_HP_RATIO = 0.3
CASUAL_REPAIR_PROB = 0.6
CASUAL_DEFEND_PROB = 0.2
SKILLED_DANGER_FACTOR = 1.5       # waspada bila HP <= 1.5x damage masuk yang diperkirakan

LS_BATTLES_PER_POLICY = 60
LS_MAX_FAILURES = 25              # berhenti bila 25 tetangga berturut-turut tidak lebih baik
LS_RESTARTS = 3
LS_SEED = 2026
LS_EVAL_SEED = 7000
VALIDATION_BATTLES = 400
VALIDATION_SEED = 99000
TUNE_FLAG = "--tune"
NEIGHBOR_MAX_STEPS = 2
NEIGHBOR_RETRY_LIMIT = 20

# nama field -> (min, max, langkah)
PARAM_SPACE = {
    "tank_hp": (120, 300, 10),
    "initial_drones": (2, 5, 1),
    "bom_hp": (40, 120, 5),
    "bom_damage": (15, 45, 2),
    "misil_hp": (30, 90, 5),
    "misil_damage": (10, 30, 1),
    "kamikaze_hp": (10, 50, 5),
    "kamikaze_damage": (20, 60, 3),
    "reinforce_after": (3, 15, 1),
    "reinforce_chance_pct": (5, 60, 5),
}


@dataclass(frozen=True)
class BattleOutcome:
    player_won: bool
    timed_out: bool
    rounds: int
    reinforced: bool
    damage_by_loadout: dict


@dataclass(frozen=True)
class BalanceStats:
    casual_win_rate: float
    skilled_win_rate: float
    avg_rounds: float
    reinforce_rate: float
    timeout_rate: float
    min_damage_share: float


class CachedChooser:
    """Pemilih aksi drone via minimax/alpha-beta dengan memo per state.

    Memo membuat ribuan simulasi tuning jauh lebih cepat tanpa mengubah
    keputusan (agen yang sama, state yang sama -> aksi yang sama)."""

    def __init__(self, depth=BATTLE_DEFAULT_DEPTH, mode=MODE_ALPHABETA, eval_key="seimbang"):
        self.depth = depth
        self.mode = mode
        self.eval_fn = EVAL_FUNCTIONS[eval_key][0]
        self.agent = MinimaxAgent()
        self.cache = {}

    def __call__(self, state):
        key = (state.playerHP, state.npcHP, state.playerRepairKits, state.npcRepairKits,
               state.playerDefending, state.npcDefending, state.playerMaxHP, state.npcSpec)
        if key not in self.cache:
            self.cache[key] = self.agent.choose_action(
                state, depth=self.depth, mode=self.mode, eval_fn=self.eval_fn, build_tree=False)
        return self.cache[key]


def casual_policy(enc, rng):
    """Pemain biasa: kadang reparasi saat HP rendah, kadang bertahan, sisanya
    menyerang drone acak (tidak memprioritaskan target).

    Returns: (aksi, target_id). Kompleksitas O(D)."""
    target = rng.choice(enc.alive_drones())
    low_hp = enc.tank_hp < CASUAL_REPAIR_HP_RATIO * enc.tank_max_hp
    if low_hp and enc.tank_repair_kits > 0 and rng.random() < CASUAL_REPAIR_PROB:
        return ACT_REPAIR, target.drone_id
    if rng.random() < CASUAL_DEFEND_PROB:
        return ACT_DEFEND, target.drone_id
    return ACT_ATTACK, target.drone_id


def skilled_policy(enc, rng):
    """Pemain mahir: baca ancaman, reparasi/bertahan tepat waktu, dan
    memprioritaskan kamikaze lalu drone dengan HP terendah.

    `rng` tidak dipakai (deterministik) -- ada demi antarmuka yang sama dengan
    casual_policy. Returns: (aksi, target_id). Kompleksitas O(D)."""
    alive = enc.alive_drones()
    target = min(alive, key=lambda d: (not d.loadout.suicide, d.hp))
    in_danger = enc.tank_hp <= enc.incoming_damage_estimate() * SKILLED_DANGER_FACTOR
    can_heal_fully = enc.tank_hp <= enc.tank_max_hp - REPAIR_HEAL_AMOUNT
    if in_danger and enc.tank_repair_kits > 0 and can_heal_fully:
        return ACT_REPAIR, target.drone_id
    if in_danger:
        return ACT_DEFEND, target.drone_id
    return ACT_ATTACK, target.drone_id


def simulate_battle(params, policy, rng, chooser):
    """Simulasikan satu pertempuran sampai selesai (atau macet).

    Returns: BattleOutcome. Kompleksitas O(R * D * biaya chooser)."""
    enc = Encounter(params, rng)
    while not enc.is_over() and enc.round_no <= SIM_ROUND_LIMIT:
        action, target_id = policy(enc, rng)
        enc.play_round(action, target_id, chooser)
    return BattleOutcome(
        player_won=enc.winner() == PLAYER,
        timed_out=not enc.is_over(),
        rounds=enc.round_no,
        reinforced=enc.reinforcement_count > 0,
        damage_by_loadout=dict(enc.damage_by_loadout),
    )


def _run_policy(params, policy, battles, base_seed, chooser):
    return [
        simulate_battle(params, policy, random.Random(base_seed + i), chooser)
        for i in range(battles)
    ]


def _rate(outcomes, predicate):
    return sum(1 for o in outcomes if predicate(o)) / len(outcomes)


def _min_damage_share(outcomes):
    totals = {}
    for o in outcomes:
        for key, dmg in o.damage_by_loadout.items():
            totals[key] = totals.get(key, 0.0) + dmg
    grand_total = sum(totals.values())
    if grand_total <= 0:
        return 0.0
    return min(totals.values()) / grand_total


def evaluate_params(params, battles, base_seed, chooser):
    """Ukur BalanceStats dari `battles` simulasi per kebijakan pemain.

    Seed tetap (base_seed + i) -> parameter yang sama selalu menghasilkan
    loss yang sama, sehingga local search tidak tertipu noise.
    Kompleksitas O(battles * biaya simulate_battle)."""
    if battles < 1:
        raise ValueError("battles harus >= 1")
    casual = _run_policy(params, casual_policy, battles, base_seed, chooser)
    skilled = _run_policy(params, skilled_policy, battles, base_seed, chooser)
    both = casual + skilled
    return BalanceStats(
        casual_win_rate=_rate(casual, lambda o: o.player_won),
        skilled_win_rate=_rate(skilled, lambda o: o.player_won),
        avg_rounds=sum(o.rounds for o in both) / len(both),
        reinforce_rate=_rate(both, lambda o: o.reinforced),
        timeout_rate=_rate(both, lambda o: o.timed_out),
        min_damage_share=_min_damage_share(both),
    )


def band_violation(value, low, high):
    """Jarak `value` ke luar pita [low, high]; 0 bila di dalam pita."""
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def balance_loss(stats):
    """Fungsi objektif local search: 0 = semua sasaran keseimbangan terpenuhi."""
    win_gap = (band_violation(stats.casual_win_rate, *CASUAL_WIN_BAND) ** 2
               + band_violation(stats.skilled_win_rate, *SKILLED_WIN_BAND) ** 2)
    rounds_gap = (band_violation(stats.avg_rounds, *ROUNDS_BAND) / ROUNDS_SCALE) ** 2
    reinforce_gap = band_violation(stats.reinforce_rate, *REINFORCE_RATE_BAND) ** 2
    share_gap = band_violation(stats.min_damage_share, MIN_DAMAGE_SHARE, 1.0) ** 2
    return (WEIGHT_WIN * win_gap + WEIGHT_ROUNDS * rounds_gap + WEIGHT_REINFORCE * reinforce_gap
            + WEIGHT_SHARE * share_gap + WEIGHT_TIMEOUT * stats.timeout_rate ** 2)


def random_params(rng):
    """Titik awal acak (untuk random restart) pada grid PARAM_SPACE."""
    values = {}
    for name, (low, high, step) in PARAM_SPACE.items():
        values[name] = low + step * rng.randint(0, (high - low) // step)
    return BalanceParams(**values)


def random_neighbor(params, rng):
    """Tetangga: ubah SATU parameter sebesar +-1..NEIGHBOR_MAX_STEPS langkah (terpotong ke batas).

    Returns: BalanceParams berbeda dari `params` (atau sama bila semua percobaan
    terpotong batas -- sangat jarang)."""
    neighbor = params
    attempts = 0
    while neighbor == params and attempts < NEIGHBOR_RETRY_LIMIT:
        attempts += 1
        name = rng.choice(list(PARAM_SPACE))
        low, high, step = PARAM_SPACE[name]
        shift = rng.choice([-1, 1]) * rng.randint(1, NEIGHBOR_MAX_STEPS) * step
        new_value = max(low, min(high, getattr(params, name) + shift))
        neighbor = BalanceParams(**{**params.__dict__, name: new_value})
    return neighbor


def hill_climb(start, rng, evaluate, max_failures=LS_MAX_FAILURES, log=print):
    """Stochastic first-improvement hill climbing.

    Args:
        start: BalanceParams awal.
        evaluate: fungsi(params)->loss (makin kecil makin baik).
        max_failures: berhenti setelah sekian tetangga berturut-turut tidak membaik.
    Returns: (params_terbaik, loss_terbaik).
    Kompleksitas: O(langkah * biaya evaluate); berhenti di optimum lokal."""
    current, current_loss = start, evaluate(start)
    failures = 0
    while failures < max_failures and current_loss > 0:
        candidate = random_neighbor(current, rng)
        candidate_loss = evaluate(candidate)
        if candidate_loss < current_loss:
            current, current_loss, failures = candidate, candidate_loss, 0
            log(f"    loss={current_loss:.4f}  {current}")
        else:
            failures += 1
    return current, current_loss


def run_local_search(restarts=LS_RESTARTS, seed=LS_SEED, log=print):
    """Hill climbing dengan random restart; mengembalikan (params, loss) terbaik.

    Restart pertama mulai dari TUNED_PARAMS, sisanya dari titik acak, agar
    tidak terjebak satu optimum lokal. Kompleksitas O(restarts * hill_climb)."""
    if restarts < 1:
        raise ValueError("restarts harus >= 1")
    rng = random.Random(seed)
    chooser = CachedChooser()

    def evaluate(params):
        return balance_loss(evaluate_params(params, LS_BATTLES_PER_POLICY, LS_EVAL_SEED, chooser))

    starts = [TUNED_PARAMS] + [random_params(rng) for _ in range(restarts - 1)]
    results = []
    for index, start in enumerate(starts, start=1):
        log(f"[Restart {index}/{restarts}] mulai: {start}")
        results.append(hill_climb(start, rng, evaluate, log=log))
    return min(results, key=lambda r: r[1])


def format_stats(stats):
    return (f"menang pemain biasa={stats.casual_win_rate:.0%}  pemain mahir={stats.skilled_win_rate:.0%}  "
            f"rata-rata ronde={stats.avg_rounds:.1f}  bantuan muncul={stats.reinforce_rate:.0%}  "
            f"macet={stats.timeout_rate:.0%}  porsi damage terkecil={stats.min_damage_share:.0%}")


def run_tuning_cli(args):
    """Entry point `python <file> --tune [jumlah_restart]`."""
    flag_index = args.index(TUNE_FLAG)
    has_count = flag_index + 1 < len(args)
    restarts = int(args[flag_index + 1]) if has_count else LS_RESTARTS
    best, loss = run_local_search(restarts)
    stats = evaluate_params(best, VALIDATION_BATTLES, VALIDATION_SEED, CachedChooser())
    print(f"\nParameter terbaik (loss pencarian={loss:.4f}):\n  {best}")
    print(f"Validasi {VALIDATION_BATTLES} pertempuran/kebijakan (seed baru):\n  {format_stats(stats)}")
    print("\nSalin ke TUNED_PARAMS:\n  TUNED_PARAMS = " + repr(best))
