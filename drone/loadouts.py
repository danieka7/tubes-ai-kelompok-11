from dataclasses import dataclass


# =============================================================================
# BAGIAN 5B.2 -- LOADOUT DRONE, GELOMBANG MUSUH & PARAMETER KESEIMBANGAN
# =============================================================================
# Semua angka keseimbangan (HP tank, jumlah drone, statistik tiap loadout,
# aturan bantuan musuh) dikumpulkan di BalanceParams supaya bisa dicari
# otomatis oleh local search (BAGIAN 5B.4), bukan ditebak manual.

MAX_ACTIVE_DRONES = 6           # batas drone hidup bersamaan (muat di daftar HUD)
MAX_REINFORCEMENTS = 6          # batas total drone bantuan per pertempuran
PERCENT_BASE = 100

LOADOUT_BOM = "bom"
LOADOUT_MISIL = "misil"
LOADOUT_KAMIKAZE = "kamikaze"

# Sifat tetap tiap loadout (bukan hasil tuning): yang di-tuning hanya HP & damage.
BOM_ACT_INTERVAL = 2            # lambat: bertindak tiap 2 ronde
BOM_REPAIR_KITS = 1
BOM_PIERCE = 0.0
MISIL_ACT_INTERVAL = 1
MISIL_REPAIR_KITS = 2
MISIL_PIERCE = 0.5              # misil menembus separuh efek bertahan tank
KAMIKAZE_ACT_INTERVAL = 1
KAMIKAZE_REPAIR_KITS = 0
KAMIKAZE_PIERCE = 0.0

# Bobot kemunculan loadout saat drone awal / bantuan dipilih acak.
LOADOUT_SPAWN_WEIGHTS = {LOADOUT_BOM: 3, LOADOUT_MISIL: 4, LOADOUT_KAMIKAZE: 3}


@dataclass(frozen=True)
class Loadout:
    """Statistik satu jenis drone. frozen=True agar aman dipakai sebagai kunci cache."""
    key: str
    name: str
    max_hp: int
    damage: int
    act_interval: int      # bertindak sekali tiap N ronde (beda "turn")
    repair_kits: int
    pierce: float          # 0..1: porsi efek bertahan lawan yang diabaikan
    suicide: bool          # True: hancur setelah menyerang (kamikaze)


@dataclass(frozen=True)
class BalanceParams:
    """Vektor parameter yang dicari oleh local search."""
    tank_hp: int = 160
    initial_drones: int = 3
    bom_hp: int = 70
    bom_damage: int = 26
    misil_hp: int = 50
    misil_damage: int = 17
    kamikaze_hp: int = 24
    kamikaze_damage: int = 38
    reinforce_after: int = 6          # bantuan baru boleh muncul setelah ronde ke-N
    reinforce_chance_pct: int = 25    # peluang (%) bantuan muncul tiap ronde setelahnya

    def __post_init__(self):
        if not 1 <= self.initial_drones <= MAX_ACTIVE_DRONES:
            raise ValueError(f"initial_drones harus 1..{MAX_ACTIVE_DRONES}, dapat {self.initial_drones}")
        if not 0 <= self.reinforce_chance_pct <= PERCENT_BASE:
            raise ValueError(f"reinforce_chance_pct harus 0..{PERCENT_BASE}, dapat {self.reinforce_chance_pct}")
        positive_fields = (self.tank_hp, self.bom_hp, self.bom_damage, self.misil_hp,
                           self.misil_damage, self.kamikaze_hp, self.kamikaze_damage)
        if min(positive_fields) <= 0:
            raise ValueError("HP dan damage harus > 0")


# Hasil local search terakhir (lihat `python <file> --tune`). Dipakai game.
TUNED_PARAMS = BalanceParams(tank_hp=400, initial_drones=3, bom_hp=40, bom_damage=18, misil_hp=35, misil_damage=14, kamikaze_hp=24, kamikaze_damage=47, reinforce_after=9, reinforce_chance_pct=15)


def build_loadouts(params):
    """Bangun tabel loadout dari parameter keseimbangan.

    Args:
        params: BalanceParams.
    Returns:
        dict key loadout -> Loadout.
    Kompleksitas: O(1) waktu & memori.
    """
    return {
        LOADOUT_BOM: Loadout(LOADOUT_BOM, "Bom", params.bom_hp, params.bom_damage,
                             BOM_ACT_INTERVAL, BOM_REPAIR_KITS, BOM_PIERCE, False),
        LOADOUT_MISIL: Loadout(LOADOUT_MISIL, "Misil", params.misil_hp, params.misil_damage,
                               MISIL_ACT_INTERVAL, MISIL_REPAIR_KITS, MISIL_PIERCE, False),
        LOADOUT_KAMIKAZE: Loadout(LOADOUT_KAMIKAZE, "Kamikaze", params.kamikaze_hp, params.kamikaze_damage,
                                  KAMIKAZE_ACT_INTERVAL, KAMIKAZE_REPAIR_KITS, KAMIKAZE_PIERCE, True),
    }


DEFAULT_NPC_LOADOUT = build_loadouts(TUNED_PARAMS)[LOADOUT_MISIL]
