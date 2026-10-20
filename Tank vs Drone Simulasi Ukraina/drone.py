# =============================================================================
# BAGIAN 5B.3 -- ENCOUNTER: TANK MELAWAN BANYAK DRONE + BANTUAN MUSUH
# =============================================================================
# Model murni (tanpa pygame) supaya bisa dipakai game DAN simulasi tuning.
# Tiap drone memilih aksi lewat duel 1-lawan-1 (BattleState) melawan tank,
# sehingga seluruh agen minimax/alpha-beta/expectimax lama tetap dipakai apa adanya.

class Drone:
    """Satu drone musuh di medan battle."""
    __slots__ = ("drone_id", "loadout", "hp", "repair_kits", "defending", "charge")

    def __init__(self, drone_id, loadout):
        self.drone_id = drone_id
        self.loadout = loadout
        self.hp = float(loadout.max_hp)
        self.repair_kits = loadout.repair_kits
        self.defending = False
        self.charge = 0

    def is_alive(self):
        return self.hp > 0

    def is_ready(self):
        """True bila drone ini akan bertindak pada fase drone berikutnya."""
        return self.charge + 1 >= self.loadout.act_interval
