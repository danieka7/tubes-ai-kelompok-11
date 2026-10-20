from battle_constants import (
    ACT_ATTACK,
    ACT_DEFEND,
    ACT_REPAIR,
    BATTLE_START_REPAIR_KITS,
    NPC,
    PLAYER,
)
from battle_state import BattleState
from drone import Drone
from labels import ACTION_LABELS
from loadouts import (
    LOADOUT_SPAWN_WEIGHTS,
    MAX_ACTIVE_DRONES,
    MAX_REINFORCEMENTS,
    PERCENT_BASE,
    build_loadouts,
)


class Encounter:
    """Satu pertempuran: satu tank vs banyak drone dengan loadout berbeda.

    Urutan satu ronde: tank bertindak -> tiap drone yang sudah siap bertindak
    -> cek bantuan musuh. Bantuan hanya mungkin muncul setelah ronde
    `reinforce_after` (supaya pertempuran yang berlarut-larut makin berat).
    """

    def __init__(self, params, rng):
        self.params = params
        self.rng = rng
        self.loadouts = build_loadouts(params)
        self.tank_max_hp = params.tank_hp
        self.tank_hp = float(params.tank_hp)
        self.tank_repair_kits = BATTLE_START_REPAIR_KITS
        self.tank_defending = False
        self.drones = []
        self.next_drone_id = 1
        self.round_no = 1
        self.reinforcement_count = 0
        self.damage_by_loadout = {key: 0.0 for key in self.loadouts}
        for _ in range(params.initial_drones):
            self._spawn_drone()

    # ---- kueri ----------------------------------------------------------
    def alive_drones(self):
        return [d for d in self.drones if d.is_alive()]

    def is_over(self):
        return self.tank_hp <= 0 or not self.alive_drones()

    def winner(self):
        """PLAYER / NPC, atau None bila pertempuran belum selesai."""
        if not self.is_over():
            return None
        return NPC if self.tank_hp <= 0 else PLAYER

    def incoming_damage_estimate(self):
        """Total damage maksimal yang akan masuk pada fase drone berikutnya."""
        return sum(d.loadout.damage for d in self.alive_drones() if d.is_ready())

    def projection(self, drone, turn_owner):
        """Duel 1-lawan-1 antara tank dan `drone` (dipakai agen minimax)."""
        return BattleState(
            playerHP=self.tank_hp, npcHP=drone.hp,
            playerRepairKits=self.tank_repair_kits, npcRepairKits=drone.repair_kits,
            playerDefending=self.tank_defending, npcDefending=drone.defending,
            turnOwner=turn_owner, turnIndex=self.round_no,
            playerMaxHP=self.tank_max_hp, npcSpec=drone.loadout,
        )

    # ---- aksi -----------------------------------------------------------
    def play_round(self, action, target_id, chooser):
        """Mainkan satu ronde penuh.

        Args:
            action: ACT_ATTACK / ACT_DEFEND / ACT_REPAIR untuk tank.
            target_id: drone_id yang diserang (diabaikan untuk non-serang).
            chooser: fungsi(BattleState)->aksi, memilih aksi drone non-kamikaze.
        Returns:
            list teks log kejadian ronde ini.
        Raises:
            ValueError: pertempuran sudah selesai, kit habis, atau target tidak valid.
        Kompleksitas: O(D * biaya chooser), D = jumlah drone hidup.
        """
        lines = self.player_turn(action, target_id)
        if self.is_over():
            return lines
        lines += self.drones_turn(chooser)
        if self.is_over():
            return lines
        return lines + self.end_round()

    def player_turn(self, action, target_id):
        if self.is_over():
            raise ValueError("Pertempuran sudah berakhir.")
        if action == ACT_REPAIR and self.tank_repair_kits <= 0:
            raise ValueError("Kit reparasi tank habis.")
        target = self._find_target(target_id) if action == ACT_ATTACK else self.alive_drones()[0]
        new_state = self.projection(target, PLAYER).apply_action(action)
        self._absorb(target, new_state)
        return [self._player_line(action, target)]

    def drones_turn(self, chooser):
        lines = []
        queue = self.alive_drones()
        index = 0
        # while (bukan for+break): berhenti begitu tank hancur, sisa drone tidak perlu bertindak.
        while index < len(queue) and self.tank_hp > 0:
            lines += self._drone_act(queue[index], chooser)
            index += 1
        return lines

    def end_round(self):
        """Naikkan nomor ronde dan putuskan apakah bantuan musuh tiba."""
        self.round_no += 1
        if not self._should_reinforce():
            return []
        drone = self._spawn_drone()
        self.reinforcement_count += 1
        return [f"BANTUAN! Drone {drone.loadout.name} #{drone.drone_id} tiba."]

    # ---- internal -------------------------------------------------------
    def _spawn_drone(self):
        keys = list(LOADOUT_SPAWN_WEIGHTS)
        weights = [LOADOUT_SPAWN_WEIGHTS[k] for k in keys]
        key = self.rng.choices(keys, weights=weights, k=1)[0]
        drone = Drone(self.next_drone_id, self.loadouts[key])
        self.next_drone_id += 1
        self.drones.append(drone)
        return drone

    def _should_reinforce(self):
        if self.round_no <= self.params.reinforce_after:
            return False
        if len(self.alive_drones()) >= MAX_ACTIVE_DRONES:
            return False
        if self.reinforcement_count >= MAX_REINFORCEMENTS:
            return False
        return self.rng.random() * PERCENT_BASE < self.params.reinforce_chance_pct

    def _find_target(self, target_id):
        matches = [d for d in self.alive_drones() if d.drone_id == target_id]
        if not matches:
            raise ValueError(f"Target drone #{target_id} tidak ada atau sudah hancur.")
        return matches[0]

    def _absorb(self, drone, new_state):
        """Salin hasil duel 1-lawan-1 kembali ke tank & drone sungguhan."""
        tank_damage = self.tank_hp - new_state.playerHP
        if tank_damage > 0:
            self.damage_by_loadout[drone.loadout.key] += tank_damage
        self.tank_hp = new_state.playerHP
        self.tank_repair_kits = new_state.playerRepairKits
        self.tank_defending = new_state.playerDefending
        drone.hp = new_state.npcHP
        drone.repair_kits = new_state.npcRepairKits
        drone.defending = new_state.npcDefending

    def _drone_act(self, drone, chooser):
        drone.charge += 1
        if drone.charge < drone.loadout.act_interval:
            return []  # masih mengisi daya; tidak ada kejadian untuk dilaporkan
        drone.charge = 0
        state = self.projection(drone, NPC)
        action = ACT_ATTACK if drone.loadout.suicide else chooser(state)
        if action not in (ACT_ATTACK, ACT_DEFEND, ACT_REPAIR):
            raise ValueError(f"Aksi drone tidak dikenal: {action!r}")
        self._absorb(drone, state.apply_action(action))
        return [self._drone_line(drone, action)]

    def _player_line(self, action, target):
        if action == ACT_ATTACK:
            return f"Tank: Serang #{target.drone_id} {target.loadout.name} (HP {max(0, target.hp):.0f})"
        return f"Tank: {ACTION_LABELS[action]} (HP {max(0, self.tank_hp):.0f})"

    def _drone_line(self, drone, action):
        label = "MELEDAK" if drone.loadout.suicide and action == ACT_ATTACK else ACTION_LABELS[action]
        return f"#{drone.drone_id} {drone.loadout.name}: {label} (HP tank {max(0, self.tank_hp):.0f})"
