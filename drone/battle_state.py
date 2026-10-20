from battle_constants import (
    ACT_ATTACK,
    ACT_DEFEND,
    ACT_REPAIR,
    ATTACK_DAMAGE,
    BATTLE_START_REPAIR_KITS,
    BATTLE_WIN_SCORE,
    DEFEND_REDUCTION,
    NPC,
    PLAYER,
    REPAIR_HEAL_AMOUNT,
)
from battle_eval import ACTION_ORDERS
from loadouts import DEFAULT_NPC_LOADOUT, TUNED_PARAMS


class BattleState:
    """State duel 1 lawan 1 (tank vs SATU drone) yang dipakai pohon minimax.

    s = (playerHP, npcHP, playerRepairKits, npcRepairKits,
         playerDefending, npcDefending, turnOwner, turnIndex)
    ditambah playerMaxHP (batas reparasi tank) dan npcSpec (Loadout drone).
    Pertempuran banyak drone dipecah jadi beberapa duel semacam ini oleh Encounter.
    """
    __slots__ = (
        "playerHP", "npcHP", "playerRepairKits", "npcRepairKits",
        "playerDefending", "npcDefending", "turnOwner", "turnIndex",
        "playerMaxHP", "npcSpec",
    )

    def __init__(self, playerHP, npcHP, playerRepairKits, npcRepairKits,
                 playerDefending, npcDefending, turnOwner, turnIndex,
                 playerMaxHP=None, npcSpec=None):
        self.playerHP = playerHP
        self.npcHP = npcHP
        self.playerRepairKits = playerRepairKits
        self.npcRepairKits = npcRepairKits
        self.playerDefending = playerDefending
        self.npcDefending = npcDefending
        self.turnOwner = turnOwner
        self.turnIndex = turnIndex
        self.playerMaxHP = TUNED_PARAMS.tank_hp if playerMaxHP is None else playerMaxHP
        self.npcSpec = DEFAULT_NPC_LOADOUT if npcSpec is None else npcSpec

    @staticmethod
    def initial():
        return BattleState(
            playerHP=TUNED_PARAMS.tank_hp, npcHP=DEFAULT_NPC_LOADOUT.max_hp,
            playerRepairKits=BATTLE_START_REPAIR_KITS, npcRepairKits=DEFAULT_NPC_LOADOUT.repair_kits,
            playerDefending=False, npcDefending=False,
            turnOwner=PLAYER, turnIndex=0,
        )

    def is_terminal(self):
        """Terminal(s) = (playerHP <= 0) or (npcHP <= 0)."""
        return self.playerHP <= 0 or self.npcHP <= 0

    def legal_actions(self, order=None):
        """Branching <=3: ATTACK, DEFEND selalu ada; REPAIR hilang kalau stok habis.

        `order` (opsional): urutan pengecekan aksi (list of ACT_*), dipakai
        untuk eksperimen "urutan aksi" (move ordering) terhadap efisiensi
        alpha-beta pruning. Default: ATTACK, DEFEND, REPAIR."""
        order = order or ACTION_ORDERS["default"][0]
        repair_kits = self.playerRepairKits if self.turnOwner == PLAYER else self.npcRepairKits
        actions = []
        for a in order:
            if a == ACT_REPAIR and repair_kits <= 0:
                continue
            actions.append(a)
        return actions

    def _attack_damage(self, is_player_turn, opp_defending):
        """Damage serangan giliran ini setelah memperhitungkan bertahan lawan.

        Serangan drone memakai damage loadout-nya dan bisa menembus sebagian
        efek bertahan tank (pierce); serangan tank memakai ATTACK_DAMAGE."""
        if is_player_turn:
            base_damage, pierce = ATTACK_DAMAGE, 0.0
        else:
            base_damage, pierce = self.npcSpec.damage, self.npcSpec.pierce
        if not opp_defending:
            return base_damage
        return base_damage * (1.0 - DEFEND_REDUCTION * (1.0 - pierce))

    def apply_action(self, action):
        """Kembalikan BattleState BARU (tidak memutasi state ini) supaya aman
        dipakai berulang kali di pohon pencarian minimax."""
        is_player = self.turnOwner == PLAYER
        own_hp, opp_hp = (self.playerHP, self.npcHP) if is_player else (self.npcHP, self.playerHP)
        own_kits = self.playerRepairKits if is_player else self.npcRepairKits
        opp_defending = self.npcDefending if is_player else self.playerDefending
        own_max_hp = self.playerMaxHP if is_player else self.npcSpec.max_hp

        if action == ACT_ATTACK:
            opp_hp -= self._attack_damage(is_player, opp_defending)
            if not is_player and self.npcSpec.suicide:
                own_hp = 0  # kamikaze hancur bersama ledakannya
        elif action == ACT_REPAIR:
            own_hp = min(own_max_hp, own_hp + REPAIR_HEAL_AMOUNT)
            own_kits -= 1
        # ACT_DEFEND: tidak menyerang, tidak memulihkan -- hanya set flag di bawah

        own_defending = (action == ACT_DEFEND)
        next_owner = NPC if is_player else PLAYER

        if is_player:
            return BattleState(
                playerHP=own_hp, npcHP=opp_hp,
                playerRepairKits=own_kits, npcRepairKits=self.npcRepairKits,
                playerDefending=own_defending, npcDefending=opp_defending,
                turnOwner=next_owner, turnIndex=self.turnIndex + 1,
                playerMaxHP=self.playerMaxHP, npcSpec=self.npcSpec,
            )
        return BattleState(
            playerHP=opp_hp, npcHP=own_hp,
            playerRepairKits=self.playerRepairKits, npcRepairKits=own_kits,
            playerDefending=opp_defending, npcDefending=own_defending,
            turnOwner=next_owner, turnIndex=self.turnIndex + 1,
            playerMaxHP=self.playerMaxHP, npcSpec=self.npcSpec,
        )


def battle_utility(state):
    """U(s) -- hanya didefinisikan di terminal SEJATI, sudut pandang NPC (MAX).

    U(s) = +WIN_SCORE - turnIndex   jika playerHP <= 0 (NPC menang)
    U(s) = -WIN_SCORE + turnIndex   jika npcHP <= 0    (player menang)

    -turnIndex / +turnIndex membuat NPC lebih menyukai kemenangan cepat
    dan kekalahan lambat -- ini asumsi desain, bukan hasil perhitungan.
    """
    if state.playerHP <= 0:
        return BATTLE_WIN_SCORE - state.turnIndex
    return -BATTLE_WIN_SCORE + state.turnIndex
