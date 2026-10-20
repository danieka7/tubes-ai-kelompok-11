from battle_constants import ACT_ATTACK, ACT_DEFEND, ACT_REPAIR


# =============================================================================
# BAGIAN 5B.1 -- BERBAGAI FUNGSI EVALUASI (untuk eksperimen "tingkah laku NPC")
# =============================================================================
# Semua Eval(s) di bawah dipakai HANYA pada cutoff non-terminal (depth limit),
# bukan sebagai utility. Bobot berbeda -> "kepribadian" NPC berbeda meski
# algoritma pencariannya (minimax/alpha-beta) sama persis.

def battle_eval_seimbang(state):
    """Seimbang: HP jadi prioritas utama, kit reparasi & status bertahan sekunder."""
    score = 1.0 * (state.npcHP - state.playerHP)
    score += 6.0 * (state.npcRepairKits - state.playerRepairKits)
    score += 8.0 * (1 if state.npcDefending else 0)
    score -= 8.0 * (1 if state.playerDefending else 0)
    return score


def battle_eval_agresif(state):
    """Agresif: selisih HP dibobot sangat tinggi, kit reparasi/bertahan nyaris
    diabaikan -> NPC cenderung terus menyerang, jarang bertahan/reparasi diri."""
    score = 4.0 * (state.npcHP - state.playerHP)
    score += 1.0 * (state.npcRepairKits - state.playerRepairKits)
    score += 2.0 * (1 if state.npcDefending else 0)
    score -= 2.0 * (1 if state.playerDefending else 0)
    return score


def battle_eval_defensif(state):
    """Defensif: status bertahan & HP sendiri dibobot sangat tinggi ->
    NPC cenderung bertahan/reparasi diri lebih sering, menghindari HP rendah."""
    score = 1.0 * (state.npcHP - state.playerHP)
    score += 5.0 * (state.npcRepairKits - state.playerRepairKits)
    score += 20.0 * (1 if state.npcDefending else 0)
    score -= 20.0 * (1 if state.playerDefending else 0)
    if state.npcHP < 40:
        score += 25.0  # panik kalau HP sendiri kritis -> makin ingin bertahan/reparasi
    return score


def battle_eval_hemat_reparasi(state):
    """Hemat reparasi: selisih jumlah kit reparasi dibobot paling berat -> NPC
    berusaha menimbun/tidak boros kit reparasi dibanding menyerang."""
    score = 0.5 * (state.npcHP - state.playerHP)
    score += 15.0 * (state.npcRepairKits - state.playerRepairKits)
    score += 4.0 * (1 if state.npcDefending else 0)
    score -= 4.0 * (1 if state.playerDefending else 0)
    return score


# key -> (fungsi, label tampilan)
EVAL_FUNCTIONS = {
    "seimbang": (battle_eval_seimbang, "Seimbang"),
    "agresif": (battle_eval_agresif, "Agresif"),
    "defensif": (battle_eval_defensif, "Defensif"),
    "hemat": (battle_eval_hemat_reparasi, "Hemat Reparasi"),
}

# Dipertahankan agar nama lama tetap bisa dipakai di tempat lain / kompatibel.
battle_eval_default = battle_eval_seimbang

# key -> (urutan aksi, label tampilan) -- dipakai utk eksperimen "urutan aksi"
# terhadap efisiensi alpha-beta pruning (move ordering).
ACTION_ORDERS = {
    "default": ([ACT_ATTACK, ACT_DEFEND, ACT_REPAIR], "Serang -> Bertahan -> Reparasi"),
    "bertahan_dulu": ([ACT_DEFEND, ACT_ATTACK, ACT_REPAIR], "Bertahan -> Serang -> Reparasi"),
    "reparasi_dulu": ([ACT_REPAIR, ACT_DEFEND, ACT_ATTACK], "Reparasi -> Bertahan -> Serang"),
}
