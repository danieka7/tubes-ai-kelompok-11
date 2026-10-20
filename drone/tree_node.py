class TreeNode:
    """Satu node di pohon pencarian minimax, dicatat untuk visualisasi debug.

    `action`  -- aksi yang membawa dari parent ke node ini (None untuk root).
    `owner`   -- 'MAX' (giliran NPC) atau 'MIN' (giliran player) di node ini.
    `kind`    -- 'internal' | 'terminal' | 'cutoff' (lihat Cutoff(s, depth) di desain).
    `pruned`  -- True kalau node ini sendiri TIDAK PERNAH dieksplorasi karena
                 alpha-beta memangkas cabang saudaranya (dipakai untuk menggambar
                 cabang yang terpotong, contoh klasik "move ordering").
    `prob`    -- probabilitas cabang ini di node CHANCE (mode Expectimax saja).
    `snapshot`-- salinan ringkas state di node ini (HP/kit reparasi/bertahan/
                 turnIndex), dipakai HANYA untuk panel debug "arahkan mouse
                 ke node".
    """
    __slots__ = (
        "action", "depth", "owner", "value", "kind", "pruned", "prob",
        "snapshot", "children", "x", "y",
    )

    def __init__(self, action, depth, owner=None):
        self.action = action
        self.depth = depth
        self.owner = owner
        self.value = None
        self.kind = None
        self.pruned = False
        self.prob = None
        self.snapshot = None
        self.children = []
        self.x = 0.0
        self.y = 0.0
