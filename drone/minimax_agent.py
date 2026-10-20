import math

from battle_constants import BATTLE_DEFAULT_DEPTH, MODE_ALPHABETA, MODE_EXPECTIMAX, NPC
from battle_eval import battle_eval_default
from battle_state import battle_utility
from tree_node import TreeNode


def _snapshot_of(state):
    return (
        state.playerHP, state.npcHP, state.playerRepairKits, state.npcRepairKits,
        state.playerDefending, state.npcDefending, state.turnIndex,
    )


class MinimaxAgent:
    """Agen NPC: memilih aksi lewat Minimax / Alpha-Beta / Expectimax.

    Mencatat nodes_expanded, terminal_count, cutoff_count, pruned_count &
    root_action_scores -- semuanya dibutuhkan untuk panel debug di BattleHUD --
    dan opsional membangun `last_tree` (TreeNode) untuk divisualisasikan.
    """

    def __init__(self):
        self.nodes_expanded = 0
        self.terminal_count = 0
        self.cutoff_count = 0
        self.pruned_count = 0
        self.root_action_scores = []  # [(action, score), ...] milik keputusan terakhir
        self.last_tree = None         # TreeNode root milik keputusan terakhir (untuk visualisasi)

    def _search(self, state, depth, alpha, beta, max_depth, mode, eval_fn, action_order, node):
        self.nodes_expanded += 1
        if node is not None:
            if state.turnOwner == NPC:
                node.owner = "MAX"
            elif mode == MODE_EXPECTIMAX:
                node.owner = "CHANCE"
            else:
                node.owner = "MIN"
            node.snapshot = _snapshot_of(state)

        # Cutoff(s, depth) = Terminal(s) or (depth >= MAX_DEPTH)
        if state.is_terminal():
            val = battle_utility(state)            # terminal sejati -> utility
            self.terminal_count += 1
            if node is not None:
                node.kind = "terminal"
                node.value = val
            return val
        if depth >= max_depth:
            val = eval_fn(state)                    # cutoff karena depth -> evaluation
            self.cutoff_count += 1
            if node is not None:
                node.kind = "cutoff"
                node.value = val
            return val

        if node is not None:
            node.kind = "internal"

        actions = state.legal_actions(action_order)

        if state.turnOwner == NPC:  # node MAX -- giliran NPC, selalu deterministik
            best = -math.inf
            for i, a in enumerate(actions):
                child = TreeNode(a, depth + 1) if node is not None else None
                val = self._search(state.apply_action(a), depth + 1, alpha, beta, max_depth, mode, eval_fn, action_order, child)
                if child is not None:
                    child.value = val
                    node.children.append(child)
                if val > best:
                    best = val
                if mode == MODE_ALPHABETA:
                    alpha = max(alpha, best)
                    if alpha >= beta:
                        self.pruned_count += len(actions[i + 1:])
                        if node is not None:
                            for remaining in actions[i + 1:]:
                                stub = TreeNode(remaining, depth + 1)
                                stub.pruned = True
                                node.children.append(stub)
                        break  # beta cutoff
            if node is not None:
                node.value = best
            return best

        elif mode == MODE_EXPECTIMAX:  # node CHANCE -- giliran player dimodelkan acak
            # Asumsi desain: NPC TIDAK tahu persis aksi player berikutnya, jadi
            # menganggap semua aksi legal player sama mungkin (uniform) dan
            # memilih aksi yang memaksimalkan NILAI HARAPAN (expected value),
            # bukan skenario terburuk seperti Minimax/Alpha-Beta murni.
            total = 0.0
            prob = 1.0 / len(actions) if actions else 0.0
            for a in actions:
                child = TreeNode(a, depth + 1) if node is not None else None
                val = self._search(state.apply_action(a), depth + 1, -math.inf, math.inf, max_depth, mode, eval_fn, action_order, child)
                if child is not None:
                    child.value = val
                    child.prob = prob
                    node.children.append(child)
                total += prob * val
            if node is not None:
                node.value = total
            return total

        else:  # node MIN -- Minimax biasa atau Alpha-Beta
            best = math.inf
            for i, a in enumerate(actions):
                child = TreeNode(a, depth + 1) if node is not None else None
                val = self._search(state.apply_action(a), depth + 1, alpha, beta, max_depth, mode, eval_fn, action_order, child)
                if child is not None:
                    child.value = val
                    node.children.append(child)
                if val < best:
                    best = val
                if mode == MODE_ALPHABETA:
                    beta = min(beta, best)
                    if beta <= alpha:
                        self.pruned_count += len(actions[i + 1:])
                        if node is not None:
                            for remaining in actions[i + 1:]:
                                stub = TreeNode(remaining, depth + 1)
                                stub.pruned = True
                                node.children.append(stub)
                        break  # alpha cutoff
            if node is not None:
                node.value = best
            return best

    def choose_action(self, state, depth=BATTLE_DEFAULT_DEPTH, mode=MODE_ALPHABETA,
                       eval_fn=battle_eval_default, action_order=None, build_tree=True):
        self.nodes_expanded = 0
        self.terminal_count = 0
        self.cutoff_count = 0
        self.pruned_count = 0
        self.root_action_scores = []
        self.last_tree = None

        alpha, beta = -math.inf, math.inf
        best_action = None
        best_score = -math.inf

        root = TreeNode(None, 0, "MAX" if state.turnOwner == NPC else "MIN") if build_tree else None
        if root is not None:
            root.snapshot = _snapshot_of(state)

        # Catatan: di root sendiri SEMUA aksi tetap dicoba (tidak dipangkas) supaya
        # root_action_scores lengkap untuk overlay debug -- pruning hanya terjadi
        # di level yang lebih dalam.
        for a in state.legal_actions(action_order):
            child_state = state.apply_action(a)
            child_node = TreeNode(a, 1) if build_tree else None
            score = self._search(child_state, 1, alpha, beta, depth, mode, eval_fn, action_order, child_node)
            if child_node is not None:
                child_node.value = score
                root.children.append(child_node)
            self.root_action_scores.append((a, score))
            if score > best_score:
                best_score = score
                best_action = a
            if mode == MODE_ALPHABETA:
                alpha = max(alpha, best_score)

        if build_tree:
            root.value = best_score
            root.kind = "internal"
            self.last_tree = root

        return best_action
