from collections import defaultdict
import numpy as np
import torch
import chess

from chess_engine.utils.utils import uci_dict, uci_table
from chess_engine.utils.state import createStateObj


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Visits:
    def __init__(self):
        self.a = defaultdict(Actions)
        self.sum_n = 0


class Actions:
    def __init__(self):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0


class MCTSAgent:
    def __init__(self, chess_net):
        self.tree = defaultdict(Visits)
        self.chess_net = chess_net


    def reset(self):
        self.tree = defaultdict(Visits)


    def action(self, game, board):
        self.reset()
        val = self._search_moves(board)
        policy = self._calc_policy(board)
        my_action = int(np.random.choice(range(1968), p=self._apply_temperature(policy, len(list(game.mainline_moves())))))

        return uci_table[my_action]


    def _search_moves(self, board):
        vals = []
        for _ in range(8000):
            vals.append(self._search_my_moves(board.copy(), True))

        return max(vals)


    def _search_my_moves(self, board, is_root=False):
        if board.is_game_over():
            res = 0.0
            if board.result() == "1-0":
                res = 1.0
            elif board.result() == "0-1":
                res = -1.0
            return res

        board_str = board.fen()

        if board_str not in self.tree:
            leaf_p, leaf_v = self._expand(board)
            self.tree[board_str].p = leaf_p.squeeze().numpy()
            return leaf_v

        action_t = self._select_action(board, is_root)
        virtual_loss = 3
        is_white = 1 if board.turn == chess.WHITE else -1
        visit_stats = self.tree[board_str]
        my_stats = visit_stats.a[action_t]

        visit_stats.sum_n += virtual_loss
        my_stats.n += virtual_loss
        my_stats.w -= virtual_loss * is_white
        my_stats.q = my_stats.w / my_stats.n * is_white
    
        board.push_uci(action_t.uci())
        leaf_v = self._search_my_moves(board)

        visit_stats.sum_n += virtual_loss + 1
        my_stats.n += -virtual_loss + 1
        my_stats.w += virtual_loss * is_white + leaf_v
        my_stats.q = my_stats.w / my_stats.n * is_white

        return leaf_v


    def _expand(self, board):
        state = createStateObj(board).unsqueeze(0).to(device)
        
        self.chess_net.eval()
        with torch.no_grad():
            leaf_p, leaf_v = self.chess_net(state)
        del state
        leaf_p = leaf_p.cpu()
        leaf_v = leaf_v.cpu().item()

        return leaf_p, leaf_v


    def _select_action(self, board, is_root=False):
        board_str = board.fen()
        visit_stats = self.tree[board_str]

        if visit_stats.p is not None:
            tot_p = 1e-8
            for mv in board.legal_moves:
                mv_p = visit_stats.p[uci_dict[str(mv)]]
                visit_stats.a[mv].p = mv_p
                tot_p += mv_p
            for a_s in visit_stats.a.values():
                a_s.p /= tot_p
            visit_stats.p = None

        xx = np.sqrt(visit_stats.sum_n + 1)

        e = 0 # noise epsilon
        c_puct = 1.5
        dir_alpha = 0.3

        best_s = -999
        best_a = None

        if is_root:
            noise = np.random.dirichlet([dir_alpha] * len(visit_stats.a))
        
        i = 0
        for action, a_s in visit_stats.a.items():
            p = a_s.p
            if is_root:
                p = (1-e) * p + e * noise[i]
                i += 1
            b = a_s.q + c_puct * p * xx / (1 + a_s.n)
            if b > best_s:
                best_s = b
                best_a = action

        return best_a


    def _apply_temperature(self, policy, turn):
        tau = np.power(0.99, turn + 1)

        if tau < 0.1:
            tau = 0

        if tau == 0:
            action = np.argmax(policy)
            ret = np.zeros(1968)
            ret[action] = 1.0
        else:
            ret = np.power(policy, 1/tau)
            ret /= np.sum(ret)
        
        return ret


    def _calc_policy(self, board):
        board_str = board.fen()
        visit_stats = self.tree[board_str]
        policy = np.zeros(1968)

        for action, a_s in visit_stats.a.items():
            policy[uci_dict[str(action)]] = a_s.n

        policy /= np.sum(policy)
        return policy
