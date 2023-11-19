from collections import defaultdict
import numpy as np
import torch
import chess

from chess_engine.utils.utils import uci_dict, uci_table
from chess_engine.utils.state import createStateObj


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MCTSNode:
    def __init__(self):
        self.actions = defaultdict(MCTSNode)
        self.expanded = False
        self.p_arr = None

        self.n = 0
        self.w = 0
        self.p = 0
        self.q = 0


class MCTSAgent:
    WHITE = 1
    BLACK = -1

    def __init__(self, chess_net, name="No Name"):
        self.tree = defaultdict(MCTSNode)
        self.chess_net = chess_net
        self.name = name

    def _reset(self):
        self.tree = defaultdict(MCTSNode)
        print("tree reset")
        print("tree size:", len(self.tree))

    def action(self, game, board):
        # reset the tree
        self._reset()

        # add root node to the tree
        self._expand_root(board)

        # populate self.tree with the MCTS algorithm
        self._populate_tree(board)

        policy = self._calc_policy(board)

        my_action = int(
            np.random.choice(
                range(1968),
                p=self._apply_temperature(policy, len(list(game.mainline_moves()))),
            )
        )

        return uci_table[my_action], policy

    def _populate_tree(self, board) -> int:
        vals = []

        for _ in range(800):
            vals.append(self._search_moves(board.copy(), True))

        return max(vals)

    def _search_moves(self, board, is_root=False):
        if board.is_game_over():
            res = 0.0

            if board.result() == "1-0":
                res = 1.0
            elif board.result() == "0-1":
                res = -1.0

            return res

        board_str = board.fen()
        is_white = MCTSAgent.WHITE if board.turn == chess.WHITE else MCTSAgent.BLACK

        if board_str not in self.tree:
            raise ZeroDivisionError

        if not self.tree[board_str].expanded:
            leaf_v = self._expand(board)

            if is_white == MCTSAgent.WHITE:
                return max(leaf_v)
            else:
                return min(leaf_v)

        action_t = self._select_action(board, is_root)

        virtual_loss = 3
        visit_stats = self.tree[board_str]
        my_stats = visit_stats.actions[action_t]

        my_stats.n += virtual_loss
        my_stats.w -= virtual_loss * is_white
        my_stats.q = my_stats.w / my_stats.n * is_white
        # print("checkpoint 1:", my_stats.q)

        board.push_uci(action_t.uci())
        leaf_v = self._search_moves(board)

        my_stats.n += -virtual_loss + 1
        my_stats.w += virtual_loss * is_white + leaf_v
        my_stats.q = my_stats.w / my_stats.n * is_white

        return leaf_v

    def _expand(self, board) -> np.ndarray:
        board_str = board.fen()
        legal_moves = list(board.legal_moves)

        # get all legal moves and the resulting board states
        boards = []

        for move in legal_moves:
            board.push(move)
            boards.append(board.copy())
            board.pop()

        # get the state representation of the board states
        board_states = [
            createStateObj(next_board).unsqueeze(0).to(device) for next_board in boards
        ]

        # get the policy and value predictions
        self.chess_net.eval()
        with torch.no_grad():
            leaf_p, leaf_v = self.chess_net(torch.cat(board_states, dim=0))
        del board_states

        # convert the policy and value predictions to numpy
        leaf_p = leaf_p.cpu().numpy()
        leaf_v = leaf_v.cpu().numpy()

        # add the newly expanded nodes to the tree
        for move, p_arr, val, next_board in zip(legal_moves, leaf_p, leaf_v, boards):
            self.tree[board_str].actions[move].p = self.tree[board_str].p_arr[
                uci_dict[str(move)]
            ]
            self.tree[board_str].actions[move].q = val.item()
            self.tree[next_board.fen()].p_arr = p_arr

        # set the expanded flag to True
        self.tree[board_str].expanded = True

        return leaf_v

    def _expand_root(self, board) -> None:
        # add the current board to the tree
        state = createStateObj(board).unsqueeze(0).to(device)

        # get the policy and value predictions for all possible next board states
        self.chess_net.eval()
        with torch.no_grad():
            leaf_p, _ = self.chess_net(state)
        del state

        # collect policy predictions
        leaf_p = leaf_p.cpu()
        self.tree[board.fen()].p_arr = leaf_p.squeeze().numpy()

    def _select_action(self, board, is_root=False) -> chess.Move:
        board_str = board.fen()
        visit_stats = self.tree[board_str]

        if visit_stats.p_arr is not None:
            tot_p = 1e-8

            for mv in board.legal_moves:
                mv_p = visit_stats.p_arr[uci_dict[str(mv)]]
                visit_stats.actions[mv].p = mv_p
                tot_p += mv_p

            for a_s in visit_stats.actions.values():
                a_s.p /= tot_p

            visit_stats.p_arr = None

        xx = np.sqrt(visit_stats.n + 1)

        e = 0  # noise epsilon
        c_puct = 1.5
        dir_alpha = 0.3

        best_s = -999
        best_a = None

        if is_root:
            noise = np.random.dirichlet([dir_alpha] * len(visit_stats.actions))

        i = 0

        bs = []

        for action, a_s in visit_stats.actions.items():
            p = a_s.p

            if is_root:
                p = (1 - e) * p + e * noise[i]
                i += 1

            b = a_s.q + c_puct * p * xx / (1 + a_s.n)

            bs.append([a_s.q, p, xx, a_s.n, b])

            if b > best_s:
                best_s = b
                best_a = action

        try:
            assert best_a is not None
        except AssertionError:
            print(len(visit_stats.actions.items()))
            print("bs:", bs)
            raise ZeroDivisionError

        return best_a

    def _apply_temperature(self, policy, turn) -> np.ndarray:
        tau = np.power(0.99, turn + 1)

        if tau < 0.1:
            tau = 0

        if tau == 0:
            action = np.argmax(policy)
            ret = np.zeros(1968)
            ret[action] = 1.0

        else:
            ret = np.power(policy, 1 / tau)
            ret /= np.sum(ret)

        return ret

    def _calc_policy(self, board) -> np.ndarray:
        board_str = board.fen()
        visit_stats = self.tree[board_str]
        policy = np.zeros(1968)

        for action, a_s in visit_stats.actions.items():
            policy[uci_dict[str(action)]] = a_s.n

        policy /= np.sum(policy)
        return policy
