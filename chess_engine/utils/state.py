import chess
import chess.pgn
import numpy as np
import torch
import itertools

from chess_engine.utils.utils import uci_dict, uci_table, piece_values


def createStateObj(board):
    # convert state into 19 x 8 x 8 tensor
    state = torch.zeros((19, 8, 8), dtype=torch.float32)

    piece_map = board.piece_map()
    piece_types = [
        piece.piece_type - 1 if piece.color == chess.WHITE else piece.piece_type + 5
        for piece in board.piece_map().values()
    ]
    squares = [square for square in piece_map.keys()]
    state[
        piece_types,
        [square // 8 for square in squares],
        [square % 8 for square in squares],
    ] = 1.0
    net_piece_value = sum([piece_values[str(piece)] for piece in piece_map.values()])

    # append the 5 states above
    p1_can_castle_queenside = board.has_queenside_castling_rights(chess.WHITE)
    p1_can_castle_kingside = board.has_kingside_castling_rights(chess.WHITE)
    p2_can_castle_queenside = board.has_queenside_castling_rights(chess.BLACK)
    p2_can_castle_kingside = board.has_kingside_castling_rights(chess.BLACK)
    turn = board.turn

    state[12, :, :] = float(p1_can_castle_queenside)
    state[13, :, :] = float(p1_can_castle_kingside)
    state[14, :, :] = float(p2_can_castle_queenside)
    state[15, :, :] = float(p2_can_castle_kingside)
    state[16, :, :] = float(turn)
    state[17, :, :] = float(net_piece_value)

    # enpassant
    if board.ep_square is not None:
        state[18, board.ep_square // 8, board.ep_square % 8] = 1.0

    return state


tag = {"1/2-1/2": 0.0, "1-0": 1.0, "0-1": -1.0}


def createData(fp, n_data=200):
    X, y, win = [], [], []

    for i in range(n_data):
        game = chess.pgn.read_game(fp)
        if game is None:
            if len(X) == 0:
                return [], [], []
            return X, y, win

        try:
            w = tag[game.headers["Result"]]
        except KeyError:
            continue

        board = game.board()

        lst = list(game.mainline_moves())
        len_lst = len(lst)
        for move in lst[:-1]:
            X.append(createStateObj(board))
            board.push(move)
            y_ele = torch.zeros(1968, dtype=torch.float32)
            try:
                y_ele[uci_dict[str(move)]] = 1.0
            except KeyError:
                X.pop()
                break
            y.append(y_ele)
            win.append(torch.tensor(w * (i + 1) / len_lst, dtype=torch.float32))

    return X, y, win
