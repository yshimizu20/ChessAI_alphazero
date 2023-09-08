import torch
import torch.nn as nn
import chess
import random

from chess_engine.model.model import ChessModel
from chess_engine.utils.state import createStateObj
from chess_engine.utils.dataloader import TestLoader
from chess_engine.utils.utils import uci_dict, uci_table
from chess_engine.mcts import MCTSAgent

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.001
batch_size = 200
num_games = 200
MAX_MOVES = 150

testing_iterator = TestLoader("datasets/validation/lichess_elite_2023-07.pgn")


def self_play(
    start_epoch=0,
    end_epoch=5000,
    model_path=None,
):
    model = ChessModel()
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    model.to(device)

    agent = MCTSAgent(model)

    num_epochs = end_epoch - start_epoch
    log_path = f"log_{start_epoch}.txt"

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(start_epoch, end_epoch):
        X, y, win = play_game(agent)

        X = torch.stack(X, dim=0).to(device)
        y = torch.stack(y, dim=0).to(device)
        win = torch.stack(win).unsqueeze(1).to(device)

        model.train()
        policy, value = model(X)

        # train both networks
        optimizer.zero_grad()
        policy_loss = policy_criterion(policy, y)
        value_loss = value_criterion(value, win)

        alpha = 0.5
        beta = 0.5
        loss = alpha * policy_loss + beta * value_loss
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1} of {num_epochs}")
        print(f"Policy Loss: {policy_loss}, Value Loss: {value_loss}")
        with open(log_path, "a") as fp:
            fp.write(f"Epoch {epoch + 1} of {num_epochs}\n")
            fp.write(f"Policy Loss: {policy_loss}, Value Loss: {value_loss}\n")

        del X, y, win, policy, value

        if epoch % 10 == 0:
            # evaluate
            X, y, win = (
                testing_iterator.X.to(device),
                testing_iterator.y.to(device),
                testing_iterator.win.to(device),
            )

            with torch.no_grad():
                policy, value = model(X)
                policy_loss = policy_criterion(policy, y)
                value_loss = value_criterion(value, win)

            print(f"Test Policy Loss: {policy_loss}, Test Value Loss: {value_loss}")
            with open(log_path, "a") as fp:
                fp.write(
                    f"Test Policy Loss: {policy_loss}, Test Value Loss: {value_loss}\n"
                )

            del X, y, win, policy, value

        # save model
        if epoch % 100 == 99:
            torch.save(model.state_dict(), f"saved_models/model_{epoch + 1}.pt")


def play_game(agent):
    data = []

    for i in range(num_games):
        game = chess.pgn.Game()
        board = game.board()
        states, moves = [], []
        n_moves = 0

        while not board.is_game_over() and n_moves < MAX_MOVES:
            # select best move based on MCTS
            best_move = agent.action(game, board)

            moves.append(best_move)
            board.push_uci(best_move)
            print("Move:", best_move)
            n_moves += 1

        # get winner
        winner = 0.0
        if board.result() == "1-0":
            winner = 1.0
        elif board.result() == "0-1":
            winner = -1.0

        # add to data
        for i in range(len(states)):
            data.append((states[i], moves[i], winner))

        print("game reached end. Result: ", board.result())

    random.shuffle(data)
    X, y, win = zip(*data)
    return X, y, win
