import torch
import torch.nn as nn
import chess
import random
import os
from concurrent.futures import ProcessPoolExecutor

from chess_engine.model.model import ChessModel
from chess_engine.utils.state import createStateObj
from chess_engine.utils.dataloader import TestLoader
from chess_engine.utils.utils import uci_dict, uci_table, get_time
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
    log_path=None,
):
    num_epochs = end_epoch - start_epoch
    if not log_path:
        log_path = f"log_{start_epoch}.txt"

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    for epoch in range(start_epoch, end_epoch):
        # initialize model
        if model_path is None:
            # initialize model and save weights
            model = ChessModel()
            new_model_path = f"saved_models/model_0.pt"
            torch.save(model.state_dict(), new_model_path)
            model_path = new_model_path
            os.system(f"cp {model_path} saved_models/current_best")
            del model

        X, y, win = [], [], []
        N_GAMES = 3
        args = [(model_path, model_path, 1, i + 1, log_path) for i in range(N_GAMES)]

        with ProcessPoolExecutor(max_workers=3) as executor:
            results = executor.map(play_games_async, *zip(*args))

        for result in results:
            X_batch, y_batch, win_batch, _ = result

            X += X_batch
            y += y_batch
            win += win_batch

        X = torch.stack(X, dim=0).to(device)
        y = torch.stack(y, dim=0).to(device)
        win = torch.tensor(win).unsqueeze(1).to(device)

        # initialize model
        model = ChessModel()
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # train model
        model.train()
        optimizer.zero_grad()

        policy, value = model(X)

        loss = (value - win).pow(2).sum() - (y * torch.log(policy)).sum()

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        print(f"Epoch {epoch + 1} of {num_epochs}")
        # print(f"Policy Loss: {policy_loss}, Value Loss: {value_loss}")
        # with open(log_path, "a") as fp:
        #     fp.write(f"[{get_time()}] Epoch {epoch + 1} of {num_epochs}\n")
        #     fp.write(f"[{get_time()}] Policy Loss: {policy_loss}, Value Loss: {value_loss}\n")

        del X, y, win, policy, value

        # save model
        new_model_path = f"saved_models/model_{epoch + 1}.pt"
        torch.save(model.state_dict(), new_model_path)

        # # debug
        # nan_found = False
        # for name, param in model.named_parameters():
        #     if torch.isnan(param).any():
        #         print(f"NaNs found in parameter: {name}")
        #         nan_found = True

        # assert nan_found == False

        del model

        if model_path is None:
            raise ZeroDivisionError

        else:
            # run 20 games against previous model
            prev_wins, new_wins = evaluate(model_path, new_model_path, 1)

            if new_wins / (prev_wins + new_wins) > 0.55:
                # update model_path
                model_path = new_model_path

                # empty the models/current_best folder
                for f in os.listdir("saved_models/current_best"):
                    os.remove(os.path.join("saved_models/current_best", f))

                # copy the new model to models/current_best
                os.system(f"cp {model_path} saved_models/current_best")

        # evaluate
        if epoch % 10 == 9:
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
                    f"[{get_time()}] Test Policy Loss: {policy_loss}, Test Value Loss: {value_loss}\n"
                )

            del X, y, win, policy, value

        # save model
        if epoch % 10 == 9:
            torch.save(model.state_dict(), f"saved_models/model_{epoch + 1}.pt")


def play_game(agent1, agent2, id_=None, log_path=None):
    """
    Play one game of chess and return the generated data
    """

    data = []

    game = chess.pgn.Game()
    board = game.board()
    states, moves = [], []
    n_moves = 0
    agents = random.sample([agent1, agent2], 2)

    while not board.is_game_over() and n_moves < MAX_MOVES:
        agent = agents[n_moves % 2]
        # select best move based on MCTS
        best_move, policy = agent.action(game, board)

        states.append(createStateObj(board))
        policy = torch.tensor(policy).unsqueeze(0)
        moves.append(policy)
        board.push_uci(best_move)

        if id_ is not None:
            print(f"Process {id_} Move No. {n_moves} - {best_move}")
            if log_path is not None:
                with open(log_path, "a") as fp:
                    fp.write(
                        f"[{get_time()}] Process {id_} Move No. {n_moves} - {best_move}\n"
                    )
        else:
            print(f"Move No. {n_moves} - {best_move}")
            if log_path is not None:
                with open(log_path, "a") as fp:
                    fp.write(f"[{get_time()}] Move No. {n_moves} - {best_move}\n")

        n_moves += 1
        break  # TODO: remove this

    # get winner
    win_val = 1.0  # TODO: change this to 0.0
    winner = "a"  # TODO: change this to ""

    if board.result() == "1-0":
        win_val = 1.0
        winner = agents[0].name
    elif board.result() == "0-1":
        win_val = -1.0
        winner = agents[1].name

    print("game reached end. Result: ", board.result())

    X, y, win = states, moves, [win_val for _ in range(len(states))]

    return X, y, win, winner


def play_games_async(model_path1, model_path2, n_games=1, id_=None, log_path=None):
    model1 = ChessModel()
    if model_path1 is not None:
        model1.load_state_dict(torch.load(model_path1))
    model1.to(device)

    model2 = ChessModel()
    if model_path2 is not None:
        model2.load_state_dict(torch.load(model_path2))
    model2.to(device)

    agent1 = MCTSAgent(model1)
    agent2 = MCTSAgent(model2)

    print(f"Process {id_} playing {n_games} games")

    X, y, win, winner = [], [], [], []

    for _ in range(n_games):
        X_batch, y_batch, win_batch, winner_batch = play_game(
            agent1, agent2, id_, log_path
        )

        # during training, we only want to keep the games where there is a winner
        if winner_batch == "":
            continue

        X += X_batch
        y += y_batch
        win += win_batch
        winner += winner_batch

    return X, y, win, winner


def evaluate(prev_model_path, new_model_path, rounds=2):
    """
    Evaluate the new model against the previous model by playing rounds games
    """

    print(prev_model_path, new_model_path)

    assert prev_model_path is not None
    assert new_model_path is not None

    prev_model = ChessModel()
    prev_model.load_state_dict(torch.load(prev_model_path))
    prev_model.to(device)

    new_model = ChessModel()
    new_model.load_state_dict(torch.load(new_model_path))
    new_model.to(device)

    nan_found = False
    for name, param in prev_model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaNs found in parameter: {name}")
            nan_found = True

    if nan_found:
        print("NaNs found in previous model. Exiting...")
    else:
        print("No NaNs found in previous model")

    nan_found = False
    for name, param in new_model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaNs found in parameter: {name}")
            nan_found = True

    if nan_found:
        print("NaNs found in new model. Exiting...")
        exit(1)
    else:
        print("No NaNs found in new model")

    prev_agent = MCTSAgent(prev_model, "Previous Model")
    new_agent = MCTSAgent(new_model, "New Model")

    prev_wins, new_wins = 0, 0

    curr_round = 0

    while curr_round < rounds:
        X, y, win, winner = play_game(prev_agent, new_agent)
        if winner == prev_agent.name:
            prev_wins += 1
        elif winner == new_agent.name:
            new_wins += 1
        else:
            continue

        curr_round += 1

    print(f"Previous Model Wins: {prev_wins}, New Model Wins: {new_wins}")

    return prev_wins, new_wins
