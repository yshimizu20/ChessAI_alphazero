# ChessAI_alphazero

## Description
This is a chess AI that uses the AlphaZero algorithm. It is written in Python and uses PyTorch for the neural network. The neural network is trained using reinforcement learning by playing against itself. The neural network is then used to play against a human player. The distinction between the AlphaGo and AlphaZero algorithms is that AlphaZero does not use any human games to train the neural network, whereas AlphaGo uses human games to train the neural network in the supervised learning portion.

## Current Status
Due to the very expensive nature of the self-play portion, I am yet to start the actual training. Instead, I am currently focused on the AlphaGo algorithm, which is similar to the AlphaZero algorithm, but uses human games to train the neural network in the supervised learning portion (link below), and thus is computationally less expensive. The current version of the neural network, combined with the MCTS algorithm reaches around 1000 ELO.

## Similar Projects
- [Chess Engine using the AlphaGo algorithm](https://github.com/yshimizu20/ChessAI_nonzero)
