# ChessAI_alphazero

## Description
This is a chess AI that uses the AlphaZero algorithm. It is written in Python and uses PyTorch for the neural network. The neural network is first trained using supervised learning on a dataset of chess games. Then, the neural network is trained using reinforcement learning by playing against itself. The neural network is then used to play against a human player.

## Current Status
Although the supervised learning portion has been completed over 1,000,000 iterations, due to the very expensive nature of the self-play portion, the reinforcement learning portion has not been completed. The current version of the neural network, combined with the MCTS algorithm reaches around 1000 ELO.

## Similar Projects
- [Chess Engine using the pure AlphaGo algorithm](https://github.com/yshimizu20/ChessAI_nonzero)
