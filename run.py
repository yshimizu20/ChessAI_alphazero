import os
import torch

from chess_engine.self_play import self_play

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    # find the latest model
    model_path = "saved_models/current_best/"
    model_files = os.listdir(model_path)

    if len(model_files) == 0:
        latest_model = None
        start_epoch = 0
        end_epoch = 1000
    else:
        model_files.sort()
        latest_model = model_files[-1]
        latest_model = os.path.join(model_path, latest_model)
        start_epoch = int(latest_model.split("_")[-1].split(".")[0]) + 1
        end_epoch = start_epoch + 1000

    self_play(start_epoch, end_epoch)
