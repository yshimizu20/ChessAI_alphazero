import os

if __name__ == "__main__":
    # find the latest model
    model_path = "saved_models"
    model_files = os.listdir(model_path)

    if len(model_files) == 0:
        latest_model = None
        start_epoch = 0
    else:
        model_files.sort()
        latest_model = model_files[-1]
        latest_model = os.path.join(model_path, latest_model)
        start_epoch = int(latest_model.split("_")[-1].split(".")[0]) + 1
