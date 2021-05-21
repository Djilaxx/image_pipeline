##################
# IMPORT MODULES #
##################
# SYS IMPORT
from tqdm import tqdm
import torch
from pathlib import Path
import pandas as pd
import gc
import os
import inspect
import importlib
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ML IMPORT
# MY OWN MODULES
from datasets.IMAGE_DATASET import IMAGE_DATASET

def predict(project="AERIAL_CACTUS", model_name="RESNET18"):
    print(f"Predictions on project : {project} with {model_name} model")
    # CONFIG
    config = getattr(importlib.import_module(f"projects.{project}.config"), "config")

    # LOADING DATA FILE
    df_test = pd.read_csv(config.main.TEST_FILE)
    #AUGMENTATIONS
    Augmentations = getattr(importlib.import_module(f"projects.{project}.augment"), "Augmentations")
    
    # LOADING MODEL
    for name, cls in inspect.getmembers(importlib.import_module(f"models.{model_name}"), inspect.isclass):
        if name == model_name:
            model = cls(n_class=config.main.N_CLASS, pretrain=False)

    model.to(config.main.DEVICE)

    # LOADING TRAINED WEIGHTS

    ########################
    # CREATING DATALOADERS #
    ########################
    # TEST IDs & LABELS
    test_img = df_test[config.main.IMAGE_ID].values.tolist()
    test_img = [os.path.join(config.main.TRAIN_PATH, os.path.splitext(i)[0] + config.main.IMAGE_EXT) for i in test_img]

    # VALIDATION DATASET
    test_ds = IMAGE_DATASET(
        image_path=test_img,
        label=None,
        resize=config.train.IMAGE_SIZE,
        transforms=Augmentations["test"],
        test=True
    )
    # VALIDATION DATALOADER
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config.train.VALID_BS,
        shuffle=True,
        num_workers=0
    )
    
    # PREDICTION LOOP
    model_preds = []
    for fold in range(number_of_fold):
        # LOAD MODEL WITH FOLD WEIGHTS

        with torch.no_grad():
            tk0 = tqdm(test_loader, total=len(test_loader))
            for _, data in enumerate(tk0):
                # LOADING IMAGES
                images = data["images"].to(config.main.DEVICE)

                # PREDICT
                output = model(images)
                if config.main.TASK == "CLASSIFICATION":
                    output = output.argmax(axis=1)
                output = output.cpu().detach().numpy()
                model_preds.append()

##########
# PARSER #
##########
parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="AERIAL_CACTUS")
parser.add_argument("--model_name", type=str, default="RESNET18")

args = parser.parse_args()
##################
# START TRAINING #
##################
if __name__ == "__main__":
    print("Training start...")
    predict(
        project=args.project,
        model_name=args.model_name    
        )
