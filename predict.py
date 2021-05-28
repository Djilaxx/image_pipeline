##################
# IMPORT MODULES #
##################
# SYS IMPORT
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import os
import glob
import inspect
import importlib
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ML IMPORT
# MY OWN MODULES
from datasets.IMAGE_DATASET import IMAGE_DATASET
from trainer.TRAINER import TRAINER

def predict(project="AERIAL_CACTUS", model_name="RESNET18"):
    print(f"Predictions on project : {project} with {model_name} model")
    # CONFIG
    config = getattr(importlib.import_module(f"projects.{project}.config"), "config")

    # LOADING DATA FILE
    if config.main.TEST_FILE is not None:
        df_test = pd.read_csv(config.main.TEST_FILE)
    else:
        image_path = []
        for filename in glob.glob(os.path.join(config.main.TEST_PATH, "*.jpg")):
            filename = filename.split("\\", -1)[-1]
            image_path.append(filename)
        df_test = pd.DataFrame({config.main.IMAGE_ID : image_path})
    #AUGMENTATIONS
    Augmentations = getattr(importlib.import_module(f"projects.{project}.augment"), "Augmentations")
    
    # LOADING MODEL
    for name, cls in inspect.getmembers(importlib.import_module(f"models.{model_name}"), inspect.isclass):
        if name == model_name:
            model = cls(n_class=config.main.N_CLASS, pretrain=False)

    model.to(config.main.DEVICE)
    ########################
    # CREATING DATALOADERS #
    ########################
    # TEST IDs & LABELS
    test_img = df_test[config.main.IMAGE_ID].values.tolist()
    test_img = [os.path.join(config.main.TEST_PATH, os.path.splitext(i)[0] + config.main.IMAGE_EXT) for i in test_img]

    # VALIDATION DATASET
    test_ds = IMAGE_DATASET(
        image_path=test_img,
        label=None,
        resize=config.train.IMAGE_SIZE,
        transforms=Augmentations["test"]    
    )
    # VALIDATION DATALOADER
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config.train.VALID_BS,
        shuffle=False,
        num_workers=0
    )
    
    # PREDICTION LOOP
    final_preds = None
    for fold in range(config.main.PREDICTION_FOLD_NUMBER):
        print(f"Starting predictions for fold  : {fold}")
        # LOAD MODEL WITH FOLD WEIGHTS
        weights = torch.load(config.main.WEIGHTS_PATH.rsplit("_", 1)[0] + "_" + str(fold) + ".bin")
        model.load_state_dict(weights)
        model.eval()

        trainer = TRAINER(model=model, task=config.main.TASK, device=config.main.DEVICE, optimizer=None, criterion=None)
        # DATA LOADER LOOP
        predictions = trainer.test_step(data_loader=test_loader, n_class=config.main.N_CLASS)
        predictions = np.vstack(predictions)
        predictions = predictions.reshape(len(df_test), 1, config.main.N_CLASS)
        temp_preds = None
        for p in predictions:
            if temp_preds is None:
                temp_preds = p
            else:
                temp_preds = np.vstack((temp_preds, p))
        if final_preds is None:
            final_preds = temp_preds
        else:
            final_preds += temp_preds

    final_preds /= config.main.PREDICTION_FOLD_NUMBER
    if config.main.PREDICT_PROBA is False:
        final_preds = final_preds.argmax(axis=1)
    # CONDITIONAL SUBMISSION FILE DEPENDING IF WE HAVE A TEST FILE OR NOT
    test_final_data = {config.main.IMAGE_ID : df_test[config.main.IMAGE_ID].values.tolist(), config.main.TARGET_VAR : final_preds}
    test_df = pd.DataFrame(data=test_final_data, index=None)
    test_df.to_csv(os.path.join(config.main.PROJECT_PATH, "preds.csv"))

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
