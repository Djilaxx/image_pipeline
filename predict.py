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

def predict(project="AERIAL_CACTUS", model_name="RESNET18", run_note="test"):
    """
    Predict on a test dataset using a trained model

    Parameters
    ----------
    project: str
        the name of the project you wish to work on - must be the name of the project folder under projects/
    model_name: str
        the name of the model you wish to use to predict - name of the python file under models/
    run_note: str
        An string note for your current run
    Returns
    -------
    Create a csv file containing the model predictions
    """
    
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

    # TEST DATASET
    test_ds = IMAGE_DATASET(
        image_path=test_img,
        label=None,
        resize=config.train.IMAGE_SIZE,
        transforms=Augmentations["test"]    
    )
    # TEST DATALOADER
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config.train.VALID_BS,
        shuffle=False,
        num_workers=0
    )
    
    # PREDICTION LOOP
    final_preds = []
    for fold in range(max(config.main.FOLD_NUMBER, 1)):
        print(f"Starting predictions for fold  : {fold}")
        # LOAD MODEL WITH FOLD WEIGHTS
        model_weights = torch.load(os.path.join(config.main.PROJECT_PATH, "model_output/", f"model_{model_name}_{run_note}_{fold}.bin"))
        model.load_state_dict(model_weights)
        model.eval()

        trainer = TRAINER(model=model, task=config.main.TASK, device=config.main.DEVICE, n_class=config.main.N_CLASS)
        # DATA LOADER LOOP
        predictions = trainer.test_step(data_loader=test_loader)
        final_preds.append(predictions)


        if config.main.SPLIT is True:
            break
    
    final_preds = np.mean(np.column_stack(final_preds), axis=1)
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
parser.add_argument("--run_note", type=str, default="test")

args = parser.parse_args()
##################
# START TRAINING #
##################
if __name__ == "__main__":
    print("Training start...")
    predict(
        project=args.project,
        model_name=args.model_name,
        run_note=args.run_note
    )
