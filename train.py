##################
# IMPORT MODULES #
##################
# SYS IMPORT
from collections import defaultdict
import torch
from pathlib import Path
import pandas as pd
import gc
import os
import inspect
import importlib
import argparse
import datetime
import wandb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ML IMPORT
# MY OWN MODULES
from datasets.IMAGE_DATASET import IMAGE_DATASET
from trainer.TRAINER import TRAINER
from utils import early_stopping, folding
from utils.memory_usage import reduce_memory_usage

##################
# TRAIN FUNCTION #
##################
def train(project="AERIAL_CACTUS", model_name="RESNET18", run_note="test"):
    """
    Train, validate, and log results of a model on a specified dataset

    Parameters
    ----------
    project: str
        the name of the project you wish to work on - must be the name of the project folder under projects/
    model_name: str
        the name of the model you wish to train - name of the python file under models/
    run_note: str
        An string note for your current run

    Returns
    -------
    save trained model under projects/model_saved/
    print training results and log to wandb
    """

    # LOADING PROJECT CONFIG
    config = getattr(importlib.import_module(f"projects.{project}.config"), "config")
    if config.main.SPLIT is True:
        print(f"Starting run {run_note}, training on project : {project} with {model_name} model")
    else:
        print(f"Starting run {run_note}, training on project : {project} for {config.main.FOLD_NUMBER} folds with {model_name} model")
    complete_name = f"{model_name}_{config.main.TASK}"
    # RECORD RUNS USING WANDB TOOL
    wandb.init(config=config, project=project,name=complete_name + "_" + str(run_note))
    # CREATING FOLDS
    df = pd.read_csv(config.main.TRAIN_FILE)
    df = reduce_memory_usage(df, verbose=True)
    df = folding.create_splits(df=df,
                               task=config.main.TASK,
                               n_folds=config.main.FOLD_NUMBER,
                               split=config.main.SPLIT,
                               split_size=config.main.SPLIT_SIZE,
                               target=config.main.TARGET_VAR)
    # METRIC
    #metric_selected = metrics_dict[config.train.METRIC]
    #AUGMENTATIONS
    Augmentations = getattr(importlib.import_module(f"projects.{project}.augment"), "Augmentations")

    # FOLD LOOP
    for fold in range(max(config.main.FOLD_NUMBER, 1)):
        print("Starting training...") if config.main.SPLIT is True else print(f"Starting training for fold {fold}")
        # CREATING TRAINING AND VALIDATION SETS
        df_train = df[df.splits != fold].reset_index(drop=True)
        df_valid = df[df.splits == fold].reset_index(drop=True)
        # LOADING MODEL
        for name, cls in inspect.getmembers(importlib.import_module(f"models.{model_name}"), inspect.isclass):
            if name == model_name:
                model = cls(n_class = config.main.N_CLASS, pretrain = config.train.PRETRAINED)

        model.to(config.main.DEVICE)
        ########################
        # CREATING DATALOADERS #
        ########################
        # TRAINING IDs & LABELS
        train_img = df_train[config.main.IMAGE_ID].values.tolist()
        train_img = [os.path.join(config.main.TRAIN_PATH, os.path.splitext(i)[0] + config.main.IMAGE_EXT) for i in train_img]
        train_labels = df_train[config.main.TARGET_VAR].values
        # VALIDATION IDs & LABELS
        valid_img = df_valid[config.main.IMAGE_ID].values.tolist()
        valid_img = [os.path.join(config.main.TRAIN_PATH, os.path.splitext(i)[0] + config.main.IMAGE_EXT) for i in valid_img]
        valid_labels = df_valid[config.main.TARGET_VAR].values

        # TRAINING DATASET
        train_ds = IMAGE_DATASET(
            image_path=train_img,
            label=train_labels,
            resize=config.train.IMAGE_SIZE,
            transforms=Augmentations["train"]
        )
        # TRAINING DATALOADER
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=config.train.TRAIN_BS,
            shuffle=True,
            num_workers=0
        )
        # VALIDATION DATASET
        valid_ds = IMAGE_DATASET(
            image_path=valid_img,
            label=valid_labels,
            resize=config.train.IMAGE_SIZE,
            transforms=Augmentations["valid"]        
        )
        # VALIDATION DATALOADER
        valid_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=config.train.VALID_BS,
            shuffle=True,
            num_workers=0
        )
        # IMPORT LOSS FUNCTION
        loss_function = getattr(importlib.import_module(f"loss.{config.train.LOSS}"), "loss_function")
        criterion = loss_function()
        # SET OPTIMIZER, SCHEDULER
        optimizer = torch.optim.Adam(model.parameters(), lr=config.train.LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        # SET EARLY STOPPING FUNCTION
        es = early_stopping.EarlyStopping(patience=2, mode="max")
        # CREATE TRAINER
        trainer = TRAINER(model = model, 
                        task=config.main.TASK,
                        device = config.main.DEVICE,
                        optimizer = optimizer, 
                        criterion = criterion,
                        n_class = config.main.N_CLASS)

        wandb.watch(model, criterion=criterion, idx=fold)
        # START TRAINING FOR N EPOCHS
        for epoch in range(config.train.EPOCHS):
            print(f"Starting epoch number : {epoch}")
            # TRAINING PHASE
            print("Training the model...")
            train_loss, train_metrics = trainer.training_step(train_loader)
            print(f"Training loss for epoch {epoch}: {train_loss}")
            wandb.log({f"Training loss": train_loss})
            for metric_name, metric_value in train_metrics.items():
                print(f"Training {metric_name} score for epoch {epoch}: {metric_value.avg}")
                wandb.log({f"Training {metric_name} score for fold {fold}": metric_value.avg})

            # VALIDATION PHASE
            print("Evaluating the model...")
            val_loss, valid_metrics = trainer.validation_step(valid_loader)
            print(f"Validation loss for epoch {epoch}: {val_loss}")
            wandb.log({f"Validation loss": val_loss})
            for metric_name, metric_value in valid_metrics.items():
                print(f"Validation {metric_name} score for epoch {epoch}: {metric_value.avg}")
                wandb.log({f"Validation {metric_name} score for fold {fold}": metric_value.avg})
            
            scheduler.step(val_loss)

            #SAVING CHECKPOINTS
            Path(os.path.join(config.main.PROJECT_PATH, "model_output/")).mkdir(parents=True, exist_ok=True)
            es(
                valid_metrics[config.train.METRIC].avg,
                model, 
                model_path=os.path.join(config.main.PROJECT_PATH, "model_output/", f"model_{model_name}_{run_note}_{fold}.bin")
            )
            if es.early_stop:
                print("Early Stopping")
                break
            gc.collect()

        # IF WE GO FOR A TRAIN - VALID SPLIT WE TRAIN ONE MODEL ONLY (folds=0 or 1)
        if config.main.SPLIT is True:
            break

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
    train(
        project=args.project,
        model_name=args.model_name,
        run_note=args.run_note
    )
