##################
# IMPORT MODULES #
##################
# SYS IMPORT
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
from utils.METRICS import metrics_dict
from utils import EARLY_STOPPING, FOLDING

##################
# TRAIN FUNCTION #
##################
def train(run_number, folds=5, project="AERIAL_CACTUS", model_name="RESNET18"):
    print(f"Starting run number {run_number}, training on project : {project} for {folds} folds with {model_name} model")
    # CONFIG
    config = getattr(importlib.import_module(f"projects.{project}.config"), "config")

    # INIT WANDB
    wandb.init(config=config, project=project, name=model_name + "_" + str(run_number))
    # CREATING FOLDS
    FOLDING.create_splits(input_path=config.main.TRAIN_FILE,
                         output_path=config.main.FOLD_FILE,
                         n_folds=folds,
                         split_size=config.main.SPLIT_SIZE,
                         target=config.main.TARGET_VAR)

    # LOADING DATA FILE & TOKENIZER
    df = pd.read_csv(config.main.FOLD_FILE)
    # METRIC
    metric_selected = metrics_dict[config.train.METRIC]
    #AUGMENTATIONS
    Augmentations = getattr(importlib.import_module(f"projects.{project}.augment"), "Augmentations")

    # FOLD LOOP
    for fold in range(max(folds, 1)):
        print(f"Starting training for fold : {fold}")
        # CREATING TRAINING AND VALIDATION SETS
        df_train = df[df.split != fold].reset_index(drop=True)
        df_valid = df[df.split == fold].reset_index(drop=True)

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
        es = EARLY_STOPPING.EarlyStopping(patience=2, mode="max")
        # CREATE TRAINER
        trainer = TRAINER(model = model, 
                        task=config.main.TASK,
                        device = config.main.DEVICE,
                        optimizer = optimizer, 
                        criterion = criterion)

        wandb.watch(model, criterion=criterion, idx=fold)
        # START TRAINING FOR N EPOCHS
        for epoch in range(config.train.EPOCHS):
            print(f"Starting epoch number : {epoch}")
            # TRAINING PHASE
            print("Training the model...")
            trainer.training_step(train_loader)
            # VALIDATION PHASE
            print("Evaluating the model...")
            val_loss, metric_value = trainer.eval_step(valid_loader, metric_selected)
            scheduler.step(val_loss)
            # METRICS
            print(f"Validation {config.train.METRIC} = {metric_value}")
            
            wandb.log({
                    f"Validation Loss for fold {fold}" : val_loss, 
                    f"Validation {metric_selected.__name__} for fold {fold}" : metric_value
                })
            #SAVING CHECKPOINTS
            Path(os.path.join(config.main.PROJECT_PATH, "model_output/")).mkdir(parents=True, exist_ok=True)
            es(
                metric_value, 
                model, 
                model_path=os.path.join(config.main.PROJECT_PATH, "model_output/", f"model_{model_name}_{str(datetime.date.today().isoformat())}_{run_number}_{fold}.bin")
            )
            if es.early_stop:
                print("Early Stopping")
                break
            gc.collect()

        # IF WE GO FOR A TRAIN - VALID SPLIT WE TRAIN ONE MODEL ONLY (folds=0 or 1)
        if folds < 2:
            break

##########
# PARSER #
##########
parser = argparse.ArgumentParser()
parser.add_argument("--run_number", type=int)
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--project", type=str, default="AERIAL_CACTUS")
parser.add_argument("--model_name", type=str, default="RESNET18")

args = parser.parse_args()
##################
# START TRAINING #
##################
if __name__ == "__main__":
    print("Training start...")
    train(
        run_number=args.run_number,
        folds=args.folds,
        project=args.project,
        model_name=args.model_name    
        )
