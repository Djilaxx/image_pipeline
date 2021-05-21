from easydict import EasyDict as edict
import torch

config = edict()
########
# MAIN #
########
# main is the config section related to basic info on the project
# data repo, data format, folding etc... data preparation
config.main = edict()
config.main.PROJECT_PATH = "projects/AERIAL_CACTUS/"
config.main.TRAIN_PATH = "data/aerial-cactus-identification/train/"
config.main.TEST_PATH = "data/aerial-cactus-identification/test/"
config.main.TRAIN_FILE = "data/aerial-cactus-identification/train.csv"
config.main.TEST_FILE = None
config.main.FOLD_FILE = "data/aerial-cactus-identification/train_folds.csv"
config.main.FOLD_METHOD = "SKF"  
config.main.TARGET_VAR = "has_cactus"  
config.main.IMAGE_ID = "id"
config.main.IMAGE_EXT = ".jpg"
config.main.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
config.main.N_CLASS = 2
#######################
# TRAINING PARAMETERS #
#######################
config.train = edict()
config.train.IMAGE_SIZE = (32, 32)
config.train.PRETRAINED = True
config.train.EPOCHS = 1
config.train.TRAIN_BS = 16
config.train.VALID_BS = 8
config.train.ES = 50
config.train.LR = 1e-4
config.train.LOSS = "CROSS_ENTROPY"
config.train.METRIC = "ACCURACY"
####################
# MODEL PARAMETERS #
####################
config.model = edict()
