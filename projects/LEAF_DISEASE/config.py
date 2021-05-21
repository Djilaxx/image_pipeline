from easydict import EasyDict as edict
import torch

config = edict()
########
# MAIN #
########
# main is the config section related to basic info on the project
# data repo, data format, folding etc... data preparation
config.main = edict()
config.main.PROJECT_PATH = "projects/LEAF_DISEASE/"
config.main.TRAIN_PATH = "data/leaf_disease/train_images/"
config.main.TEST_PATH = "data/leaf_disease/test_images/"
config.main.TRAIN_FILE = "data/leaf_disease/train.csv"
config.main.TEST_FILE = None
config.main.FOLD_FILE = "data/leaf_disease/train_folds.csv"
config.main.FOLD_METHOD = "SKF"
config.main.TARGET_VAR = "label"
config.main.IMAGE_ID = "image_id"
config.main.IMAGE_EXT = ".jpg"
config.main.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
config.main.N_CLASS = 5
#######################
# TRAINING PARAMETERS #
#######################
config.train = edict()
config.train.IMAGE_SIZE = (256, 256)
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
