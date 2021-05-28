from easydict import EasyDict as edict
import torch

config = edict()
########
# MAIN #
########
# main is the config section related to basic info on the project
# data repo, data format, folding etc... data preparation
config.main = edict()
config.main.PROJECT_PATH = "projects/MELANOMA/"
config.main.TRAIN_PATH = "data/melanoma/train/train/"
config.main.TEST_PATH = "data/melanoma/test/test/"
config.main.TRAIN_FILE = "data/melanoma/train_concat.csv"
config.main.TEST_FILE = None
config.main.FOLD_FILE = "data/melanoma/train_folds.csv"
config.main.FOLD_METHOD = "SKF"
config.main.TASK = "CLASSIFICATION"
config.main.TARGET_VAR = "target"
config.main.IMAGE_ID = "image_name"
config.main.IMAGE_EXT = ".jpg"
config.main.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
config.main.N_CLASS = 2
config.main.PREDICTION_FOLD_NUMBER = 5
config.main.WEIGHTS_PATH = "projects/MELANOMA/model_output/model_RESNET18_2021-05-25_0.bin"
config.main.PREDICT_PROBA = False

#######################
# TRAINING PARAMETERS #
#######################
config.train = edict()
config.train.IMAGE_SIZE = (128, 128)
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
