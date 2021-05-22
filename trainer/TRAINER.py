##################
# IMPORT MODULES #
##################
import torch
from tqdm import tqdm
from utils.AVERAGE_METER import AverageMeter
from utils.METRICS import metrics_dict
import warnings
warnings.filterwarnings("ignore")
#################
# TRAINER CLASS #
#################
class TRAINER:
    '''
    trn_function train the model for one epoch
    eval_function evaluate the current model on validation data and output current loss and other evaluation metric
    '''
    def __init__(self, model, optimizer, device, criterion, task):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion
        self.task = task
    #################
    # TRAINING STEP #
    #################
    def training_step(self, data_loader):
        # LOSS AVERAGE
        losses = AverageMeter()
        # MODEL TO TRAIN MODE
        self.model.train()
        # TRAINING LOOP
        tk0 = tqdm(data_loader, total=len(data_loader))
        for _, data in enumerate(tk0):
            # LOADING IMAGES & LABELS
            images = data["images"].to(self.device)
            labels = data["labels"].to(self.device)
            # RESET GRADIENTS
            self.model.zero_grad()
            # CALCULATE LOSS
            output = self.model(images)
            loss = self.criterion(output, labels)
            # CALCULATE GRADIENTS
            loss.backward()
            self.optimizer.step()
            # UPDATE LOSS
            losses.update(loss.item(), images.size(0))
            tk0.set_postfix(loss=losses.avg)

    ###################
    # VALIDATION STEP #
    ###################
    def eval_step(self, data_loader, metric):
        # LOSS & METRIC AVERAGE
        losses = AverageMeter()
        metrics_avg = AverageMeter()
        # MODEL TO EVAL MODE
        self.model.eval()
        # VALIDATION LOOP
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader))
            for _, data in enumerate(tk0):
                # LOADING IMAGES & LABELS
                images = data["images"].to(self.device)
                labels = data["labels"].to(self.device)

                # CALCULATE LOSS & METRICS
                output = self.model(images)
                loss = self.criterion(output, labels)

                if self.task == "CLASSIFICATION":
                    output = output.argmax(axis=1)
                output = output.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                metric_value = metric(labels, output)

                losses.update(loss.item(), images.size(0))
                metrics_avg.update(metric_value.item(), images.size(0))

                tk0.set_postfix(loss=losses.avg)
        print(f"Validation Loss = {losses.avg}")
        return loss, metrics_avg.avg
