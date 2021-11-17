import torch
import torch.nn as nn
from torchvision import models

class RESNET18(nn.Module):
    """
    nn.Module of a resnet18 model

    Parameters:
    -----------
    n_class: int
        number of class in the dataset (1 for regression tasks)
    pretrain: Boolean
        using pretrained weights (on ImageNet) for finetuning or random weights (full training)
    """
    def __init__(self, n_class=2, pretrain=True):
        super(RESNET18, self).__init__()

        self.base_model = models.resnet18(pretrained=pretrain)
        in_features = self.base_model.fc.out_features
        #self.nb_features = self.base_model.fc.in_features
        self.l0 = nn.Linear(in_features, n_class)

    def forward(self, image):
        x = self.base_model(image)
        out = self.l0(x)
        return out
