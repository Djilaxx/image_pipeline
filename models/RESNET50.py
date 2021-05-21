import torch.nn as nn
from torchvision import models

class RESNET50(nn.Module):
    def __init__(self, n_class=2, pretrain=True):
        super(RESNET50, self).__init__()

        self.base_model = models.resnet50(pretrained=pretrain)
        in_features = self.base_model.fc.out_features
        #self.nb_features = self.base_model.fc.in_features
        self.l0 = nn.Linear(in_features, n_class)

    def forward(self, image):
        x = self.base_model(image)
        out = self.l0(x)
        return out
