import torch.nn as nn
from torchvision import models

class GenderNet(nn.Module):
    def __init__(self):
        super(GenderNet, self).__init__()
        self.base_model = models.efficientnet_b0()
        self.base_model.classifier = nn.Linear(1280, 1280)

        self.gender_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True), 
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        gender = self.gender_head(x)
        return gender
