import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
from torchvision import models

class AgeGenderNet(nn.Module):
    def __init__(self):
        super(AgeGenderNet, self).__init__()
        self.base_model = models.efficientnet_b0()
        self.base_model.classifier = nn.Linear(1280, 1280)

        self.gender_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True), 
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self.age_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        gender = self.gender_head(x)
        age = self.age_head(x)
        age = torch.clamp(age, 0.05, 1)
        return gender, age

