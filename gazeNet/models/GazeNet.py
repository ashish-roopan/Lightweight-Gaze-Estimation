import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
from torchvision import models

class GazeNet(nn.Module):
    def __init__(self, num_bins=90):
        super(GazeNet, self).__init__()
        block = models.resnet.BasicBlock
        self.base_model = models.efficientnet_b0(pretrained=True)
        self.base_model.classifier = nn.Linear(1280, 512 * block.expansion)
        # self.base_model = models.resnet50(pretrained=True)
        # self.base_model.fc = nn.Linear(2048, 512 * block.expansion)
        self.fc_yaw_gaze = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch_gaze = nn.Linear(512 * block.expansion, num_bins)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.base_model(x)
        # gaze
        x = self.dropout(x)
        pre_yaw_gaze =  self.fc_yaw_gaze(x)
        pre_pitch_gaze = self.fc_pitch_gaze(x)
        return pre_yaw_gaze, pre_pitch_gaze



