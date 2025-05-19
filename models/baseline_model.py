import torch
import torch.nn as nn

import torchvision.models as models

import torchvision.models as models

class CRNN(nn.Module):
    def __init__(self, n_classes):
        super(CRNN, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-3])  # output: [B, 256, H/8, W/8]

        self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.cnn(x)
        print("After CNN:", x.shape)  # Debug: [B, C, H, W]

        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, w, c * h)
        print("After reshape for RNN:", x.shape)  # 