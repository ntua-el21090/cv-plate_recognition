import torch
import torch.nn as nn

import torchvision.models as models

import torchvision.models as models

class CRNN(nn.Module):
    def __init__(self, n_classes):
        super(CRNN, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-3])  

        self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.cnn(x)
        print("After CNN:", x.shape)  

        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, w, c * h)
        print("After reshape for RNN:", x.shape)  

        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.permute(1, 0, 2) 