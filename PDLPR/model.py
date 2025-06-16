import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CRNN_CTC(nn.Module):
    
    def __init__(self, num_chars: int, seq_len: int, img_h: int, img_w: int):
        super().__init__()
       
        self.stn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7, padding=3),
            nn.MaxPool2d(2), nn.ReLU(inplace=True),
            nn.Conv2d(8, 10, kernel_size=5, padding=2),
            nn.MaxPool2d(2), nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(10 * (img_h // 4) * (img_w // 4), 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6)
        )
        
        self.stn[-1].weight.data.zero_()
        self.stn[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

  
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  

       
        self.proj = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

       
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 512 // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // 16, 512, kernel_size=1),
            nn.Sigmoid(),
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, seq_len))

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.classifier = nn.Linear(512 * 2, num_chars + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        theta = self.stn(x).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        f = self.backbone(x)
        p = self.proj(f)
        w = self.se(p)
        p = p * w

        p = self.pool(p)

        t = p.squeeze(2).permute(0, 2, 1)

        out, _ = self.lstm(t)   

        logits = self.classifier(out)  
        return logits
