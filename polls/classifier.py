import torch
from torchvision.modesl import resnet34

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet34(pretrained=True)
        self.model.fc = nn.Linear(512, 18)
        self.fc = nn.Sequential( nn.Linear(256,18), 
                                nn.ReLU(),
                                nn.Dropout(0.1),
        )

    def forward(self, x):
        x = self.model(x)
        return x