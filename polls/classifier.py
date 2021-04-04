import torch
from torchvision.modesl import resnet34

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet34(pretrained=True)
        #         self.model = timm.create_model('inception_resnet_v2', pretrained = True)
        self.model.fc = nn.Linear(512, 18)

    #         self.model.classif = nn.Linear(1536, 18)

    def forward(self, x):
        x = self.model(x)
        return x