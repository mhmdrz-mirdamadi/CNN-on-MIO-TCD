import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from copy import deepcopy


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

image_size = 32
num_classes = 5

transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

resnet18 = models.resnet18(weights=models.resnet.ResNet18_Weights.IMAGENET1K_V1)

class PreTrainedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18 = deepcopy(resnet18)
        in_features = resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512),
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=32, out_features=num_classes)
        )
        
    def forward(self, x):
        return self.resnet18(x)
    
    def freeze(self):
        for param in self.resnet18.parameters():
            param.requires_grad = False
        for param in self.resnet18.fc.parameters():
            param.requires_grad = True
        
    def unfreeze(self):
        for param in self.resnet18.parameters():
            param.requires_grad = True
