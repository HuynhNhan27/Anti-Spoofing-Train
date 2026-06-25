import torch
import torch.nn as nn
import torchvision.models as models
from src.models.minifasv2.model import FTGenerator

class ResNet18Fourier(nn.Module):
    """
    ResNet-18 model combined with a Fourier Transform auxiliary supervision head.
    Extracts intermediate features from layer2 (128 channels) to feed into the FT branch.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet18Fourier, self).__init__()
        
        # Load backbone ResNet-18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Replace classification head
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
        # Fourier auxiliary head
        # The output of ResNet-18's layer2 has 128 channels, which matches
        # the default in_channels expected by FTGenerator.
        self.FTGenerator = FTGenerator(in_channels=128, out_channels=1)

    def forward(self, x):
        # Extract features block by block from ResNet-18
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        
        # Fourier auxiliary branch is active only during training
        if self.training:
            fourier_transform = self.FTGenerator(x)
            
        # Complete the rest of the ResNet-18 backbone
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        classifier_output = self.resnet.fc(x)
        
        if self.training:
            return classifier_output, fourier_transform
        else:
            return classifier_output
