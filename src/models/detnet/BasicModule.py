import os
import time
import torch
import torch.nn as nn

class BasicModule(torch.nn.Module):
    '''
    Wrapper around nn.Module providing helper save/load methods.
    '''
    def __init__(self, opt=None):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self)) # Default model name

    def load(self, path):
        '''
        Load model weights from path.
        '''
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        '''
        Save model weights. Defaults to 'classname_MMDD_HH:MM:SS.pth' in checkpoints.
        '''
        if name is None:
            os.makedirs('checkpoints', exist_ok=True)
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

class MydetNet59(BasicModule):
    def __init__(self, pretrained=False):
        super(MydetNet59, self).__init__()
        from src.models.detnet.detnet import detnet59
        model = detnet59(pretrained=pretrained)
        self.resnet_lay = nn.Sequential(*list(model.children())[:-2])
        self.conv1_lay = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
        self.relu1_lay = nn.ReLU(inplace=True)
        self.drop_lay = nn.Dropout2d(0.5)
        self.global_average = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_Linear_lay2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.resnet_lay(x)
        x = self.conv1_lay(x)
        x = self.relu1_lay(x)
        x = self.drop_lay(x)
        x = self.global_average(x)
        x = x.view(x.size(0), -1)
        x = self.fc_Linear_lay2(x)
        return x