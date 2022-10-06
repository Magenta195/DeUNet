
import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self, num_cls):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(7 * 7 * 64, num_cls, bias=True)

    def feature_forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)

        return output

    def classifier_forward(self, x):
        output = x.view(x.size(0), -1)
        output = self.fc(output)

        return output

    def forward(self, x):
        output = self.feature_forward(x)
        output = self.classifier_forward(output)

        return output