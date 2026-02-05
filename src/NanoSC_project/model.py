import torch
import torch.nn as nn


class TwoLayerCNN(nn.Module):
    def __init__(self):
        super(TwoLayerCNN, self).__init__()

        # Activation, normalisation & dropout functions
        self.relu = nn.ReLU()
        self.bnorm32 = nn.BatchNorm2d(num_features=32)
        self.bnorm64 = nn.BatchNorm2d(num_features=64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(p=0.4)

        # Layer 1: Conv (1 input channel -> 32 output channels)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding='same')

        # Layer 2: Conv (32 -> 64)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        
        # Layer 3: Fully Connected
        self.fc1 = nn.Linear(12544, 64)

        # Layer 4: Fully Connected
        self.out = nn.Linear(64, 10) # Output 10 values
        self.softmax = nn.Softmax(dim=1) # Squash to 0-1
        
    def forward(self, x):
        # Layer 1
        x = self.relu(self.conv1(x))
        # Layer 2
        x = self.pool(self.relu(self.conv2(x)))
        # Dropout
        x = self.drop(x)
        
        # Flatten: (Batch, 64, 1, 1) -> (Batch, 64)
        x = x.view(x.size(0), -1) 
        
        # Dense Layers
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.softmax(self.out(x))
        return x

