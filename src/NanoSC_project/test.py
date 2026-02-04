import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from NanoSC_project import filter_and_prepare, TwoLayerCNN


# Check for GPU, otherwise use CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")


# Load MNIST Data (Train and Test)
print('Loading MNIST dataset')
train_data_raw = datasets.MNIST(root='../../data', train=True, download=True, transform=None)
test_data_raw = datasets.MNIST(root='../../data', train=False, download=True, transform=None)
print('Loading complete')


# Prepare Train and Test tensors
x_train, y_train = filter_and_prepare(train_data_raw)
x_test, y_test = filter_and_prepare(test_data_raw)


# Create DataLoaders (The iterator that feeds the model)
train_ds = TensorDataset(x_train, y_train)
test_ds = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)


# Load model
model = TwoLayerCNN().to(device)
model.load_state_dict(torch.load('../../models/model1.pt', weights_only=True))
model.eval()

print('Testing model')
# Test model
correct = 0
total = 0

# Turn off Gradient Tracking
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total * 10 #* 100
print(f"Final Test Accuracy: {accuracy:.2f}%")