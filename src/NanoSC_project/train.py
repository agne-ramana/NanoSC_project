import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from NanoSC_project import filter_and_prepare, TwoLayerCNN

# Set seed for reproducibility
torch.manual_seed(42)

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

model = TwoLayerCNN().to(device)

# print(summary(model, input_size=(1, 28, 28), device=device))


# Train model

# Loss and Optimizer
criterion = nn.CrossEntropyLoss() # Mean absolute error
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
train_losses = []

print("Starting Training...")

for epoch in range(epochs):
    running_loss = 0.0
    
    for images, labels in train_loader:
        # Move data to GPU/MPS if available
        images, labels = images.to(device), labels.to(device)
        
        # 1. Zero Gradients (Clear previous step's history)
        optimizer.zero_grad()
        
        # 2. Forward Pass (Make prediction)
        outputs = model(images)
        
        # 3. Calculate Loss (Compare prediction to ground truth)
        loss = criterion(outputs, labels)
        
        # 4. Backward Pass (Calculate gradients via Backprop)
        loss.backward()
        
        # 5. Optimizer Step (Update weights)
        optimizer.step()
        
        running_loss += loss.item()
        
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

print("Training Complete!")


torch.save(model.state_dict(), '../../models/model1.pt')