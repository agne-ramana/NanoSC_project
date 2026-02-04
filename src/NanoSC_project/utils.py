import torch


def filter_and_prepare(dataset):
    # Access the data (N, 28, 28) and targets (N)
    x = dataset.data.float() / 255.0  # Normalize to image_index-1
    y = dataset.targets
    
    
    # 3. Relabel: '0','2' -> 0 (Class 0) and '1' -> 1 (Class 1)
    # We use a trick: (label == 1) returns True/False, convert to float
    # 3. Relabel
    y_labels = torch.tensor([     [1.,0,0,0,0,0,0,0,0,0] if y0 == 0 
                             else [0,1.,0,0,0,0,0,0,0,0] if y0 == 1 
                             else [0,0,1.,0,0,0,0,0,0,0] if y0 == 2
                             else [0,0,0,1.,0,0,0,0,0,0] if y0 == 3
                             else [0,0,0,0,1.,0,0,0,0,0] if y0 == 4
                             else [0,0,0,0,0,1.,0,0,0,0] if y0 == 5
                             else [0,0,0,0,0,0,1.,0,0,0] if y0 == 6
                             else [0,0,0,0,0,0,0,1.,0,0] if y0 == 7
                             else [0,0,0,0,0,0,0,0,1.,0] if y0 == 8
                             else [0,0,0,0,0,0,0,0,0,1.]
                             for y0 in y])#.unsqueeze(1)
    
    # Add Channel Dimension for CNN: (N, 28, 28) -> (N, 1, 28, 28)
    # PyTorch uses (Batch, Channel, Height, Width) unlike Keras' (B, H, W, C)
    x = x.unsqueeze(1)
    
    return x, y_labels