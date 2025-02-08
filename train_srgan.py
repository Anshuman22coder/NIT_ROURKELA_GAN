import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.srgan import SRGAN
from dataset import SuperResolutionDataset  # Custom dataset for SRGAN
from dataset import SuperResolutionDataset
from dataset import SuperResolutionDataset

# Load dataset
dataset = SuperResolutionDataset("data/superres")  # Ensure dataset folder exists
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize model, loss function, and optimizer
srgan = SRGAN()
criterion = torch.nn.MSELoss()  # Placeholder loss function
optimizer = optim.Adam(srgan.parameters(), lr=0.0001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for low_res, high_res in dataloader:
        optimizer.zero_grad()
        
        # Generate super-resolution image
        sr_image = srgan.enhance(low_res)
        
        # Compute loss (compare SR image with high-res ground truth)
        loss = criterion(sr_image, high_res)
        
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save trained model
torch.save(srgan.state_dict(), "checkpoints/srgan.pth")
print("SRGAN model saved at checkpoints/srgan.pth")
