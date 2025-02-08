import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.modnet import MODNet
from dataset import MattingDataset

# Load dataset
dataset = MattingDataset("data/alphamatting")  # Ensure dataset folder exists
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize model, loss function, and optimizer
modnet = MODNet()
criterion = torch.nn.MSELoss()  # Loss function for alpha mask prediction
optimizer = optim.Adam(modnet.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for images, alpha in dataloader:
        optimizer.zero_grad()
        predicted_alpha = modnet(images)
        loss = criterion(predicted_alpha, alpha.unsqueeze(1))  # Ensure shape matches
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save trained model
torch.save(modnet.state_dict(), "checkpoints/modnet.pth")
print("Model saved at checkpoints/modnet.pth") 