import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.gan_model import ConditionalGAN
from dataset import BackgroundDataset  # Custom dataset for GAN training

# Load dataset
dataset = BackgroundDataset("data/backgrounds")  # Ensure dataset folder exists
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize model, loss function, and optimizer
cgan = ConditionalGAN()
criterion = torch.nn.MSELoss()  # Placeholder loss function
optimizer = optim.Adam(cgan.generator.parameters(), lr=0.0002)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for foregrounds, alphas, real_backgrounds in dataloader:
        optimizer.zero_grad()
        
        # Generate fake backgrounds
        fake_backgrounds = cgan.generate(foregrounds, alphas)
        
        # Compute loss (compare fake and real backgrounds)
        loss = criterion(fake_backgrounds, real_backgrounds)
        
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save trained model
torch.save(cgan.generator.state_dict(), "checkpoints/cgan.pth")
print("Conditional GAN model saved at checkpoints/cgan.pth")
