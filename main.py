import torch
import cv2
import numpy as np
from models.modnet import MODNet
from models.gan_model import ConditionalGAN

# Load MODNet
def load_modnet():
    modnet = MODNet()
    modnet.load_state_dict(torch.load("checkpoints/modnet.pth", map_location=torch.device('cpu')))
    modnet.eval()
    return modnet

# Process image
def process_image(image_path):
    image = cv2.imread(image_path)
    modnet = load_modnet()
    
    with torch.no_grad():
        image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        alpha = modnet(image_tensor).squeeze().numpy()
    
    foreground = image * alpha[:, :, None]
    cv2.imwrite("output.jpg", foreground * 255)

# Run the process
process_image("data/test/sample.jpg")
