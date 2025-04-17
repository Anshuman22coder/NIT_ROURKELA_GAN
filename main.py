import torch
import cv2
import numpy as np
from models.modnet import MODNet
from models.gan_model import ConditionalGAN
from torch.utils.data import DataLoader
from torchvision import models, transforms
from skimage.metrics import structural_similarity as ssim
from PIL import Image

# Load MODNet
def load_modnet(device):
    modnet = MODNet()
    modnet.load_state_dict(torch.load("checkpoints/modnet.pth", map_location=device))
    modnet.to(device)
    modnet.eval()
    return modnet

# Load Conditional GAN
def load_conditional_gan(device):
    gan = ConditionalGAN()
    gan.load_state_dict(torch.load("checkpoints/conditional_gan.pth", map_location=device))
    gan.to(device)
    gan.eval()
    return gan

# Generate an image using Conditional GAN
def generate_gan_image(gan, condition, device):
    with torch.no_grad():
        generated_image = gan(condition)
        return generated_image

# Inception Score Calculation
def calculate_inception_score(images, device, batch_size=32, splits=10):
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.to(device)
    inception_model.eval()
    
    def get_predicted_probs(images):
        images = images.to(device)
        with torch.no_grad():
            preds = inception_model(images)
            return torch.nn.functional.softmax(preds, dim=1)
    
    # Preprocessing the images
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),  # Inception v3 requires 299x299 input
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image_list = [transform(img).unsqueeze(0) for img in images]
    image_tensor = torch.cat(image_list)
    predicted_probs = get_predicted_probs(image_tensor)
    
    # Compute the Inception Score
    scores = []
    for i in range(splits):
        part = predicted_probs[i * (len(predicted_probs) // splits): (i + 1) * (len(predicted_probs) // splits)]
        py = part.mean(0)
        scores.append(torch.distributions.kl_divergence(part, py).mean().item())
    
    return np.exp(np.mean(scores))

# SSIM Calculation
def calculate_ssim(img1, img2):
    # Convert to grayscale for SSIM comparison
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM
    return ssim(img1_gray, img2_gray)

# Process image and combine with GAN
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image at {image_path} could not be loaded.")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    modnet = load_modnet(device)
    gan = load_conditional_gan(device)
    
    # Foreground extraction with MODNet
    with torch.no_grad():
        image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image_tensor = image_tensor.to(device)
        alpha = modnet(image_tensor).squeeze().cpu().numpy()
    
    foreground = image * alpha[:, :, None]
    
    # Generate image using Conditional GAN (use some condition here, e.g., random tensor or an image)
    condition = torch.randn(1, 100, device=device)  # Example condition (e.g., a random latent vector)
    gan_image = generate_gan_image(gan, condition, device)

    # Post-process the generated image
    gan_image = gan_image.squeeze().cpu().numpy()
    gan_image = np.clip(gan_image, 0, 1) * 255  # Scale to [0, 255] range
    gan_image = gan_image.astype(np.uint8)

    # Combine foreground and GAN image (here we are just overlaying them for simplicity)
    combined_image = foreground * 0.5 + gan_image * 0.5  # Blending
    combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)

    # Save the result
    cv2.imwrite("output.jpg", combined_image)

    # Inception Score (Assuming you have multiple generated images in a list)
    images = [combined_image]  # Replace with actual list of generated images
    inception_score = calculate_inception_score(images, device)
    print(f"Inception Score: {inception_score}")

    # SSIM (Compare with original image or another reference image)
    original_image = cv2.imread(image_path)  # Original image
    ssim_value = calculate_ssim(original_image, combined_image)
    print(f"SSIM: {ssim_value}")

# Run the process
process_image("data/test/sample.jpg")
