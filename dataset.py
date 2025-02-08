import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MattingDataset(Dataset):
    def __init__(self, root_dir):
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.image_files = sorted(os.listdir(self.image_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale

        return self.transform(image), self.transform(mask)

class SRGANDataset(Dataset):
    def __init__(self, root_dir):
        self.lr_dir = os.path.join(root_dir, "low_res")
        self.hr_dir = os.path.join(root_dir, "high_res")
        self.lr_files = sorted(os.listdir(self.lr_dir))
        self.hr_files = sorted(os.listdir(self.hr_dir))

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])

        lr_image = Image.open(lr_path).convert("RGB")
        hr_image = Image.open(hr_path).convert("RGB")

        return self.transform(lr_image), self.transform(hr_image)


class SuperResolutionDataset(Dataset):
    def __init__(self, hr_folder):
        self.hr_folder = hr_folder
        self.image_filenames = [os.path.join(hr_folder, x) for x in os.listdir(hr_folder) if x.endswith(".jpg")]
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize high-res images
            transforms.ToTensor(),  # Convert to tensor
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        hr_image = Image.open(self.image_filenames[idx]).convert("RGB")
        hr_image = self.transform(hr_image)
        return hr_image