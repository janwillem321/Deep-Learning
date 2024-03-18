import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PairedImageDataset(Dataset):
    def __init__(self, rootA, rootB, transform=None):
        """
        Args:
            rootA (string): Directory with all the images in trainA.
            rootB (string): Directory with all the images in trainB.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.rootA = rootA
        self.rootB = rootB
        self.transform = transform

        # Assuming filenames in both folders are the same and in order
        self.filenames = sorted(os.listdir(rootA))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_nameA = os.path.join(self.rootA, self.filenames[idx])
        imageA = Image.open(img_nameA).convert('RGB')
        
        img_nameB = os.path.join(self.rootB, self.filenames[idx])
        imageB = Image.open(img_nameB).convert('RGB')

        if self.transform:
            imageA = self.transform(imageA)
            imageB = self.transform(imageB)

        return imageA, imageB
    
