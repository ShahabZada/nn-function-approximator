import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class NNApproxDataset(Dataset):
    def __init__(self):
        
        self.img_dir = image_path = 'images/1.jpg'
        
        image = Image.open(image_path)
        # Convert image to grayscale
        gray_image = image.convert('L')
        image_array = np.array(gray_image)
        image_array = ~image_array/255

        self.height, self.width = image_array.shape
        n = self.height *self.width
        positions = np.array([(x/self.height, y/self.width) for x in range(self.height) for y in range(self.width)], dtype=np.float32)
        self.input_data = torch.tensor(positions, dtype=torch.float32)
        self.output_data = torch.tensor(image_array.flatten(), dtype=torch.float32).view(-1, 1)  # Reshape to (n, 1)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        
        pix_position, pix_value = self.input_data[idx], self.output_data[idx]
        return pix_position, pix_value
    
    def get_image_shape(self):
        return self.height, self.width