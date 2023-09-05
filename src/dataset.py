import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class NNApproxDataset(Dataset):
    def __init__(self):
        
        self.img_dir = image_path = 'images/5.jpg'
        
        image = Image.open(image_path)
        # Convert image to grayscale
        gray_image = image.convert('L')
        image_array = np.array(gray_image)
        image_array = ~image_array
        binary_image = np.where(image_array >= 128, 1, 0)
        

        self.height, self.width = image_array.shape
        n = self.height *self.width
        positions = np.array([(x/self.height, y/self.width) for x in range(self.height) for y in range(self.width)], dtype=np.float32)
        self.input_data = torch.tensor(positions, dtype=torch.float32)
        self.output_data = torch.tensor(binary_image.flatten(), dtype=torch.float32).view(-1, 1)  # Reshape to (n, 1)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        
        pix_position, pix_value = self.input_data[idx], self.output_data[idx]
        return pix_position, pix_value
    
    def get_image_shape(self):
        return self.height, self.width
    

class NNApproxDatasetFourier(Dataset):
    def __init__(self, n_fourier_features):
        
        self.img_dir = image_path = 'images/3.jpg'
        
        image = Image.open(image_path)
        # Convert image to grayscale
        gray_image = image.convert('L')
        image_array = np.array(gray_image)
        image_array = ~image_array
        binary_image = np.where(image_array >= 128, 1, 0)
        

        self.height, self.width = image_array.shape
        n = self.height *self.width
        positions = []
        for x in range(self.width):
            for y in range(self.height):
                normalized_x = self.normalize_to_pi(x, 0,self.width )
                normalized_y = self.normalize_to_pi(y, 0, self.height)
                positions.append(np.array(self.get_fourier_feature(normalized_x, normalized_y, int(n_fourier_features/4)))) 
        positions = np.array(positions, dtype=np.float32)
        self.input_data = torch.tensor(positions, dtype=torch.float32)
        self.output_data = torch.tensor(binary_image.flatten(), dtype=torch.float32).view(-1, 1)  
    def normalize_to_pi(self,value, domain_min, domain_max):
        x_min = domain_min
        x_max = domain_max
        y_min = -np.pi
        y_max = np.pi

        normalized_value = ((value - x_min) / (x_max - x_min)) * (y_max - y_min) + y_min
        return normalized_value
    def get_fourier_feature(self, x, y , upto_order):
        feature_array = []
        for n in range( 1,upto_order+1,1):
            feature_array.extend([np.sin(n*x), np.cos(n*x), np.sin(n*y), np.cos(n*y)])
        return feature_array
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        
        pix_position, pix_value = self.input_data[idx], self.output_data[idx]
        return pix_position, pix_value
    
    def get_image_shape(self):
        return self.height, self.width