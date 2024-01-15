from torch.utils.data import Dataset
import numpy as np
import torch
import cv2
import os

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = []
        self.target_paths = []
        
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.png') and 'GT_pose' in file:
                    self.target_paths.append(os.path.join(root, file))
                elif file.endswith('.png') and 'merged_integral' in file:
                    self.image_paths.append(os.path.join(root, file))

        self.image_paths.sort()
        self.target_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        target_path = self.target_paths[index]

        image = cv2.imread(image_path).astype(np.float32) / 255.
        target = cv2.imread(target_path).astype(np.float32) / 255.
        
        image = np.transpose(image, (2, 0, 1))
        target = np.transpose(target, (2, 0, 1))

        return torch.from_numpy(image), torch.from_numpy(target[np.newaxis, 0, ...])#, parameters

class Guided_Dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = []
        self.target_paths = []
        self.param_paths = []
        self.temp_min = 260.
        self.norm_factor = 1 / (310. - 260.)
        
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.png') and 'GT_pose' in file:
                    self.target_paths.append(os.path.join(root, file))
                elif file.endswith('.png') and 'merged_integral' in file:
                    self.image_paths.append(os.path.join(root, file))
                elif file.endswith('.txt') and 'Parameters' in file:
                    self.param_paths.append(os.path.join(root, file))

        self.image_paths.sort()
        self.target_paths.sort()
        self.param_paths.sort()

        self.temps = []
        key = "compass direction of sunlight in degrees ="
        for param_path in self.param_paths:
            with open(param_path, 'r') as param_file:
                lines = param_file.readlines()
                for line in lines:
                    if line.startswith(key):
                        temp = line.split('=')[1].strip()
                        temp = (float(temp) - self.temp_min) * self.norm_factor
                        temp = torch.tensor(temp, dtype=torch.float32)
                        self.temps.append(temp)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        target_path = self.target_paths[index]
        temp = self.temps[index]

        image = cv2.imread(image_path).astype(np.float32) / 255.
        target = cv2.imread(target_path).astype(np.float32) / 255.
        
        image = np.transpose(image, (2, 0, 1))
        target = np.transpose(target, (2, 0, 1))

        return (torch.from_numpy(image), temp), torch.from_numpy(target[np.newaxis, 0, ...])#, parameters