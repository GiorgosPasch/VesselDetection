import torch
import cv2
from torchvision import transforms as transforms
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class VesselDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels,root, transforms=False):
        self.transforms = transforms
        self.paths = paths
        self.labels = labels
        self.root = root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        x = cv2.imread(self.root+str(path)).astype(np.float32)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) / 255
        x = transform(x)
        label = self.labels[idx]
        return x, label, path
