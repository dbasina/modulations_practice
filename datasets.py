import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
class MNIST_coordinate_dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5,], std= [0.5,])
        ])
        self.dataset = torchvision.datasets.MNIST(train = True, transform=self.transform, download=True, root = './data')
        
        #create grid
        C,H,W = self.dataset[0][0].shape
        self.x_coordinates = torch.linspace(-1,1,W)
        self.y_coordinates = torch.linspace(-1,1,H)
        Y_grid, X_grid = torch.meshgrid(self.y_coordinates, self.x_coordinates, indexing='ij')
        self.grid = torch.stack((X_grid, Y_grid), dim=-1)
        self.grid = self.grid.reshape(-1,2)
        self.grid = self.grid.float()

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        intensities = image.view(-1,1).contiguous()
        intensities = intensities.float()
        return self.grid, intensities, idx
    
    def __len__(self):
        return len(self.dataset)