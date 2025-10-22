import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
class MNIST_coordinate_dataset(Dataset):
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

class KodakCoordinateDataset(Dataset):
    def __init__(self, resize = None):
       self.image_path_csv = "/home/divesh-basina/Documents/research/Thesis/modulations_practice/Kodak/kodak_images.csv"
       self.images = pd.read_csv(self.image_path_csv)["image_path"].tolist()
       self.transform = transforms.Compose([
            transforms.Resize((resize,resize)) if resize is not None else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t if (t.amax() - t.amin()) == 0 
                                          else (t - t.amin()) / (t.amax() - t.amin()))
            ])

       #load sample image for dimensions and construct grid
       self.sample_image = torchvision.io.read_image(self.images[0])
       C,H,W = self.sample_image.shape
       if resize is not None:
           H = resize
           W = resize

       self.xcoordinates = torch.linspace(-1,1,W)
       self.ycoordinates = torch.linspace(-1,1,H)
       self.Y_grid, self.X_grid = torch.meshgrid(self.ycoordinates, self.xcoordinates, indexing='ij')
       self.grid = torch.stack( (self.X_grid,self.Y_grid), dim = -1)
       self.grid = self.grid.reshape(-1,2)
       self.grid = self.grid.float()

    def __getitem__(self,index):
        image = Image.open(self.images[index])
        image = image.convert("RGB")
        image = self.transform(image)
        intensities = image.permute(1,2,0).view(-1,3).contiguous()
        intensities = intensities.float()
        return self.grid, intensities, index

    def __len__(self):
        return len(self.images)