import torch
import torch.nn as nn
import math
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import ImageFolder, MNIST

from torch.utils.data import DataLoader as Dataloader

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

class INR_Layer(nn.Module):
    def __init__(self, input_features, output_features, latent_dim = 64, is_first = False, bias = True, omega_0 = 30):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.latent_dim = latent_dim
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(self.input_features,self.output_features, bias=bias)

        self.scale = nn.Linear(in_features=self.latent_dim, out_features=self.output_features)
        self.shift = nn.Linear(in_features=self.latent_dim, out_features=self.output_features)

        self.init_weights()

    def init_weights(self):
        # initialization as described in the SIREN paper
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.input_features, 1 / self.input_features)
            else:
                bound = math.sqrt(6 / self.input_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x, latent):
        linear_output = self.omega_0 * self.linear(x)
        sine_output = torch.sin(linear_output)

        scale_modulation = self.scale(latent)
        shift_modulation = self.shift(latent)
        
        scale_modulation = scale_modulation.unsqueeze(1)
        shift_modulation = shift_modulation.unsqueeze(1)
        modulated_output = scale_modulation*sine_output + shift_modulation
        return modulated_output

class INR(nn.Module):
    def __init__(self,input_features=2, hidden_features = 256, output_features = 1, dataset_size = 0, latent_dim = 64, omega_0 = 30):
        super().__init__()
        self.latent_dimensions = latent_dim
        self.dataset_size = dataset_size
        self.embeddings = nn.Embedding(self.dataset_size, self.latent_dimensions)
        nn.init.uniform_(self.embeddings.weight, -1e-4, 1e-4)

        self.INR_1 = INR_Layer(input_features=input_features, output_features=hidden_features,is_first=True, omega_0=omega_0)
        self.INR_2 = INR_Layer(input_features=hidden_features, output_features=hidden_features,is_first= False, omega_0=omega_0)
        self.output_layer = nn.Linear(hidden_features, output_features)

    def forward(self, x,image_ids):
        latent = self.embeddings(image_ids)
        output_1 = self.INR_1(x, latent)
        output_2 = self.INR_2(output_1, latent)
        output = self.output_layer(output_2)
        return output

dataset = MNIST_coordinate_dataset()
mnist_image_coords = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True)
image_coordinates, intensities, id = next(iter(mnist_image_coords))
print(image_coordinates.shape, intensities.shape)

model = INR(dataset_size=len(dataset))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(100):
    model.train()
    epoch_loss = 0
    for coords, intensity, image_ids in mnist_image_coords:
        coords, intensity,image_ids = coords.to(device), intensity.to(device), image_ids.to(device)
        optimizer.zero_grad()
        preds = model(coords, image_ids)
        loss = loss_fn(preds, intensity)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    average_loss = epoch_loss/len(mnist_image_coords)
    embeddings_std = model.embeddings.weight.std(dim=0).mean().item()
    print(f"Epoch {epoch+1}/100, Average Loss: {average_loss}, Embeddings Std: {embeddings_std}")

    if (epoch+1) % 20 == 0:
        model.eval()
        idx = 0
        coords, intensities, _ = dataset[idx]
        coords, intensities = coords.unsqueeze(0).to(device), intensities.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(coords, torch.tensor([idx], device=device))
        pred_img = pred.view(28, 28).detach().cpu()

        plt.subplot(1,2,1)
        plt.title("Ground Truth")
        plt.imshow(intensities.view(28, 28).cpu(), cmap='gray')

        plt.subplot(1,2,2)
        plt.title("Reconstruction")
        plt.imshow(pred_img, cmap='gray')
        plt.show()

