import torch
import torch.nn as nn
import math
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import ImageFolder, MNIST
from torch.utils.data import DataLoader as Dataloader
from datasets import MNIST_coordinate_dataset, KodakCoordinateDataset
from train_utils import inner_loop, generate_samples, eval_dataset, visualize_reconstruction, plot_images
from fkan_inr import FKAN_INR_Modulated
from tqdm import tqdm

batch_size = 16
eval_batch_size = 4
outer_epochs = 4000
# dataset = MNIST_coordinate_dataset()
dataset = KodakCoordinateDataset(resize=32)
dataset_subset = torch.utils.data.Subset(dataset, list(range(5)))
dataloader = DataLoader(dataset, batch_size= batch_size, shuffle=False, pin_memory=True)
eval_dataloader = DataLoader(dataset, batch_size= eval_batch_size, shuffle=True, pin_memory=True)

model = FKAN_INR_Modulated(2, 128, 3, 1024, 512)
outer_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
outer_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()
model.to(device)

best_average_psnr = 0.0
for epoch in tqdm(range(outer_epochs)):
    model.train()

    #inner update of modulation vectors
    for coords, intensity, image_ids in dataloader:
        coords, intensity, image_ids = coords.to(device), intensity.to(device), image_ids.to(device)
        sampled_coords, sampled_intensities = generate_samples(coords, intensity, sample_size=64)
        
        inner_loss, modulation_vectors = inner_loop(sampled_coords, sampled_intensities, image_ids, model, loss_fn, device)
        
        intensity_predictions = model(coords, modulation_vectors)
        outer_loss = loss_fn(intensity_predictions, intensity)

        outer_optimizer.zero_grad()
        outer_loss.backward()
        outer_optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        average_loss, average_psnr = eval_dataset(model, eval_dataloader, device)
        if average_psnr > best_average_psnr:
            best_average_psnr = average_psnr
            torch.save(model.state_dict(), "Checkpoints/best_FKAN_INR_kodak.pth")

        print(f"Epoch {epoch+1}/{outer_epochs}, Mean Eval PSNR: {average_psnr:.2f}dB")
        coords_viz, intensity_viz, preds_viz = visualize_reconstruction(model, dataset_subset, device, image_index=0, inner_steps=20)
        plot_images(coords_viz, intensity_viz, preds_viz, epoch+1)    

    







    
    

 



