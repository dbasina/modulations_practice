import torch
import torch.nn as nn
import math
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import ImageFolder, MNIST
from torch.utils.data import DataLoader as Dataloader
from datasets import MNIST_coordinate_dataset
from models import INR
from train_utils import inner_loop, generate_samples
from fkan_inr import FKAN_INR_Modulated
from tqdm import tqdm

batch_size = 16
eval_batch_size = 4
outer_epochs = 2000
dataset = MNIST_coordinate_dataset()
dataset_subset = torch.utils.data.Subset(dataset, list(range(64)))
mnist_dataloader = DataLoader(dataset_subset, batch_size= batch_size, shuffle=False, pin_memory=True)
mnist_eval_dataloader = DataLoader(dataset_subset, batch_size= eval_batch_size, shuffle=True, pin_memory=True)

model = FKAN_INR_Modulated(2, 128, 1, 1024, 512)
outer_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
outer_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()
model.to(device)


for epoch in tqdm(range(outer_epochs)):
    model.train()

    #inner update of modulation vectors
    for coords, intensity, image_ids in mnist_dataloader:
        coords, intensity, image_ids = coords.to(device), intensity.to(device), image_ids.to(device)
        sampled_coords, sampled_intensities = generate_samples(coords, intensity, sample_size=512)
        
        inner_loss, modulation_vectors = inner_loop(sampled_coords, sampled_intensities, image_ids, model, loss_fn, device)
        
        intensity_predictions = model(coords, modulation_vectors)
        outer_loss = loss_fn(intensity_predictions, intensity)

        outer_optimizer.zero_grad()
        outer_loss.backward()
        outer_optimizer.step()
    
    if (epoch+1) % 10 == 0:
        model.eval()

        # get a random batch of at least 4 images
        coords, intensity, image_ids = next(iter(mnist_eval_dataloader))  # ensure shuffle=True or build a random subset
        coords, intensity, image_ids = coords.to(device), intensity.to(device), image_ids.to(device)

        # (Optional but recommended) split support/query for eval
        coords_s, ints_s = generate_samples(coords, intensity, sample_size=512)  # support
        # adapt φ with grads
        _, modulation_vectors = inner_loop(coords_s, ints_s, image_ids, model, loss_fn, device)

        # predict without grads
        with torch.no_grad():
            intensity_predictions = model(coords, modulation_vectors)

        # reshape & visualize
        intensity_predictions = intensity_predictions.cpu().view(-1, 1, 28, 28)
        intensity = intensity.cpu().view(-1, 1, 28, 28)

        fig, axs = plt.subplots(4, 2, figsize=(6, 12))
        for i in range(min(4, intensity.shape[0])):
            axs[i, 0].imshow(intensity[i, 0], cmap='gray', vmin=0, vmax=1)
            axs[i, 0].set_title('Ground Truth'); axs[i, 0].axis('off')

            axs[i, 1].imshow(intensity_predictions[i, 0], cmap='gray', vmin=0, vmax=1)
            axs[i, 1].set_title('Reconstruction'); axs[i, 1].axis('off')

            mse = torch.mean((intensity_predictions[i] - intensity[i])**2).item()
            psnr_value = 10 * math.log10(1.0 / (mse + 1e-12))
            axs[i, 1].set_xlabel(f'PSNR: {psnr_value:.2f} dB')

        plt.suptitle(f'Epoch {epoch} Reconstructions')
        plt.tight_layout(); plt.show()
    







    
    

 



