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
from train_utils import inner_loop, generate_samples
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
        model.eval()

        # ---- fetch a random eval batch ----
        coords, intensity, image_ids = next(iter(eval_dataloader))  # eval_dataloader has shuffle=True
        coords  = coords.to(device)         # (B, N, 2) with N = H*W
        intensity = intensity.to(device)    # (B, N, 3)
        image_ids = image_ids.to(device)

        # ---- support/query split for adaptation ----
        coords_s, ints_s = generate_samples(coords, intensity, sample_size=2048)
        # inner adaptation (grads ON for latents only)
        _, modulation_vectors = inner_loop(coords_s, ints_s, image_ids, model, loss_fn, device)

        # ---- predict without grads on the full pixel set ----
        with torch.no_grad():
            preds = model(coords, modulation_vectors)   # (B, N, 3)

        # ---- rasterize preds/gt to (B, 3, H, W) ----
        B, N, C = preds.shape
        H = W = int(round(N ** 0.5))  # 32 for your current resize
        assert H * W == N, "N must be a square grid (resize your dataset to H==W)."

        # map coords -> integer pixel indices (handles [-1,1] or [0,1] automatically)
        with torch.no_grad():
            c = coords
            cmin, cmax = float(c.min().item()), float(c.max().item())
            if cmin >= -1.001 and cmax <= 1.001:
                x = (((c[..., 0] + 1) / 2) * (W - 1)).round().long()
                y = (((c[..., 1] + 1) / 2) * (H - 1)).round().long()
            else:
                x = (c[..., 0] * (W - 1)).round().long()
                y = (c[..., 1] * (H - 1)).round().long()
            x = x.clamp(0, W - 1); y = y.clamp(0, H - 1)

            # sort into raster order
            raster_idx = y * W + x                    # (B, N)
            order = torch.argsort(raster_idx, dim=1)  # (B, N)

            preds_sorted = torch.gather(preds,     1, order.unsqueeze(-1).expand(-1, -1, C))
            gt_sorted    = torch.gather(intensity, 1, order.unsqueeze(-1).expand(-1, -1, C))

            pred_img = preds_sorted.permute(0, 2, 1).contiguous().view(B, C, H, W).cpu()
            gt_img   = gt_sorted.permute(0, 2, 1).contiguous().view(B, C, H, W).cpu()

        # ---- visualize & PSNR ----
            rows = min(4, B)
            fig, axs = plt.subplots(rows, 2, figsize=(6, 3 * rows), constrained_layout=True)

            # dynamic range for PSNR: 1.0 for [0,1], 2.0 for [-1,1]
            with torch.no_grad():
                gmin, gmax = float(gt_img.min().item()), float(gt_img.max().item())
                max_i = 2.0 if gmin < -1e-6 else 1.0

            for i in range(rows):
                axs[i, 0].imshow(gt_img[i].permute(1, 2, 0).clamp(0, 1))
                axs[i, 0].set_title('Ground Truth')
                axs[i, 0].axis('off')

                mse = torch.mean((pred_img[i] - gt_img[i]) ** 2).item()
                psnr_value = 10.0 * math.log10((max_i ** 2) / (mse + 1e-12))

                axs[i, 1].imshow(pred_img[i].permute(1, 2, 0).clamp(0, 1))
                axs[i, 1].set_title(f'Reconstruction — PSNR: {psnr_value:.2f} dB')
                axs[i, 1].axis('off')

            plt.suptitle(f'Epoch {epoch} Reconstructions')
            plt.show()
            plt.close(fig)

    







    
    

 



