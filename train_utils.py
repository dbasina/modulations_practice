import torch
import matplotlib.pyplot as plt
import numpy as np

def update_modulations(coords,intensity, modulation_vectors, model, loss_fn, device, inner_learning_rate=0.01):

    with torch.enable_grad():
        preds = model(coords, modulation_vectors)
        loss = loss_fn(preds, intensity)
        #compute gradients for modulation vectors only
        (modulation_loss_gradients,) = torch.autograd.grad(loss*len(intensity), (modulation_vectors,), create_graph=True)

        #update modulation vectors with gradient descent step
        modulation_vectors = modulation_vectors - (inner_learning_rate * modulation_loss_gradients)
        return loss, modulation_vectors

def inner_loop(coords=None, intensities=None, image_ids=None, model=None, loss_fn=None, device=None, num_steps=20):
    
    modulation_vectors = torch.zeros(len(image_ids), model.latent_dimensions, requires_grad=True).to(device)
    for _ in range(num_steps):
        loss, modulation_vectors = update_modulations(coords, intensities, modulation_vectors, model, loss_fn, device)
    return loss, modulation_vectors

def generate_samples(coords, intensities, sample_size=256):
    #coods in shape (B, Pixesel_count, 2)
    #intensity in shape (B, Pixel_count, 1)
    N = coords.shape[1]
    sample_size = min(sample_size, N)
    perm = torch.randperm(N, device=coords.device)[:sample_size]
    sampled_coords = coords[:, perm, :]
    sampled_intensities = intensities[:, perm, :]
    return sampled_coords, sampled_intensities

def psnr(mse):
    return 10 * torch.log10(1 / mse)

def eval_dataset(model, dataloader, device):
    model.eval()
    loss_fn = torch.nn.MSELoss()
    total_loss = 0.0
    total_psnr = 0.0
    count = 0

    for coords, intensity, image_ids in dataloader:
        coords, intensity, image_ids = coords.to(device), intensity.to(device), image_ids.to(device)

        #inner adaptation
        coords_s, ints_s = generate_samples(coords, intensity, sample_size=2048)
        _, modulation_vectors = inner_loop(coords_s, ints_s, image_ids, model, loss_fn, device)

        #predict on full set
        with torch.no_grad():
            preds = model(coords, modulation_vectors)

        loss = loss_fn(preds, intensity)
        total_loss += loss.item() * coords.shape[0]
        total_psnr += psnr(loss).item() * coords.shape[0]
        count += coords.shape[0]

    avg_loss = total_loss / count
    avg_psnr = total_psnr / count
    return avg_loss, avg_psnr

def visualize_reconstruction(model, dataset, device, image_index=0, inner_steps=20):
    model.eval()
    coords, intensity, _ = dataset[image_index]
    coords = coords.unsqueeze(0).to(device)
    intensity = intensity.unsqueeze(0).to(device)
    image_id = torch.tensor([image_index], device=device)

    #inner adaptation
    coords_s, ints_s = generate_samples(coords, intensity, sample_size=2048)
    _, modulation_vectors = inner_loop(coords_s, ints_s, image_id, model, torch.nn.MSELoss(), device, num_steps=inner_steps)

    #predict on full set
    preds = model(coords, modulation_vectors)
    return coords.cpu(), intensity.cpu(), preds.detach().cpu()

def plot_images(coords, intensity, preds, epoch):
    """
    coords:   (1, N, 2) or (N, 2)  — not necessarily raster-ordered
    intensity:(1, N, C) or (N, C)
    preds:    (1, N, C) or (N, C)
    C in {1, 3}
    """
    import torch, numpy as np, math
    import matplotlib.pyplot as plt

    # Squeeze optional batch dim
    if coords.dim() == 3 and coords.size(0) == 1:
        coords = coords.squeeze(0)
    if intensity.dim() == 3 and intensity.size(0) == 1:
        intensity = intensity.squeeze(0)
    if preds.dim() == 3 and preds.size(0) == 1:
        preds = preds.squeeze(0)

    assert coords.dim() == 2 and coords.size(-1) == 2, f"coords must be (N,2), got {tuple(coords.shape)}"
    assert intensity.dim() == 2 and preds.dim() == 2, "intensity/preds must be (N,C)"
    N, C = intensity.shape
    H = W = int(round(math.sqrt(N)))
    assert H * W == N, f"N must be a perfect square. Got N={N}"

    # Map coords -> integer pixel indices for sorting into raster order
    with torch.no_grad():
        c = coords
        cmin, cmax = float(c.min().item()), float(c.max().item())

        # handle normalized coords in [-1,1] or [0,1], else treat as pixel space
        if cmin >= -1.001 and cmax <= 1.001:
            x = (((c[:, 0] + 1) / 2) * (W - 1)).round().long()
            y = (((c[:, 1] + 1) / 2) * (H - 1)).round().long()
        elif cmin >= -1e-6 and cmax <= 1.0 + 1e-6:
            x = (c[:, 0] * (W - 1)).round().long()
            y = (c[:, 1] * (H - 1)).round().long()
        else:
            x = c[:, 0].round().long()
            y = c[:, 1].round().long()

        x = x.clamp(0, W - 1)
        y = y.clamp(0, H - 1)

        # sort into scanline raster order
        raster_idx = y * W + x                # (N,)
        order = torch.argsort(raster_idx)     # (N,)

        gt_sorted    = intensity[order]       # (N, C)
        preds_sorted = preds[order]           # (N, C)

    # reshape to images
    if C == 1:
        gt_img    = gt_sorted.view(H, W).detach().cpu().numpy()
        pred_img  = preds_sorted.view(H, W).detach().cpu().numpy()
    elif C == 3:
        gt_img    = gt_sorted.view(H, W, 3).detach().cpu().numpy()
        pred_img  = preds_sorted.view(H, W, 3).detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported channel count C={C}; expected 1 or 3.")

    # plot
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    if C == 1:
        axs[0].imshow(gt_img, cmap='gray', vmin=0, vmax=1)
        axs[1].imshow(pred_img, cmap='gray', vmin=0, vmax=1)
    else:
        axs[0].imshow(np.clip(gt_img, 0, 1))
        axs[1].imshow(np.clip(pred_img, 0, 1))
    axs[0].set_title('Ground Truth'); axs[0].axis('off')
    axs[1].set_title('Reconstruction'); axs[1].axis('off')
    plt.suptitle(f'Epoch {epoch}')
    plt.show()
