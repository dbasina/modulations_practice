import torch


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
