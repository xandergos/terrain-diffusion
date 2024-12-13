import torch.nn as nn
import torchvision.transforms.functional as TF
import torch
import numpy as np


class LaplacianPyramidEncoder(nn.Module):
    def __init__(self,
                 resize_scales,
                 sigma,
                 raw_mean,
                 raw_std,
                 final_mean=0,
                 final_std=0.5,
                 ):
        """
        Args:
            resize_scales (list): Amount to downsample each layer. The first layer is always full resolution.
                For example [4, 16] will result in 3 layers with full, 1/4th, and 1/16th resolution.
            sigma (list|float): Sigma used for gaussian blur, 0 for no blur. If a list, should be same length as resize_scales.
            raw_mean (list): Expected mean of each channel. Length should be len(scale) + 1.
            raw_std (list): Expected standard deviation of each channel. Length should be len(scale) + 1.
            final_mean (float|list, optional): Desired mean of the final latents. Defaults to 0.
            final_std (float|list, optional): Desired standard deviation of the final latents. Defaults to 0.5.
        """
        super().__init__()
        self.resize_scales = resize_scales
        self.sigma = sigma
        if not isinstance(sigma, list):
            self.sigma = [sigma] * len(resize_scales)
        self.scale = np.float32(final_std) / np.float32(raw_std)
        self.bias = np.float32(final_mean) - np.float32(raw_mean) * self.scale

        self.inv_scale = np.float32(raw_std) / np.float32(final_std)
        self.inv_bias = np.float32(raw_mean) - np.float32(final_mean) * self.inv_scale
        
        self.final_std = final_std
        self.final_mean = final_mean

    def encode(self, x: torch.Tensor, return_downsampled=False):
        """Encode an image into a pyramid encoding.

        Args:
            x (torch.Tensor): Image to encode. Shape (..., 1, H, W).

        Returns:
            torch.Tensor: Pyramid encoding. Shape (..., num_layers, H, W).
        """
        imgs = [x]
        downsampled_imgs = [x]
        for d, sigma in zip(self.resize_scales, self.sigma):
            img, downsampled = self._resample(imgs[-1], d, sigma, return_downsampled=True)
            imgs.append(img)
            downsampled_imgs.append(downsampled)
        for i in range(len(imgs) - 1):
            imgs[i] = (imgs[i] - imgs[i + 1]) * self.scale[i] + self.bias[i]
        imgs[-1] = imgs[-1] * self.scale[-1] + self.bias[-1]
        for i in range(len(downsampled_imgs)):
            downsampled_imgs[i] = downsampled_imgs[i] * self.scale[i] + self.bias[i]
        
        if return_downsampled:
            return torch.cat(imgs, dim=-3), downsampled_imgs
        else:
            return torch.cat(imgs, dim=-3)

    def decode(self, x):
        """
        Args:
            x (torch.Tensor): Pyramid encoding. Shape (..., num_layers, H, W).

        Returns:
            torch.Tensor: Image. Shape (..., 1, H, W).
        """
        encodings = torch.tensor_split(x.to(torch.float32), len(self.scale), dim=-3)
        out = [torch.zeros_like(encodings[0]) for _ in range(len(encodings))]
        for i, (s, b) in enumerate(zip(self.inv_scale, self.inv_bias)):
            out[i] = out[i] + encodings[i] * torch.full_like(encodings[i], s) + torch.full_like(encodings[i], b)
        return torch.sum(torch.cat(out, dim=-3), dim=-3, keepdim=True)
    
    def forward(self, x):
        return self.encode(x)
    
    def _resample(self, x, resize_scale, sigma, resize=True, return_downsampled=False):
        """Resamples an image with a blur and/or double resize.

        Args:
            x (torch.Tensor): Image to resample. (..., 1, H, W)
            resize_scale (int): Amount to downsample the image.
            resize (bool, optional): Whether to resize the image before upscaling. Defaults to True.

        Returns:
            torch.Tensor: Resampled image.
        """
        if sigma > 0:
            radius = int(sigma * 3 / 2) * 2 + 1
        else:
            radius = 0
        
        # Remember original shape
        spatial_dims = x.shape[-2:]
        batch_dims = x.shape[:-3]
        assert spatial_dims[0] == spatial_dims[1]
        assert x.shape[-3] == 1
        
        # Flatten batch dimensions
        x = x.view(-1, 1, *x.shape[-2:])
        
        if resize:
            size = spatial_dims[-1]
            next = TF.resize(x, (size // resize_scale, size // resize_scale))
        else:
            size = spatial_dims[-1] * resize_scale
            next = x
        
        if sigma > 0:
            next = TF.pad(next, radius, padding_mode='edge')
            next = TF.gaussian_blur(next, radius, sigma)
            downsampled = next[..., radius:-radius, radius:-radius]
            next = torch.nn.functional.interpolate(next, size=(size + 2 * radius * resize_scale, size + 2 * radius * resize_scale), 
                                                   mode='bicubic', align_corners=False)
            next = next[..., radius * resize_scale:-radius * resize_scale, radius * resize_scale:-radius * resize_scale]
        else:
            next = torch.nn.functional.interpolate(next, size=(size, size), mode='bicubic', align_corners=False)
        
        # Restore batch dimensions
        next = next.view(*batch_dims, 1, size, size)
        downsampled = downsampled.view(*batch_dims, 1, size // resize_scale, size // resize_scale)
        
        if return_downsampled:
            return next, downsampled
        else:
            return next
    
def denoise_pyramid_layer(x0, encoder, depth, loss_fn=torch.nn.functional.mse_loss, maxiter=10):
    """Denoises a predicted pyramid encoding.

    Args:
        x (torch.Tensor): Predicted pyramid encoding. Shape (B, 1, H, W).
        encoder (LaplacianPyramidEncoder): Encoder used to decode the pyramid.
        depth (int): Depth of the pyramid layer to denoise.
        loss_fn (callable, optional): Loss function to use. Defaults to torch.nn.functional.mse_loss.
        maxiter (int, optional): Maximum number of iterations. Defaults to 10.
    Returns:
        torch.Tensor: Denoised image.
    """
    from scipy.optimize import minimize
    
    use_np = isinstance(x0, np.ndarray)
    if use_np:
        x0 = torch.tensor(x0, dtype=torch.float32)
    x0 = x0.to(torch.float32).detach().cpu()
    
    # We want to recreate the original image
    target = x0
    
    # Downsample the image to the latent size of the pyramid layer
    x0 = TF.resize(x0, (x0.shape[-1] // encoder.resize_scales[depth-1], x0.shape[-1] // encoder.resize_scales[depth-1]))
    
    # We have to flatten for scipy, save shape to reconstruct.
    x0_shape = x0.shape
    x0 = x0.flatten().numpy()
    
    def objective(x):
        # Ensure we can get a gradient with torch
        x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        # Now we scale the image up with the encoder's resample function
        resampled = encoder._resample(x.reshape(x0_shape), encoder.resize_scales[depth-1], encoder.sigma[depth-1], resize=False)
        loss = loss_fn(resampled, target)
        loss.backward()
        return loss.item(), x.grad.cpu().numpy().flatten()
    
    result = minimize(objective, x0, method='L-BFGS-B', jac=True, options={'maxiter': maxiter})
    with torch.no_grad():
        out = encoder._resample(torch.tensor(result.x.reshape(x0_shape), dtype=torch.float32), 
                                encoder.resize_scales[depth-1], encoder.sigma[depth-1], resize=False)
        if use_np:
            return out.cpu().numpy()
        else:
            return out
        
def denoise_pyramid(image, encoder):
    encoded = encoder.encode(encoder.decode(image))
    return torch.cat([image[..., [0], :, :], encoded[..., 1:, :, :]], dim=-3)
        
def encode_postprocess(encoder):
    def postprocessor(tile_y, tile_x, image):
        return encoder.encode(image)
    return postprocessor

def decode_postprocess(encoder, denoise=True):
    def postprocessor(tile_y, tile_x, image):
        if denoise:
            image = denoise_pyramid(image, encoder)
        decoded = encoder.decode(image)
        return decoded
    return postprocessor