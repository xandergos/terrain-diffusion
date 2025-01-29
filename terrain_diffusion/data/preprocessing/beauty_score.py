import click
import h5py
import torch
import numpy as np
from tqdm import tqdm
from terrain_diffusion.data.laplacian_encoder import laplacian_decode

def analyze_terrain_frequency(heightmap_tensor, bins):
    """
    Analyzes terrain features by binning FFT components based on their distance from the center.
    
    Args:
        heightmap_tensor (torch.Tensor): Input heightmap (2D tensor)
        bins (int): Number of bins to divide the frequency spectrum into
        
    Returns:
        tuple: (frequency_magnitudes, bin_powers) where:
            - frequency_magnitudes: center distance of each bin
            - bin_powers: mean power of the FFT components in each bin
    """
    # Compute 2D FFT and shift zero-frequency to center
    fft = torch.fft.fft2(heightmap_tensor)
    fft_shifted = torch.fft.fftshift(fft)
    
    # Create spatial frequency grid
    h, w = heightmap_tensor.shape
    y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
    dist_from_center = torch.sqrt(x**2 + y**2)
    
    # Calculate power spectrum
    power_spectrum = torch.log(torch.abs(fft_shifted))
    
    # Create bins based on distance from center
    bin_edges = torch.linspace(0, 1, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Initialize arrays for results
    bin_powers = []
    
    # Calculate mean power for each bin
    for i in range(bins):
        mask = (dist_from_center >= bin_edges[i]) & (dist_from_center < bin_edges[i+1])
        if mask.any():
            bin_power = power_spectrum[mask].mean()
            bin_powers.append(bin_power.item())
        else:
            bin_powers.append(0.0)
    
    return bin_centers.tolist(), bin_powers

def calculate_beauty_score(lowfreq, residual):
    """
    Calculate beauty score using the linear regression model.
    Args:
        lowfreq (torch.Tensor): Low frequency components of the terrain
        residual (torch.Tensor): Residual components of the terrain
    Returns:
        float: Predicted beauty score
    """
    # Decode the terrain
    decoded = laplacian_decode(residual, lowfreq)
    
    # Calculate features
    freq_mags, powers = analyze_terrain_frequency(decoded, bins=4)
    std = torch.std(decoded).item()
    std_features = [
        np.log(std),          # log of std
        250 / std,            # inverse scaled std
        np.sqrt(std),         # square root of std
    ]
    features = powers + std_features
    
    # Coefficients from the linear regression model
    coefficients = [
        0.551959, -1.774091, 3.117426, -1.835090,
        -1.996856, -0.053519, 0.488380
    ]
    intercept = 4.44
    
    # Calculate predicted score
    score = sum(coef * feat for coef, feat in zip(coefficients, features)) + intercept
    return float(score)

@click.command()
@click.argument('dataset-file')
def assign_beauty_scores(dataset_file):
    """
    Assign beauty scores to all terrains in the dataset using a linear regression model.
    
    Args:
        dataset_file (str): Path to the HDF5 file containing the dataset.
    """
    with h5py.File(dataset_file, 'a') as f:
        for res in f.keys():
            res_group = f[res]
            for cid in tqdm(res_group.keys(), desc=f"Processing res={res} terrains"):
                cid_group = res_group[cid]
                for sub_cid in cid_group.keys():
                    obj = cid_group[sub_cid]
                    if isinstance(obj, h5py.Group) and 'lowfreq' in obj and 'residual' in obj:
                        lowfreq = torch.from_numpy(obj['lowfreq'][:]).float()
                        residual = torch.from_numpy(obj['residual'][:]).float()
                        
                        score = calculate_beauty_score(lowfreq, residual)
                        obj.attrs['beauty_score'] = score
        
        print("Finished assigning beauty scores to all terrains in the dataset.")

if __name__ == '__main__':
    assign_beauty_scores()
