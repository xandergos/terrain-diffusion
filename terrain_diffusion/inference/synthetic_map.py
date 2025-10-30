import random
import numpy as np
import rasterio
from pyfastnoiselite.pyfastnoiselite import FastNoiseLite, NoiseType, FractalType
import torch
from terrain_diffusion.inference.perlin_transform import build_quantiles, transform_perlin

def make_synthetic_map_factory(frequency_mult=1.0, seed=None):
    elev_img = rasterio.open("data/global/wc2.1_10m_elev.tif").read(1)
    temp_img = rasterio.open("data/global/wc2.1_10m_bio_1.tif").read(1)
    temp_std_img = rasterio.open("data/global/wc2.1_10m_bio_4.tif").read(1)
    precip_img = rasterio.open("data/global/wc2.1_10m_bio_12.tif").read(1)
    precip_std_img = rasterio.open("data/global/wc2.1_10m_bio_15.tif").read(1)

    def process_image(img):
        # Crop to middle 2/3rds (remove first and last 30 degrees of latitude)
        h = img.shape[0]
        crop_start = h // 6
        crop_end = h - h // 6
        img = img[crop_start:crop_end, :]
        img[img < -30000] = np.nan
        return img

    elev_img = process_image(elev_img)
    temp_img = process_image(temp_img)
    temp_std_img = process_image(temp_std_img)
    precip_img = process_image(precip_img)
    precip_std_img = process_image(precip_std_img)

    def get_lapse_rate(precip):
        lapse_rate = -6.5 + 0.0015 * precip
        lapse_rate = np.clip(lapse_rate, -9.8, -4.0)
        return lapse_rate / 1000

    # Compute linear relationship between temp and temp_std
    valid_mask = ~np.isnan(temp_img)
    temp_flat = temp_img[valid_mask]
    temp_std_flat = temp_std_img[valid_mask]

    # Fit linear regression: temp_std = a * temp + b
    coeffs = np.polyfit(temp_flat, temp_std_flat, 1)
    a_temp_std, b_temp_std = coeffs[0], coeffs[1]

    # Remove linear relationship from temp_std
    temp_std_predicted = a_temp_std * temp_img + b_temp_std
    temp_std_img = temp_std_img - temp_std_predicted

    temp_img = temp_img - get_lapse_rate(precip_img) * np.maximum(0, elev_img)
    
    temp_std_p1 = np.percentile(temp_std_img[valid_mask], 0.1)
    temp_std_p99 = np.percentile(temp_std_img[valid_mask], 99.9)

    def build_synthetic_map(base_image, frequency, octaves, lacunarity, gain, seed):
        noise = FastNoiseLite(seed=seed)
        noise.noise_type = NoiseType.NoiseType_Perlin
        noise.frequency = frequency
        noise.fractal_type = FractalType.FractalType_FBm
        noise.fractal_octaves = octaves
        noise.fractal_lacunarity = lacunarity
        noise.fractal_gain = gain
        
        # Generate noise for statistics
        size = 32 * 1024
        x = np.arange(0, size, 32, dtype=np.float32)
        y = np.arange(0, size, 32, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        # Flatten the grid coordinates
        Xs = xx.flatten()
        Ys = yy.flatten()
        coords = np.array([Xs, Ys], dtype=np.float32)

        # Generate noise for all coordinates
        noise_values = noise.gen_from_coords(coords)
        noise_quantiles = build_quantiles(noise_values.flatten(), n_quantiles=64, eps=1e-4)
        base_image_quantiles = build_quantiles(base_image.flatten(), n_quantiles=64, eps=1e-4)
        
        transform_fn = lambda x: transform_perlin(x, noise_quantiles, base_image_quantiles)
        return noise, transform_fn
        
        
    def sample_synthetic_map(noise, transform_fn, i1, j1, i2, j2):
        x = np.arange(i1, i2, dtype=np.float32)
        y = np.arange(j1, j2, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        # Flatten the grid coordinates
        Xs = xx.flatten()
        Ys = yy.flatten()
        coords = np.array([Xs, Ys], dtype=np.float32)

        # Generate noise for all coordinates
        noise_values = noise.gen_from_coords(coords)
        transformed_values = transform_fn(noise_values)
        return transformed_values.reshape(i2 - i1, j2 - j1)

    synthetic_elev_params = build_synthetic_map(elev_img, 0.05 * frequency_mult, 4, 2.0, 0.5, seed=(seed or random.randint(0, 2**30))+1)
    synthetic_temp_params = build_synthetic_map(temp_img, 0.05 * frequency_mult, 2, 2.0, 0.5, seed=(seed or random.randint(0, 2**30))+2)
    synthetic_temp_std_params = build_synthetic_map(temp_std_img, 0.05 * frequency_mult, 4, 2.0, 0.5, seed=(seed or random.randint(0, 2**30))+3)
    synthetic_precip_params = build_synthetic_map(precip_img, 0.05 * frequency_mult, 4, 2.0, 0.5, seed=(seed or random.randint(0, 2**30))+4)
    synthetic_precip_std_params = build_synthetic_map(precip_std_img, 0.05 * frequency_mult, 4, 2.0, 0.5, seed=(seed or random.randint(0, 2**30))+5)

    def sample_full_synthetic_map(i1, j1, i2, j2):
        synthetic_elev = sample_synthetic_map(*synthetic_elev_params, i1, j1, i2, j2)
        synthetic_temp = sample_synthetic_map(*synthetic_temp_params, i1, j1, i2, j2)
        synthetic_temp_std = sample_synthetic_map(*synthetic_temp_std_params, i1, j1, i2, j2)
        synthetic_precip = sample_synthetic_map(*synthetic_precip_params, i1, j1, i2, j2)
        synthetic_precip_std = sample_synthetic_map(*synthetic_precip_std_params, i1, j1, i2, j2)

        # Correcting temp
        synthetic_temp = synthetic_temp + get_lapse_rate(synthetic_precip) * np.maximum(0, synthetic_elev)
        synthetic_temp = np.clip(synthetic_temp, -10, 40)

        # Correcting  temp std
        t = (synthetic_temp_std - temp_std_p1) / (temp_std_p99 - temp_std_p1)
        baseline = np.maximum(temp_std_p1, -(a_temp_std * synthetic_temp + b_temp_std))
        synthetic_temp_std = t * (temp_std_p99 - baseline) + baseline
        synthetic_temp_std = synthetic_temp_std + (a_temp_std * synthetic_temp + b_temp_std)
        synthetic_temp_std = np.maximum(synthetic_temp_std, 20)

        # Correcting precip std
        synthetic_precip_std = synthetic_precip_std * np.maximum(0, (185 - 0.04111 * synthetic_precip) / 185)
        
        synthetic_elev = np.sign(synthetic_elev) * np.sqrt(np.abs(synthetic_elev))

        synthetic_map = np.stack([synthetic_elev, synthetic_temp, synthetic_temp_std, synthetic_precip, synthetic_precip_std], axis=0)
        return torch.from_numpy(synthetic_map).float()
    
    return sample_full_synthetic_map