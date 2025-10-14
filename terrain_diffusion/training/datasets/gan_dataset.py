import random
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


class GANDataset(Dataset):
    """Dataset for GAN training that outputs the conditional image from H5LatentsDataset without landcover map.
    
    This dataset reads from an HDF5 file containing latent codes, low frequency components and climate data,
    and outputs the conditional image tensor containing the low frequency and climate data.
    """
    def __init__(self, h5_file, crop_size, pct_land_ranges=None, subset_resolutions=None, 
                 subset_weights=None, subset_class_labels=None, split=None, eval_dataset=False,
                 clip_edges=False, use_climate=True):
        self.h5_file = h5_file
        self.crop_size = crop_size
        self.eval_dataset = eval_dataset
        self.clip_edges = clip_edges
        self.subset_weights = subset_weights or [1]
        self.subset_class_labels = subset_class_labels
        self.use_climate = use_climate
        
        if pct_land_ranges is None:
            pct_land_ranges = [[0, 1]]
        if subset_resolutions is None:
            subset_resolutions = [90]
            
        self.keys = [set() for _ in range(len(subset_resolutions))]
        
        with h5py.File(self.h5_file, 'r') as f:
            for i, (pct_land_range, res) in enumerate(zip(pct_land_ranges, subset_resolutions)):
                if pct_land_range is None:
                    pct_land_range = [0, 1]
                    
                if str(res) not in f:
                    continue
                    
                res_group = f[str(res)]
                for chunk_id in res_group:
                    chunk_group = res_group[chunk_id]
                    for subchunk_id in chunk_group:
                        subchunk_group = chunk_group[subchunk_id]
                        if 'latent' not in subchunk_group:
                            continue
                            
                        dset = subchunk_group['latent']
                        pct_land_valid = pct_land_range[0] <= dset.attrs['pct_land'] <= pct_land_range[1]
                        split_valid = split is None or dset.attrs['split'] == split
    
                        if pct_land_valid and split_valid:
                            self.keys[i].add((chunk_id, res, subchunk_id))
        self.keys = [list(keys) for keys in self.keys]

    def __len__(self):
        return max(len(keys) for keys in self.keys)

    def __getitem__(self, idx):
        LOWFREQ_MEAN = -31.4
        LOWFREQ_STD = 38.6
        
        # Draw a random subset based on subset weights
        subset_idx = random.choices(range(len(self.subset_weights)), weights=self.subset_weights, k=1)[0]
        index = random.randrange(len(self.keys[subset_idx]))
        
        chunk_id, res, subchunk_id = self.keys[subset_idx][index]
        with h5py.File(self.h5_file, 'r') as f:
            group_path = f"{res}/{chunk_id}/{subchunk_id}"
            data_lowfreq = f[f"{group_path}/lowfreq"]
            data_climate = f[f"{group_path}/climate"] if self.use_climate and 'climate' in f[group_path] else None
            
            shape = data_lowfreq.shape
            
            if not self.eval_dataset:
                if self.clip_edges:
                    i = random.randint(1, shape[0] - self.crop_size - 1)
                    j = random.randint(1, shape[1] - self.crop_size - 1)
                else:
                    i = random.randint(0, shape[0] - self.crop_size)
                    j = random.randint(0, shape[1] - self.crop_size)
            else:
                i = (shape[0] - self.crop_size) // 2
                j = (shape[1] - self.crop_size) // 2
                
            h = w = self.crop_size
            
            data_lowfreq = torch.from_numpy(data_lowfreq[i:i+h, j:j+w])[None]
            data_lowfreq = (data_lowfreq - LOWFREQ_MEAN) / LOWFREQ_STD
            
            if self.use_climate and data_climate is not None:
                data_mean_temp = data_climate[0, i:i+h, j:j+w]
                data_temp_seasonality = data_climate[3, i:i+h, j:j+w]
                data_annual_precip = data_climate[11, i:i+h, j:j+w]
                data_precip_seasonality = data_climate[14, i:i+h, j:j+w]
                
                # Create nan mask (True where any climate variable is nan)
                nan_mask = (np.isnan(data_mean_temp) | 
                          np.isnan(data_temp_seasonality) | 
                          np.isnan(data_annual_precip) | 
                          np.isnan(data_precip_seasonality))
                
                # Normalize climate data by subtracting mean and dividing by std
                climate_mean = f[str(res)].attrs['climate_mean']
                climate_std = f[str(res)].attrs['climate_std']
                
                data_mean_temp = torch.from_numpy(data_mean_temp - climate_mean[0]) / climate_std[0]
                data_temp_seasonality = torch.from_numpy(data_temp_seasonality - climate_mean[3]) / climate_std[3]
                data_annual_precip = torch.from_numpy(data_annual_precip - climate_mean[11]) / climate_std[11]
                data_precip_seasonality = torch.from_numpy(data_precip_seasonality - climate_mean[14]) / climate_std[14]
                
                climate_means = [
                    torch.nanmean(data_mean_temp),
                    torch.nanmean(data_temp_seasonality),
                    torch.nanmean(data_annual_precip),
                    torch.nanmean(data_precip_seasonality)
                ]
                climate_means = torch.as_tensor([torch.nan_to_num(mean, nan=0.0) for mean in climate_means])
            
            transform_idx = random.randrange(8) if not self.eval_dataset else 0
            flip = (transform_idx // 4) == 1
            rotate_k = transform_idx % 4
            
            # Apply transformations to all channels
            if flip:
                data_lowfreq = torch.flip(data_lowfreq, dims=[-1])
            if rotate_k != 0:
                data_lowfreq = torch.rot90(data_lowfreq, k=rotate_k, dims=[-2, -1])
                
            return {'image': data_lowfreq, 'additional_vars': climate_means}

