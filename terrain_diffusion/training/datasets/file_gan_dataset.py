import random
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio


class FileGANDataset(Dataset):
    """
    A PyTorch dataset class that returns random crops from climate/elevation rasters.
    
    Args:
        dataset_names (list): List of dataset names in desired stacking order
        crop_size (tuple): Size of the crop (height, width)
        resize_size (tuple): Size to resize the crop to
        data_dir (str): Directory containing the TIF files
        
    Attributes:
        data (numpy.ndarray): Array of shape (n_datasets, height, width) containing stacked data
        available_datasets (list): List of available dataset names
    """
    def __init__(self, dataset_names, crop_size=(32, 32), resize_size=(32, 32), 
                 filter_threshold=None, filter_pct=0.5, max_filter_tries=50):
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.filter_threshold = filter_threshold
        self.filter_pct = float(filter_pct)
        self.max_filter_tries = int(max_filter_tries)
        
        data_arrays = []
        for tif_file in dataset_names:
            with rasterio.open(tif_file) as src:
                # Get the data bounds
                bounds = src.bounds
                
                # Calculate row indices for -60 to 60 degrees latitude
                height = src.height
                lat_res = (bounds.top - bounds.bottom) / height
                start_row = int((bounds.top - 60) / lat_res)
                end_row = int((bounds.top + 60) / lat_res)
                
                # Read the data within latitude bounds
                data = src.read(1)[start_row:end_row, :]
                if data.dtype == np.int16:
                    data = data.astype(np.float32)
                    data[data == -32768] = np.nan
                else:
                    data[np.abs(data) > 1e6] = np.nan
                    
                # Normalize the data
                u, s = np.nanmean(data), np.nanstd(data)
                print(tif_file, u, s)
                data[np.isnan(data)] = u
                data = (data - u) / s
                
                # Get filename without extension and path for dictionary key
                data_arrays.append(data)
            
        # Get data arrays in requested order and stack them
        self.data = np.stack(data_arrays, axis=0)  # Shape: (n_datasets, height, width)
        
        # Store valid dimensions for random cropping
        self.height, self.width = self.data.shape[1:]
        
    def __len__(self):
        """Return a fixed length (can be adjusted based on needs)"""
        return 10000  # Arbitrary number of samples per epoch
        
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def __getitem__(self, idx):
        """
        Get a random crop from the stacked datasets.
        
        Args:
            idx (int): Index (not used since we're returning random crops)
            
        Returns:
            torch.Tensor: Tensor of shape (n_datasets, crop_height, crop_width)
        """
        # Calculate valid ranges for the top-left corner of the crop
        max_h = self.height - self.crop_size[0]
        max_w = self.width - self.crop_size[1]
        
        # Choose target side of threshold if requested
        if self.filter_threshold is not None:
            want_above = (np.random.random() < self.filter_pct)
            h_start = w_start = 0
            found = False
            for _ in range(self.max_filter_tries):
                h_start = np.random.randint(0, max_h + 1)
                w_start = np.random.randint(0, max_w + 1)
                first_mean = self.data[0, 
                                       h_start:h_start + self.crop_size[0],
                                       w_start:w_start + self.crop_size[1]].mean()
                if (first_mean > self.filter_threshold) == want_above:
                    found = True
                    break
            if not found:
                # fallback to a random crop if condition not found
                h_start = np.random.randint(0, max_h + 1)
                w_start = np.random.randint(0, max_w + 1)
        else:
            # Generate random crop coordinates
            h_start = np.random.randint(0, max_h + 1)
            w_start = np.random.randint(0, max_w + 1)
        
        # Extract the crop
        crop = self.data[:, 
                        h_start:h_start + self.crop_size[0],
                        w_start:w_start + self.crop_size[1]]
        
        crop = torch.from_numpy(crop).float()
        # Randomly flip horizontally and vertically
        if np.random.random() < 0.5:
            crop = torch.flip(crop, dims=[-1])
            
        # Randomly rotate by multiples of 90 degrees
        k = np.random.randint(4)  # Number of 90 degree rotations
        if k > 0:
            crop = torch.rot90(crop, k=k, dims=(-2, -1))
        
        
        if self.resize_size is not None:
            crop = torch.nn.functional.interpolate(
                crop.unsqueeze(0),
                size=self.resize_size,
                mode='area',
                antialias=False
            ).squeeze(0)
            return {'image': crop}
        
        return {'image': torch.from_numpy(crop).float()}

