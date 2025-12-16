import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import h5py


class H5AutoencoderDataset(Dataset):
    """Dataset for reading high frequency terrain data from an HDF5 file."""
    def __init__(self, 
                 h5_file, 
                 crop_size, 
                 pct_land_ranges, 
                 subset_resolutions, 
                 subset_weights=None,
                 subset_class_labels=None,
                 split=None,
                 residual_mean=None,
                 residual_std=None,
                 signed_sqrt=True):
        """
        Args:
            h5_file (str): Path to the HDF5 file containing the dataset.
            crop_size (int): Size of the random crop to extract from each image.
            pct_land_ranges (list): Ranges of acceptable pct_land values [min, max], for each subset.
            subset_resolutions (list): Resolutions to filter subsets by.
            subset_weights (list): Weights for each subset. Default is None (uniform sampling).
            subset_class_labels (list): Class labels for each subset. Defaults to None.
            split (str): Split to use. Defaults to None (all splits).
            sigma_data (float): Data standard deviation. Defaults to 0.5.
            signed_sqrt (bool): Whether to use signed sqrt transform. Defaults to True. If False, the transform is inverted at runtime.
        """
        if subset_weights is None:
            subset_weights = [1] * len(pct_land_ranges)
        self.h5_file = h5_file
        self.crop_size = crop_size
        self.pct_land_ranges = pct_land_ranges
        self.subset_resolutions = subset_resolutions
        self.subset_weights = subset_weights
        self.subset_class_labels = subset_class_labels
        self.signed_sqrt = signed_sqrt
        
        # Initialize keys
        num_subsets = len(subset_weights)
        assert len(pct_land_ranges) == len(subset_resolutions) == num_subsets, \
            "Number of subsets must match between pct_land_ranges, dataset_resolutions, and subset_weights."
        assert subset_class_labels is None or len(subset_class_labels) == num_subsets, \
            "Number of subset class labels must match number of subsets."
        
        self.keys = [set() for _ in range(num_subsets)]
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
                        if 'residual' not in subchunk_group:
                            continue
                            
                        dset = subchunk_group['residual']
                        pct_land_valid = pct_land_range[0] <= dset.attrs['pct_land'] <= pct_land_range[1]
                        split_valid = split is None or dset.attrs['split'] == split
    
                        if pct_land_valid and split_valid:
                            self.keys[i].add((chunk_id, res, subchunk_id))
        self.keys = [list(keys) for keys in self.keys]
        
        self.residual_mean = residual_mean
        self.residual_std = residual_std
        self.rng = torch.Generator()
        if self.residual_mean is None or self.residual_std is None:
            self.calculate_stats()
        
        
    def calculate_stats(self, num_samples=10000):
        """Compute per-channel mean and std using a streaming Welford algorithm.

        This avoids stacking samples in memory and works for any number of channels
        returned by __getitem__ under key 'image' with shape [C, H, W].
        """
        torch.set_grad_enabled(False)

        running_count = None
        running_mean = None
        running_m2 = None

        for _ in tqdm(range(num_samples), desc="Calculating stats"):
            sample = self.__getitem__(torch.randint(len(self), (1,), generator=self.rng).item())
            x = sample['image']
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)
                
            c, h, w = x.shape
            x_flat = x.reshape(c, -1)
            batch_count = x_flat.shape[1]

            # Per-channel batch stats
            batch_mean = x_flat.mean(dim=1)
            batch_m2 = x_flat.var(dim=1, unbiased=False) * batch_count

            if running_count is None:
                running_count = torch.full_like(batch_mean, fill_value=batch_count, dtype=torch.float32)
                running_mean = batch_mean.clone().to(torch.float32)
                running_m2 = batch_m2.clone().to(torch.float32)
                continue

            # Combine current running aggregates with this batch (per-channel)
            total_count = running_count + batch_count
            delta = batch_mean - running_mean
            running_mean = running_mean + delta * (batch_count / total_count)
            running_m2 = running_m2 + batch_m2 + (delta.pow(2) * running_count * batch_count / total_count)
            running_count = total_count

        # Finalize
        variance = running_m2 / running_count
        std = variance.sqrt()

        # Print concise results per channel
        for ch, (m, s) in enumerate(zip(running_mean.tolist(), std.tolist())):
            print(f"Channel {ch}: mean={m:.6f}, std={s:.6f}")
        
        self.channel_means = running_mean
        self.channel_stds = std

        torch.set_grad_enabled(True)

    def __len__(self):
        return max(len(keys) for keys in self.keys)
    
    def set_seed(self, seed):
        self.rng = torch.Generator()
        self.rng.manual_seed(int(seed))

    def __getitem__(self, index):
        subset_weights_tensor = torch.tensor(self.subset_weights, dtype=torch.float32)
        subset_idx = torch.multinomial(subset_weights_tensor, 1, generator=self.rng).item()
        index = torch.randint(len(self.keys[subset_idx]), (1,), generator=self.rng).item()
        class_label = self.subset_class_labels[subset_idx] if self.subset_class_labels is not None else None
        
        chunk_id, res, subchunk_id = self.keys[subset_idx][index]
        with h5py.File(self.h5_file, 'r') as f:
            group_path = f"{res}/{chunk_id}/{subchunk_id}"
            
            # Get crop indices
            dset = f[f"{group_path}/residual"]
            data_shape = dset.shape
            
            effective_crop_size = self.crop_size
            if not self.signed_sqrt:
                effective_crop_size += 80
            
            # Ensure i and j are multiples of 8 by first dividing by 8, then multiplying back
            max_i = (data_shape[-2] - effective_crop_size) // 8
            max_j = (data_shape[-1] - effective_crop_size) // 8
            i = torch.randint(0, max_i + 1, (1,), generator=self.rng).item() * 8
            j = torch.randint(0, max_j + 1, (1,), generator=self.rng).item() * 8
            h, w = effective_crop_size, effective_crop_size
            
            # Load residual data (elevation)
            residual_data = torch.from_numpy(dset[i:i+h, j:j+w])[None]
            
            if not self.signed_sqrt:
                from terrain_diffusion.data.laplacian_encoder import laplacian_decode, laplacian_encode
                lowfreq_data = torch.from_numpy(f[f"{group_path}/lowfreq"][i//8:i//8+h//8, j//8:j//8+w//8])[None]
                full_terrain = laplacian_decode(residual_data, lowfreq_data)
                full_terrain = np.sign(full_terrain) * np.abs(full_terrain)**2
                residual_data, _ = laplacian_encode(full_terrain, effective_crop_size//8, sigma=5)
                
                # Central crop residual_data to self.crop_size
                crop_h, crop_w = residual_data.shape[-2], residual_data.shape[-1]
                target_h, target_w = self.crop_size, self.crop_size
                start_i = (crop_h - target_h) // 2
                start_j = (crop_w - target_w) // 2
                residual_data = residual_data[..., start_i:start_i+target_h, start_j:start_j+target_w]
            
            if self.residual_mean is not None:
                residual_data = (residual_data - self.residual_mean) / self.residual_std
            
            data = residual_data.float()
            
        # Apply transforms
        transform_idx = torch.randint(8, (1,), generator=self.rng).item()
        flip = (transform_idx // 4) == 1
        rotate_k = transform_idx % 4
            
        if flip:
            data = torch.flip(data, dims=[-1])
        if rotate_k != 0:
            data = torch.rot90(data, k=rotate_k, dims=[-2, -1])
            
        if class_label is not None:
            return {'image': data, 'cond_inputs': [torch.tensor(class_label)]}
        else:
            return {'image': data}
    
    def denormalize_residual(self, residual):
        return (residual * self.residual_std) + self.residual_mean

