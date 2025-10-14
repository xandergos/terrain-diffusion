import random
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


class H5LatentsDataset(Dataset):
    """Dataset for training diffusion models to generate terrain latents."""
    def __init__(self, 
                 h5_file, 
                 crop_size, 
                 pct_land_ranges, 
                 subset_resolutions, 
                 subset_weights=None,
                 subset_class_labels=None,
                 eval_dataset=False,
                 latents_mean=None, 
                 latents_std=None, 
                 sigma_data=0.5, 
                 clip_edges=True,
                 split=None,
                 beauty_dist=None,
                 residual_mean=None,
                 residual_std=None,
                 watercover_mean=None,
                 watercover_std=None):
        """
        Args:
            h5_file (str): Path to the HDF5 file containing the dataset.
            crop_size (int): Size of the random crop to extract from each image.
            pct_land_ranges (list): Ranges of acceptable pct_land values [min, max], for each subset. Elements can be None to include everything.
            subset_resolutions (list): Resolutions to filter subsets by. Each subset will only include elevation data with the corresponding resolution. Elements be None to include all resolutions.
            subset_weights (list): Weights for each subset, determining the relative probability of sampling from that subset. Default is None, which results in uniform sampling.
            subset_class_labels (list): Class labels for each subset, determining the class of each subset. Defaults to None, which results in no class labels being returned.
            eval_dataset (bool, optional): Whether the dataset should be transformed deterministically. Defaults to False.
            latents_mean (list, optional): Mean values for normalizing latents. Defaults to None.
            latents_std (list, optional): Standard deviation values for normalizing latents. Defaults to None.
            sigma_data (float, optional): Data standard deviation. Defaults to 0.5.
            clip_edges (bool, optional): Whether to clip edges when cropping. Defaults to True.
            split (str, optional): Split to use. Defaults to None (all splits).
            beauty_dist (list, optional): Weights for sampling beauty scores 1-5. 
                                        Must sum to 1. Defaults to uniform distribution.
        """
        self.h5_file = h5_file
        self.crop_size = crop_size
        self.pct_land_ranges = pct_land_ranges or [[0, 1]]
        self.subset_resolutions = subset_resolutions or [480]
        self.subset_weights = subset_weights or [1.0]
        self.subset_class_labels = subset_class_labels
        self.latents_mean = torch.tensor(latents_mean).view(-1, 1, 1) if isinstance(latents_mean, list) else torch.clone(latents_mean).view(-1, 1, 1)
        self.latents_std = torch.tensor(latents_std).view(-1, 1, 1) if isinstance(latents_std, list) else torch.clone(latents_std).view(-1, 1, 1)
        self.sigma_data = sigma_data
        self.split = split
        self.eval_dataset = eval_dataset
        self.clip_edges = clip_edges
        self.beauty_dist = beauty_dist or None
        
        num_subsets = len(subset_weights)
        assert len(pct_land_ranges) == len(subset_resolutions) == num_subsets, \
            "Number of subsets must match between pct_land_ranges, dataset_resolutions, and subset_weights."
        assert subset_class_labels is None or len(subset_class_labels) == num_subsets, \
            "Number of subset class labels must match number of subsets."
        
        if beauty_dist is not None:
            self.keys = [[set() for _ in range(5)] for _ in range(num_subsets)]
        else:
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
                        if 'latent' not in subchunk_group:
                            continue
                            
                        dset = subchunk_group['latent']
                        pct_land_valid = pct_land_range[0] <= dset.attrs['pct_land'] <= pct_land_range[1]
                        split_valid = split is None or dset.attrs['split'] == split
    
                        if pct_land_valid and split_valid:
                            if beauty_dist is not None:
                                beauty_score = float(subchunk_group.attrs['beauty_score'])
                                beauty_score = max(1, min(5, round(beauty_score))) - 1
                                self.keys[i][beauty_score].add((chunk_id, res, subchunk_id))
                            else:
                                self.keys[i].add((chunk_id, res, subchunk_id))
        if beauty_dist is not None:
            self.keys = [[list(subkeys) for subkeys in keys] for keys in self.keys]
            print("Using beauty distribution. Have sizes:", [[len(subkeys) for subkeys in keys] for keys in self.keys])
        else:
            self.keys = [list(keys) for keys in self.keys]
            print("Not using beauty distribution.")
        
        self.residual_mean = residual_mean
        self.residual_std = residual_std
        self.watercover_mean = watercover_mean
        self.watercover_std = watercover_std

    def __len__(self):
        return 100000

    def augment_onehot(self, onehot):
        """Augments onehot in place by randomly adding classes to pixels."""
        # Get indices of present classes
        present_indices = torch.where(onehot.sum(dim=(-2,-1)) > 0)[0]
        
        # Randomly choose number of partitions (k)
        k = random.randint(1, len(present_indices))
        
        # Randomly shuffle and split present indices into k groups
        shuffled_indices = present_indices[torch.randperm(len(present_indices))]
        
        # Create random split sizes that sum to total length
        split_points = np.sort(np.random.choice(len(present_indices)-1, k-1, replace=False) + 1)
        split_sizes = np.diff(np.concatenate([[0], split_points, [len(present_indices)]]))
        
        # Create groups based on random split sizes
        start = 0
        index_groups = []
        for size in split_sizes:
            index_groups.append(shuffled_indices[start:start+size])
            start += size
        
        # For each group, perform separate augmentation
        for group_indices in index_groups:
            # Create mask for pixels belonging to any class in this group
            group_mask = onehot[group_indices].sum(dim=0) > 0
            
            # Set all classes in this group to 1 where group mask is True
            for idx in range(len(group_indices)):
                onehot[group_indices[idx]][group_mask] = 1
            
            # For classes in this group that aren't present in these pixels,
            # randomly decide whether to include additional ones
            zero_indices = torch.where(onehot[:].sum(dim=(-2,-1)) == 0)[0]
            if len(zero_indices) > 0:
                num_to_set = random.randint(0, len(zero_indices))
                indices_to_set = zero_indices[torch.randperm(len(zero_indices))[:num_to_set]]
                
                # Set selected classes to 1 where group mask is True
                for idx in indices_to_set:
                    onehot[idx][group_mask] = 1
                        
    def __getitem__(self, idx):
        LOWFREQ_MEAN = -31.4
        LOWFREQ_STD = 38.6
        
        # Draw a random subset based on subset weights
        subset_idx = random.choices(range(len(self.subset_weights)), weights=self.subset_weights, k=1)[0]
        class_label = self.subset_class_labels[subset_idx] if self.subset_class_labels is not None else None
        
        if self.beauty_dist is not None:
            beauty_score = random.choices(range(5), weights=self.beauty_dist[subset_idx], k=1)[0]
            index = random.randrange(len(self.keys[subset_idx][beauty_score]))
            chunk_id, res, subchunk_id = self.keys[subset_idx][beauty_score][index]
        else:
            index = random.randrange(len(self.keys[subset_idx]))
            chunk_id, res, subchunk_id = self.keys[subset_idx][index]
        
        with h5py.File(self.h5_file, 'r') as f:
            group_path = f"{res}/{chunk_id}/{subchunk_id}"
            res_group = f[str(res)]
            data_latent = f[f"{group_path}/latent"]
            data_lowfreq = f[f"{group_path}/lowfreq"]
            data_climate = f[f"{group_path}/climate"]
            
            shape = data_latent.shape
            assert data_lowfreq.shape == shape[2:]
            assert data_climate.shape[1:] == shape[2:]
            
            if self.clip_edges:
                assert shape[2] >= self.crop_size + 2, f"Crop size is larger than image size + 2. Crop size: {self.crop_size}, Image size: {shape[2]}"
            else:
                assert shape[2] >= self.crop_size, f"Crop size is larger than image size. Crop size: {self.crop_size}, Image size: {shape[2]}"
        
            if not self.eval_dataset:
                if self.clip_edges:
                    i = random.randint(1, shape[2] - self.crop_size - 1)
                    j = random.randint(1, shape[3] - self.crop_size - 1)
                else:
                    i = random.randint(0, shape[2] - self.crop_size)
                    j = random.randint(0, shape[3] - self.crop_size)
            else:
                i = (shape[2] - self.crop_size) // 2
                j = (shape[3] - self.crop_size) // 2
                
            h = w = self.crop_size
            li, lj, lh, lw = i, j, h, w
                
            transform_idx = random.randrange(8) if not self.eval_dataset else 0
            flip = (transform_idx // 4) == 1
            rotate_k = transform_idx % 4
            
            # Adjust lowfreq crop for flips and rotations. Note that we are inverting the transformation so we do it in reverse.
            for _ in range(rotate_k):
                li, lj = lj, shape[2] - li - lh
            if flip:
                lj = shape[2] - lj - lw
            
            # Load latent data
            data_latent = torch.from_numpy(data_latent[transform_idx, :, i:i+h, j:j+w])
            assert data_latent.shape[-1] > 0, f"Latent crop is empty. i: {i}, j: {j}, h: {h}, w: {w}"
            latent_channels = data_latent.shape[0]
            means, logvars = data_latent[:latent_channels//2], data_latent[latent_channels//2:]
            sampled_latent = torch.randn_like(means) * (logvars * 0.5).exp() + means
            sampled_latent = (sampled_latent - self.latents_mean) / self.latents_std * self.sigma_data
            
            # Load lowfreq data
            data_lowfreq = torch.from_numpy(data_lowfreq[li:li+lh, lj:lj+lw])[None]
            if flip:
                data_lowfreq = torch.flip(data_lowfreq, dims=[-1])
            if rotate_k != 0:
                data_lowfreq = torch.rot90(data_lowfreq, k=rotate_k, dims=[-2, -1])
            data_lowfreq = data_lowfreq.float()
            data_lowfreq = (data_lowfreq - LOWFREQ_MEAN) / LOWFREQ_STD * self.sigma_data
            lowfreq_mean = torch.mean(data_lowfreq) / self.sigma_data
            
            # Load climate data
            # (0=temp, 3=temp seasonality, 11=precip, 14=precip seasonality)
            if f"{group_path}/climate" in f:
                climate_data = f[f"{group_path}/climate"][[0, 3, 11, 14]][:, li:li+lh, lj:lj+lw]
                climate_data = (climate_data - res_group.attrs['climate_mean'][[0, 3, 11, 14], None, None]) / res_group.attrs['climate_std'][[0, 3, 11, 14], None, None] * self.sigma_data
                climate_data = torch.from_numpy(climate_data).float()
                if flip:
                    climate_data = torch.flip(climate_data, dims=[-1])
                if rotate_k != 0:
                    climate_data = torch.rot90(climate_data, k=rotate_k, dims=[-2, -1])
                any_nan_climate = torch.isnan(climate_data).all(dim=(-2, -1)).any().item()
                if any_nan_climate:
                    climate_nanmean = torch.zeros(4)
                else:
                    climate_nanmean = torch.nanmean(climate_data, dim=(-2, -1))
                climate_data_mask = torch.isnan(climate_data).any(dim=0).float().unsqueeze(0) * np.sqrt(2) * self.sigma_data
                climate_data = torch.nan_to_num(climate_data, nan=0.0)
            else:
                climate_data = torch.zeros(4, lh, lw)
                climate_data_mask = torch.zeros(1, lh, lw)
                climate_data_mask[0] = np.sqrt(2) * self.sigma_data
                climate_nanmean = torch.zeros(4)
                any_nan_climate = True
            
        image = torch.cat([
            sampled_latent,
            data_lowfreq,
            climate_data,
            climate_data_mask
        ], dim=0)
        
        # Only include sampled_latent in img tensor instead of concatenating with other data
        cond_inputs = [lowfreq_mean.reshape([]).float(),
                       climate_nanmean[0].float().reshape([]),
                       climate_nanmean[1].float().reshape([]),
                       climate_nanmean[2].float().reshape([]),
                       climate_nanmean[3].float().reshape([]),
                       torch.tensor(1 if any_nan_climate else 0, dtype=torch.int64).reshape([])]
        if class_label is not None:
            cond_inputs += [torch.tensor(class_label)]
            
        return {'image': image.float(), 'cond_inputs': cond_inputs, 'path': group_path}

    def denormalize_climate(self, climate, resolution):
        """Denormalize climate data to original scale. Climate data should have shape [B, 4, H, W]"""
        with h5py.File(self.h5_file, 'r') as f:
            res_group = f[str(resolution)]
            return climate * res_group.attrs['climate_std'][None, [0, 3, 11, 14], None, None] + res_group.attrs['climate_mean'][None, [0, 3, 11, 14], None, None]

    def denormalize_residual(self, residual):
        return residual * self.residual_std + self.residual_mean

    def denormalize_watercover(self, watercover):
        return watercover * self.watercover_std + self.watercover_mean
    
    def denormalize_lowfreq(self, lowfreq, resolution=None):
        LOWFREQ_MEAN = -31.4
        LOWFREQ_STD = 38.6
        return lowfreq * LOWFREQ_STD + LOWFREQ_MEAN

