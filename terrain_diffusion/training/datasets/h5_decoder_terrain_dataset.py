import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import h5py
import torch.nn.functional as F


class H5DecoderTerrainDataset(Dataset):
    """Dataset for reading terrain data from an HDF5 file with an encoding."""
    def __init__(self, 
                 h5_file, 
                 crop_size, 
                 pct_land_ranges, 
                 subset_resolutions, 
                 subset_weights=None,
                 subset_class_labels=None,
                 eval_dataset=False,
                 clip_edges=True,
                 split=None,
                 residual_mean=None,
                 residual_std=None,
                 sigma_data=0.5):
        """
        Args:
            h5_file (str): Path to the HDF5 file containing the dataset.
            crop_size (int): Size of the random crop to extract from each image.
            pct_land_ranges (list): Ranges of acceptable pct_land values [min, max], for each subset.
            subset_resolutions (list): Resolutions to filter subsets by.
            subset_weights (list): Weights for each subset. Default is None (uniform sampling).
            subset_class_labels (list): Class labels for each subset. Defaults to None.
            eval_dataset (bool): Whether to use deterministic transforms. Defaults to False.
            clip_edges (bool): Whether to clip edges when cropping. Defaults to True.
            split (str): Split to use. Defaults to None (all splits).
            residual_mean (float): Mean for residual normalization. Defaults to None (will compute).
            residual_std (float): Std for residual normalization. Defaults to None (will compute).
            sigma_data (float): Data standard deviation. Defaults to 0.5.
        """
        if subset_weights is None:
            subset_weights = [1] * len(pct_land_ranges)
        self.h5_file = h5_file
        self.crop_size = crop_size
        self.pct_land_ranges = pct_land_ranges
        self.subset_resolutions = subset_resolutions
        self.subset_weights = subset_weights
        self.subset_class_labels = subset_class_labels
        self.eval_dataset = eval_dataset
        self.clip_edges = clip_edges
        self.sigma_data = sigma_data
        
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
            
        assert crop_size % 8 == 0, "Crop size must be divisible by 8"
        
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
            sample_idx = torch.randint(len(self), (1,), generator=self.rng).item()
            sample = self.__getitem__(sample_idx)
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
        
        self.residual_mean = running_mean[0]
        self.residual_std = std[0]

        torch.set_grad_enabled(True)

    def __len__(self):
        return max(len(keys) for keys in self.keys)
    
    def set_seed(self, seed):
        self.rng = torch.Generator()
        self.rng.manual_seed(int(seed))

    def __getitem__(self, index):
        # Draw a random subset based on subset weights
        subset_weights_tensor = torch.tensor(self.subset_weights, dtype=torch.float32)
        subset_idx = torch.multinomial(subset_weights_tensor, 1, generator=self.rng).item()
        class_label = self.subset_class_labels[subset_idx] if self.subset_class_labels is not None else None
        
        index = torch.randint(len(self.keys[subset_idx]), (1,), generator=self.rng).item()
        chunk_id, res, subchunk_id = self.keys[subset_idx][index]
        
        with h5py.File(self.h5_file, 'r') as f:
            group_path = f"{res}/{chunk_id}/{subchunk_id}"
            data_latent = f[f"{group_path}/latent"]
            
            latent_shape = data_latent.shape
            residual_shape = f[f"{group_path}/residual"].shape
            
            if self.clip_edges:
                assert residual_shape[1] >= self.crop_size + 2, f"Crop size is larger than image size + 2. Crop size: {self.crop_size}, Image size: {residual_shape[2]}"
            else:
                assert residual_shape[1] >= self.crop_size, f"Crop size is larger than image size. Crop size: {self.crop_size}, Image size: {residual_shape[2]}"
        
            if not self.eval_dataset:
                if self.clip_edges:
                    i = torch.randint(1, (latent_shape[2] - self.crop_size // 8 - 1) + 1, (1,), generator=self.rng).item()
                    j = torch.randint(1, (latent_shape[3] - self.crop_size // 8 - 1) + 1, (1,), generator=self.rng).item()
                else:
                    i = torch.randint(0, (latent_shape[2] - self.crop_size // 8) + 1, (1,), generator=self.rng).item()
                    j = torch.randint(0, (latent_shape[3] - self.crop_size // 8) + 1, (1,), generator=self.rng).item()
            else:
                i = (latent_shape[2] - self.crop_size // 8) // 2
                j = (latent_shape[3] - self.crop_size // 8) // 2
                
            h = w = self.crop_size // 8
            li, lj, lh, lw = i*8, j*8, h*8, w*8
                
            transform_idx = torch.randint(8, (1,), generator=self.rng).item() if not self.eval_dataset else 0
            flip = (transform_idx // 4) == 1
            rotate_k = transform_idx % 4
            
            # Adjust lowfreq crop for flips and rotations. Note that we are inverting the transformation so we do it in reverse.
            for _ in range(rotate_k):
                li, lj = lj, residual_shape[1] - li - lh
            if flip:
                lj = residual_shape[1] - lj - lw
            
            # Load latent data
            data_latent = torch.from_numpy(data_latent[transform_idx, :, i:i+h, j:j+w])
            assert data_latent.shape[-1] > 0, f"Latent crop is empty. i: {i}, j: {j}, h: {h}, w: {w}"
            latent_channels = data_latent.shape[0]
            means, logvars = data_latent[:latent_channels//2], data_latent[latent_channels//2:]
            noise = torch.randn(means.shape, generator=self.rng, device=means.device, dtype=means.dtype)
            sampled_latent = noise * (logvars * 0.5).exp() + means
            
            # Load residual data
            data_residual = torch.from_numpy(f[f"{group_path}/residual"][li:li+lh, lj:lj+lw])[None]
            data_residual = (data_residual - self.residual_mean) / self.residual_std
            if flip:
                data_residual = torch.flip(data_residual, dims=[-1])
            if rotate_k != 0:
                data_residual = torch.rot90(data_residual, k=rotate_k, dims=[-2, -1])
                
        image = data_residual * self.sigma_data
        
        cond_image = F.interpolate(sampled_latent[None], size=(self.crop_size, self.crop_size), mode='nearest')[0]
        
        if class_label is not None:
            cond_inputs = [torch.tensor(class_label)]
        else:
            cond_inputs = []
            
        return {'image': image.float(), 'cond_img': cond_image, 'cond_inputs': cond_inputs, 'path': group_path}
        
    def denormalize_residual(self, residual):
        return (residual * self.residual_std) + self.residual_mean

