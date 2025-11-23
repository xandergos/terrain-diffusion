import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

from terrain_diffusion.data.laplacian_encoder import laplacian_decode
from terrain_diffusion.models.mp_layers import mp_concat


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
                 beauty_dist=True,
                 residual_mean=None,
                 residual_std=None,
                 cond_input_mean=None,
                 cond_input_std=None,
                 cond_input_dropout=0.0,
                 cond_input_max_noise=0.0,
                 val_dset=False):
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
            beauty_dist (list[bool], optional): Whether to use random beauty distribution for each subset. Defaults to [True] * len(subset_weights).
            residual_mean (float, optional): Mean for residual normalization. Defaults to None.
            residual_std (float, optional): Std for residual normalization. Defaults to None.
            cond_input_mean (list, optional): Mean values for normalizing cond inputs. Defaults to None (will compute).
            cond_input_std (list, optional): Standard deviation values for normalizing cond inputs. Defaults to None (will compute).
            cond_input_dropout (float, optional): Probability of dropping out cond input pixels. Defaults to 0.0.
            cond_input_max_noise (float, optional): Maximum noise for cond input pixels. Defaults to 0.0 (no noise).
            val_dataset (bool, optional): Whether to include the ground truth in the dataset, and zero noise and dropout. Defaults to False.
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
        self.beauty_dist = beauty_dist or [True] * len(subset_weights)
        self.val_dset = val_dset
        self.cond_input_dropout = cond_input_dropout
        self.cond_input_max_noise = cond_input_max_noise
        self.rng = torch.Generator()
        
        num_subsets = len(subset_weights)
        assert len(pct_land_ranges) == len(subset_resolutions) == num_subsets, \
            "Number of subsets must match between pct_land_ranges, dataset_resolutions, and subset_weights."
        assert subset_class_labels is None or len(subset_class_labels) == num_subsets, \
            "Number of subset class labels must match number of subsets."
        
        if self.beauty_dist != [False] * len(subset_weights):
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
                            if self.beauty_dist[i]:
                                beauty_score = float(subchunk_group.attrs['beauty_score'])
                                beauty_score = max(1, min(5, round(beauty_score))) - 1
                                self.keys[i][beauty_score].add((chunk_id, res, subchunk_id))
                            else:
                                self.keys[i][0].add((chunk_id, res, subchunk_id))
        if self.beauty_dist != [False] * len(subset_weights):
            self.keys = [[sorted(list(subkeys)) for subkeys in keys] for keys in self.keys]
            print("Using beauty distribution. Have sizes:", [[len(subkeys) for subkeys in keys] for keys in self.keys])
        else:
            self.keys = [sorted(list(keys)) for keys in self.keys]
            print("Not using beauty distribution.")
        
        self.residual_mean = residual_mean
        self.residual_std = residual_std
        
        self.cond_input_mean = cond_input_mean or None
        self.cond_input_std = cond_input_std or None
        if self.cond_input_mean is None:
            self.compute_cond_input_stats()
    
    def compute_cond_input_stats(self):
        values = []
        for i in range(1024):
            elem = self.__getitem__(i, _raw_cond_inputs=True)
            values.append(elem['cond_inputs_img'])
        values = torch.stack(values).numpy()
        self.cond_input_mean = [np.nanmean(values[:, :1]), np.nanmean(values[:, 1:2]), np.nanmean(values[:, 2:3]), 
                                np.nanmean(values[:, 3:4]), np.nanmean(values[:, 4:5]), np.nanmean(values[:, 5:6]),
                                np.nanmean(values[:, 6:7])]
        values[:, :1] = np.nan_to_num(values[:, :1], self.cond_input_mean[0])
        values[:, 1:2] = np.nan_to_num(values[:, 1:2], self.cond_input_mean[1])
        self.cond_input_std = [np.std(values[:, :1]), np.std(values[:, 1:2]), np.nanstd(values[:, 2:3]), 
                               np.nanstd(values[:, 3:4]), np.nanstd(values[:, 4:5]), np.nanstd(values[:, 5:6]),
                               np.nanstd(values[:, 6:7])]
        print(f"Computed cond input mean: {self.cond_input_mean}, std: {self.cond_input_std}")

    def __len__(self):
        return 100000
    
    def set_seed(self, seed):
        self.rng = torch.Generator()
        self.rng.manual_seed(int(seed))
        
    def _get_cond_image(self, f, group_path, li, lj, lh, lw, flip, rotate_k, raw=False):
        HALO = 32
        data_lowres_exact = f[f"{group_path}/lowres_exact"]  # same shape as data_lowfreq
        H, W = data_lowres_exact.shape

        # Source window (pre-transform coords)
        si0 = li - HALO
        sj0 = lj - HALO
        si1 = li + lh + HALO
        sj1 = lj + lw + HALO

        # Clip to bounds to avoid loading the whole dataset
        ri0 = max(0, si0)
        rj0 = max(0, sj0)
        ri1 = min(H, si1)
        rj1 = min(W, sj1)

        # Allocate output and place the read chunk
        out = np.full((lh + 2*HALO, lw + 2*HALO), np.nan, dtype=np.float32)
        if ri1 > ri0 and rj1 > rj0:
            di0 = ri0 - si0
            dj0 = rj0 - sj0
            chunk = data_lowres_exact[ri0:ri1, rj0:rj1]  # only load what we need
            chunk = chunk
            out[di0:di0 + (ri1 - ri0), dj0:dj0 + (rj1 - rj0)] = chunk
            
        out_climate = np.full((4, lh + 2*HALO, lw + 2*HALO), np.nan, dtype=np.float32)
        if ri1 > ri0 and rj1 > rj0:
            di0 = ri0 - si0
            dj0 = rj0 - sj0
            chunk = f[f"{group_path}/climate"][[0, 3, 11, 14], ri0:ri1, rj0:rj1]
            chunk = chunk
            out_climate[:, di0:di0 + (ri1 - ri0), dj0:dj0 + (rj1 - rj0)] = chunk

        lowres_exact = torch.from_numpy(out)[None]  # [1, crop_size+64, crop_size+64]
        climate = torch.from_numpy(out_climate)[None]  # [1, 4, crop_size+64, crop_size+64]

        # Apply same forward transforms as lowfreq to keep alignment
        if flip:
            lowres_exact = torch.flip(lowres_exact, dims=[-1])
            climate = torch.flip(climate, dims=[-1])
        if rotate_k != 0:
            lowres_exact = torch.rot90(lowres_exact, k=rotate_k, dims=[-2, -1])
            climate = torch.rot90(climate, k=rotate_k, dims=[-2, -1])
            
        out_width = (lw+2*HALO) // HALO
        out_height = (lh+2*HALO) // HALO
          
        means = lowres_exact.reshape(1, out_height, HALO, out_width, HALO).mean(dim=(2, 4))
        climate_means = climate.reshape(4, out_height, HALO, out_width, HALO).mean(dim=(2, 4))
        p5 = torch.from_numpy(np.quantile(lowres_exact.reshape(1, out_height, HALO, out_width, HALO).numpy(), 0.05, axis=(2, 4)))
        
        mask = 1 - torch.isnan(means).float()
        
        if self.cond_input_dropout and not self.val_dset:
            mask = mask * (torch.rand(mask.shape, generator=self.rng) > self.cond_input_dropout)
            means[mask == 0] = np.nan
            p5[mask == 0] = np.nan
            
        if self.cond_input_max_noise and not self.val_dset:
            noise_level = torch.rand(1, generator=self.rng)
            noise_std = noise_level * self.cond_input_max_noise
            means = means + torch.randn(means.shape, generator=self.rng) * noise_std
            p5 = p5 + torch.randn(p5.shape, generator=self.rng) * noise_std
        else:
            noise_level = torch.tensor(0.0)

        if self.cond_input_mean is not None:
            means = means.nan_to_num(self.cond_input_mean[0])
            p5 = p5.nan_to_num(self.cond_input_mean[1])
            stack = torch.cat([means, p5, climate_means, mask], dim=0)
            stack = (stack - torch.tensor(self.cond_input_mean).view(-1, 1, 1)) / torch.tensor(self.cond_input_std).view(-1, 1, 1)
        else:
            stack = torch.cat([means, p5, climate_means, mask], dim=0)
        
        return stack, noise_level
    
    def build_cond_inputs(self, cond_img, histogram_raw, noise_level):
        noise_level = (noise_level - 0.5) * np.sqrt(12)
        
        center_h = cond_img.shape[-2] // 2
        center_w = cond_img.shape[-1] // 2
        crop_h_start = center_h - 2
        crop_h_end = center_h + 2
        crop_w_start = center_w - 2
        crop_w_end = center_w + 2
        
        means_crop = cond_img[0:1, crop_h_start:crop_h_end, crop_w_start:crop_w_end]
        p5_crop = cond_img[1:2, crop_h_start:crop_h_end, crop_w_start:crop_w_end]
        climate_means_crop = cond_img[2:6, crop_h_start+1:crop_h_end-1, crop_w_start+1:crop_w_end-1].mean(dim=(1, 2))
        mask_crop = cond_img[6:7, crop_h_start:crop_h_end, crop_w_start:crop_w_end]
        
        nan_mask = torch.isnan(climate_means_crop)
        climate_means_crop[nan_mask] = torch.randn(climate_means_crop[nan_mask].shape, generator=self.rng)
        
        return mp_concat([means_crop.flatten(), p5_crop.flatten(), climate_means_crop.flatten(), mask_crop.flatten(), histogram_raw, noise_level.view(1)], dim=0)
        
    def __getitem__(self, idx, _raw_cond_inputs=False):
        LOWFREQ_MEAN = -31.4
        LOWFREQ_STD = 38.6
        
        # Draw a random subset based on subset weights
        subset_weights_tensor = torch.tensor(self.subset_weights, dtype=torch.float32)
        subset_idx = torch.multinomial(subset_weights_tensor, 1, generator=self.rng).item()
        class_label = self.subset_class_labels[subset_idx] if self.subset_class_labels is not None else None
        
        if self.beauty_dist[subset_idx]:
            subset_lens = torch.tensor([len(self.keys[subset_idx][i]) for i in range(5)]).float()
            baseline_probs = torch.log(subset_lens / subset_lens.sum())
            histogram_raw = torch.randn(5, generator=self.rng) if not self.val_dset else torch.zeros(5)
            histogram = torch.softmax(histogram_raw + baseline_probs, dim=0)
            beauty_score = torch.multinomial(histogram, 1, generator=self.rng).item()
            index = torch.randint(len(self.keys[subset_idx][beauty_score]), (1,), generator=self.rng).item()
            chunk_id, res, subchunk_id = self.keys[subset_idx][beauty_score][index]
        else:
            histogram_raw = torch.randn(5, generator=self.rng)
            index = torch.randint(len(self.keys[subset_idx][0]), (1,), generator=self.rng).item()
            chunk_id, res, subchunk_id = self.keys[subset_idx][0][index]
        
        with h5py.File(self.h5_file, 'r') as f:
            group_path = f"{res}/{chunk_id}/{subchunk_id}"
            data_latent = f[f"{group_path}/latent"]
            data_lowfreq = f[f"{group_path}/lowfreq"]
            
            shape = data_latent.shape
            assert data_lowfreq.shape == shape[2:]
            
            if self.clip_edges:
                assert shape[2] >= self.crop_size + 2, f"Crop size is larger than image size + 2. Crop size: {self.crop_size}, Image size: {shape[2]}"
            else:
                assert shape[2] >= self.crop_size, f"Crop size is larger than image size. Crop size: {self.crop_size}, Image size: {shape[2]}"
        
            if not self.eval_dataset:
                if self.clip_edges:
                    i = torch.randint(1, shape[2] - self.crop_size, (1,), generator=self.rng).item()
                    j = torch.randint(1, shape[3] - self.crop_size, (1,), generator=self.rng).item()
                else:
                    i = torch.randint(0, shape[2] - self.crop_size + 1, (1,), generator=self.rng).item()
                    j = torch.randint(0, shape[3] - self.crop_size + 1, (1,), generator=self.rng).item()
            else:
                i = (shape[2] - self.crop_size) // 2
                j = (shape[3] - self.crop_size) // 2
                
            h = w = self.crop_size
            li, lj, lh, lw = i, j, h, w
                
            transform_idx = torch.randint(8, (1,), generator=self.rng).item() if not self.eval_dataset else 0
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
            sampled_latent = torch.randn(means.shape, generator=self.rng) * (logvars * 0.5).exp() + means
            sampled_latent = (sampled_latent - self.latents_mean) / self.latents_std * self.sigma_data
            
            # Load lowfreq data
            if not self.clip_edges:
                data_lowfreq = torch.from_numpy(data_lowfreq[li:li+lh, lj:lj+lw])[None]
            else:
                data_lowfreq = torch.from_numpy(data_lowfreq[li-1:li+lh+1, lj-1:lj+lw+1])[None]
            if flip:
                data_lowfreq = torch.flip(data_lowfreq, dims=[-1])
            if rotate_k != 0:
                data_lowfreq = torch.rot90(data_lowfreq, k=rotate_k, dims=[-2, -1])
            if self.clip_edges:
                lowfreq_padded = data_lowfreq
                data_lowfreq = data_lowfreq[..., 1:-1, 1:-1]
            else:
                lowfreq_padded = None
            data_lowfreq = data_lowfreq.float()
            data_lowfreq = (data_lowfreq - LOWFREQ_MEAN) / LOWFREQ_STD * self.sigma_data
            
            if self.val_dset:
                data_residual = torch.from_numpy(f[f"{group_path}/residual"][li*8:li*8+lh*8, lj*8:lj*8+lw*8])[None]
                data_residual = data_residual.float()
                if flip:
                    data_residual = torch.flip(data_residual, dims=[-1])
                if rotate_k != 0:
                    data_residual = torch.rot90(data_residual, k=rotate_k, dims=[-2, -1])
                if self.clip_edges:
                    ground_truth = laplacian_decode(data_residual, lowfreq_padded, pre_padded=True)
                else:
                    ground_truth = laplacian_decode(data_residual, self.denormalize_lowfreq(data_lowfreq / self.sigma_data), extrapolate=True)
            
            cond_tensor_img, noise_level = self._get_cond_image(f, group_path, li, lj, lh, lw, flip, rotate_k)
            cond_tensor = self.build_cond_inputs(cond_tensor_img, histogram_raw, noise_level)
            
        image = torch.cat([
            sampled_latent,
            data_lowfreq
        ], dim=0)
        
        cond_inputs = [cond_tensor.float()]
        if class_label is not None:
            cond_inputs += [torch.tensor(class_label)]
        
        out = {'image': image.float(), 'cond_inputs': cond_inputs, 'cond_inputs_img': cond_tensor_img, 'path': group_path,
               'histogram_raw': histogram_raw, 'noise_level': noise_level}
        if self.val_dset:
            out['ground_truth'] = ground_truth
        return out

    def denormalize_residual(self, residual):
        return residual * self.residual_std + self.residual_mean
    
    def denormalize_lowfreq(self, lowfreq):
        LOWFREQ_MEAN = -31.4
        LOWFREQ_STD = 38.6
        return lowfreq * LOWFREQ_STD + LOWFREQ_MEAN

if __name__ == "__main__":
    params = {
        "h5_file": "data/dataset.h5",
        "crop_size": 96,
        "pct_land_ranges": [[0, 0.01], [0.01, 1]],
        "subset_resolutions": [90, 90],
        "subset_weights": [0.01, 0.99],
        "latents_mean": [0, 0, 0, 0],
        "latents_std": [1, 1, 1, 1],
        "sigma_data": 0.5,
        "beauty_dist": [False, True],
        "split": "train",
        "residual_mean": 0.0,
        "residual_std": 1.1678,
        "cond_input_dropout": 0.2,
        "cond_input_max_noise": 0.1,
        "cond_input_mean": [13.36, 10.03, 16.17, 612.99, 869.73, 71.64, 0.674],
        "cond_input_std": [22.22, 22.13, 10.23, 453.81, 797.71, 34.23, 0.460],
    }

    dataset = H5LatentsDataset(**params)
    x = dataset[0]
    import matplotlib.pyplot as plt
    for i in range(x['cond_inputs_img'].shape[0]):
        plt.imshow(x['cond_inputs_img'][i].numpy(), cmap='terrain')
        plt.show()