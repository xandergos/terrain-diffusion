"""
Takes a folder of elevation .tiff files and preprocesses them for training a diffusion model.
The output can be used to train superresolution or base diffusion models.

Requires a pre-trained encoder model.
"""

import numpy as np
import torch
import h5py
from tqdm import tqdm
import click
from terrain_diffusion.models.edm_autoencoder import EDMAutoencoder
import torch.nn.functional as F

@click.command()
@click.option('--dataset', required=True, help='Path to base HDF5 dataset containing highfreq/lowfreq')
@click.option('--resolution', type=int, required=True, help='Resolution of the input images in meters')
@click.option('--encoder', 'encoder_model_path', required=True, help='Path to encoder model checkpoint')
@click.option('--use-fp16', is_flag=True, help='Use FP16 for encoding', default=False)
@click.option('--compile-model', is_flag=True, help='Compile the model', default=False)
@click.option('--overwrite', is_flag=True, help='Overwrite existing datasets', default=False)
@click.option('--min-pct-land', type=float, help='Minimum percentage of land in the chunk (0-1)', default=0.0)
@click.option('--residual-mean', type=float, default=0.0, show_default=True, help='Mean used to normalize residual channel')
@click.option('--residual-std', type=float, default=1.1678, show_default=True, help='Std used to normalize residual channel')
def process_encoded_dataset(dataset, resolution, encoder_model_path, use_fp16, compile_model, overwrite, min_pct_land, residual_mean, residual_std, water_mean, water_std):
    """
    Add latent encodings to an existing HDF5 dataset containing high/low frequency components.
    
    Takes a base HDF5 dataset containing highfreq/lowfreq components and adds corresponding latent
    encodings using the specified encoder model. Data is organized in groups:
    {resolution}/{chunk_id}/{subchunk_id}/{data_type}
    """
    device = 'cuda'
    model = EDMAutoencoder.from_pretrained(encoder_model_path)
    model.to(device)
    
    printed_latent_shape = False

    if compile_model:
        model = torch.compile(model)

    print(f"Normalizing residual with mean {residual_mean} and std {residual_std}")

    with h5py.File(dataset, 'a') as f:
        if str(resolution) not in f:
            raise ValueError(f"Resolution {resolution} not found in dataset")
        
        res_group = f[str(resolution)]
        
        # Process each chunk and subchunk
        for chunk_id in tqdm(res_group.keys(), desc="Processing chunks"):
            chunk_group = res_group[chunk_id]
            for subchunk_id in chunk_group.keys():
                subchunk_group = chunk_group[subchunk_id]
                
                if 'latent' in subchunk_group and not overwrite:
                    continue
                
                if 'residual' not in subchunk_group:
                    print(f"No residual data found at resolution {resolution} for chunk {chunk_id}, subchunk {subchunk_id}")
                    continue
                
                if subchunk_group['residual'].attrs['pct_land'] < min_pct_land:
                    continue

                # Process residual data (full resolution)
                residual = torch.from_numpy(subchunk_group['residual'][:])[None]
                residual = (residual - residual_mean) / residual_std

                # Concatenate all channels
                input_data = residual.float()
                
                transformed_latent = []
                for horiz_flip in [False, True]:
                    for rot_deg in [0, 90, 180, 270]:
                        input_transformed = input_data
                        if horiz_flip:
                            input_transformed = torch.flip(input_transformed, dims=[-1])
                        if rot_deg != 0:
                            input_transformed = torch.rot90(input_transformed, k=rot_deg // 90, dims=[-2, -1])
                        
                        with torch.no_grad():
                            model_input = input_transformed[None].to(device=device)
                            with torch.autocast(device_type=device, dtype=torch.float16 if use_fp16 else torch.float32):
                                latent_residual_means, latent_residual_logvars = model.preencode(model_input, conditional_inputs=[])
                                latent_residual = torch.cat([latent_residual_means, latent_residual_logvars], dim=1)[0].cpu()
                        if not printed_latent_shape:
                            print(f"Autoencoder input shape: {model_input.shape}")
                            print(f"Latent shape: {latent_residual.shape}")
                            printed_latent_shape = True
                        transformed_latent.append(latent_residual.numpy())
                
                transformed_latent = np.stack(transformed_latent).astype(np.float16)
                
                if 'latent' in subchunk_group:
                    del subchunk_group['latent']
                dset = subchunk_group.create_dataset('latent', data=transformed_latent, compression='lzf', chunks=(1, transformed_latent.shape[1], 32, 32))
                dset.attrs.update(subchunk_group['residual'].attrs)
                dset.attrs['data_type'] = 'latent'
        
        # Calculate per-channel latent statistics
        print("Calculating per-channel latent mean and std using Welford's algorithm...")
        
        # Initialize statistics arrays using first latent to get number of channels
        num_channels = None
        for chunk_id in res_group:
            for subchunk_id in res_group[chunk_id]:
                if 'latent' in res_group[chunk_id][subchunk_id]:
                    num_channels = res_group[chunk_id][subchunk_id]['latent'].shape[1] // 2
                    break
                
        if num_channels is None:
            print("No latent datasets found. Cannot calculate statistics.")
            return
            
        means = np.zeros(num_channels)
        M2s = np.zeros(num_channels)
        count = 0
        
        for chunk_id in tqdm(res_group.keys(), desc="Computing statistics"):
            chunk_group = res_group[chunk_id]
            for subchunk_id in chunk_group.keys():
                subchunk_group = chunk_group[subchunk_id]
                if 'latent' not in subchunk_group:
                    continue
                    
                data = subchunk_group['latent'][:]
                latent_mean, latent_logstd = data[:, :num_channels], data[:, num_channels:]
                sampled_latent = np.random.randn(*latent_mean.shape) * np.exp(latent_logstd * 0.5) + latent_mean
                
                batch_size = sampled_latent.shape[0] * sampled_latent.shape[2] * sampled_latent.shape[3]
                data = sampled_latent.transpose(0, 2, 3, 1).reshape(-1, num_channels)
                
                count += batch_size
                delta = data - means
                means += np.sum(delta, axis=0) / count
                delta2 = data - means
                M2s += np.sum(delta * delta2, axis=0)
        
        latent_means = means
        latent_stds = np.sqrt(M2s / (count - 1))
        
        print(f"Latent means: {latent_means}")
        print(f"Latent stds: {latent_stds}")
        
        # Store in global attributes
        res_group.attrs[f'latent_stds'] = latent_stds
        res_group.attrs[f'latent_means'] = latent_means

        print(f"Finished processing resolution {resolution}")

if __name__ == '__main__':
    process_encoded_dataset()