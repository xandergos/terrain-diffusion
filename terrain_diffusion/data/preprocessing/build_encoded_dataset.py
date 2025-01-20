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
from terrain_diffusion.training.unet import EDMAutoencoder

@click.command()
@click.option('--dataset', required=True, help='Path to base HDF5 dataset containing highfreq/lowfreq')
@click.option('--resolution', type=int, required=True, help='Resolution of the input images in meters')
@click.option('--encoder', 'encoder_model_path', required=True, help='Path to encoder model checkpoint')
@click.option('--use-fp16', is_flag=True, help='Use FP16 for encoding', default=False)
@click.option('--compile-model', is_flag=True, help='Compile the model', default=False)
@click.option('--overwrite', is_flag=True, help='Overwrite existing datasets', default=False)
def process_encoded_dataset(dataset, resolution, encoder_model_path, use_fp16, compile_model, overwrite):
    """
    Add latent encodings to an existing HDF5 dataset containing high/low frequency components.
    
    Takes a base HDF5 dataset containing highfreq/lowfreq components and adds corresponding latent
    encodings using the specified encoder model. Data is organized in groups:
    {resolution}/{chunk_id}/{subchunk_id}/{data_type}
    """
    device = 'cuda'
    model = EDMAutoencoder.from_pretrained(encoder_model_path)
    model = model.encoder
    use_watercover = model.config.in_channels == 2
    model.to(device)

    if compile_model:
        model = torch.compile(model)

    with h5py.File(dataset, 'a') as f:
        if str(resolution) not in f:
            raise ValueError(f"Resolution {resolution} not found in dataset")
        
        res_group = f[str(resolution)]
        
        # Process each chunk and subchunk
        for chunk_id in tqdm(res_group.keys(), desc="Processing chunks"):
            chunk_group = res_group[chunk_id]
            for subchunk_id in chunk_group.keys():
                subchunk_group = chunk_group[subchunk_id]
                
                # Skip if latent already exists and we're not overwriting
                if 'latent' in subchunk_group and not overwrite:
                    continue
                
                if 'residual' not in subchunk_group:
                    print(f"No residual data found at resolution {resolution} for chunk {chunk_id}, subchunk {subchunk_id}")
                    continue
                
                # Process residual data
                residual = torch.from_numpy(subchunk_group['residual'][:])[None]
                residual = (residual - res_group.attrs['residual_mean']) / \
                          res_group.attrs['residual_std']
                
                input_data = residual
                if use_watercover:
                    # Add watercover channel, defaulting to zeros if not present
                    watercover = torch.zeros_like(residual)
                    if 'watercover' in subchunk_group:
                        watercover = torch.from_numpy(subchunk_group['watercover'][:])[None] / 100
                    input_data = torch.cat([residual, watercover], dim=0)
                
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
                                latent_residual = model(model_input, noise_labels=None, conditional_inputs=[]).cpu()[0]
                        transformed_latent.append(latent_residual.numpy())
                
                transformed_latent = np.stack(transformed_latent)
                
                if 'latent' in subchunk_group:
                    del subchunk_group['latent']
                dset = subchunk_group.create_dataset('latent', data=transformed_latent, compression='lzf', chunks=(1, 8, 32, 32))
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