import click
import h5py
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

def calculate_stats_welford(res_group, dataset_name):
    """
    Calculate mean and standard deviation using Welford's algorithm.
    Handles both single-channel and multi-channel datasets.
    
    Args:
        res_group: HDF5 group containing the hierarchical data structure
        dataset_name: Name of the dataset to process (e.g., 'residual', 'climate')
    
    Returns:
        tuple: (means, stds) - numpy arrays of means and standard deviations
    """
    print(f"Calculating {dataset_name} mean and std using Welford's algorithm...")
    
    # Get first non-None dataset to determine shape
    first_data = None
    for chunk_id in res_group.keys():
        for subchunk_id in res_group[chunk_id].keys():
            if dataset_name in res_group[chunk_id][subchunk_id]:
                first_data = res_group[chunk_id][subchunk_id][dataset_name][:]
                break
    assert first_data is not None, f"No {dataset_name} data found in the dataset"
    
    if len(first_data.shape) == 3:
        num_channels = first_data.shape[0]
        means = np.zeros(num_channels, dtype=np.float64)
        M2s = np.zeros(num_channels, dtype=np.float64)
        counts = np.zeros(num_channels, dtype=np.int64)
    else:
        means = 0.0
        M2s = 0.0
        counts = 0
    
    for chunk_id in (pbar := tqdm(res_group.keys(), desc=f"Processing {dataset_name}")):
        chunk_group = res_group[chunk_id]
        for subchunk_id in chunk_group.keys():
            subchunk_group = chunk_group[subchunk_id]
            if dataset_name in subchunk_group:
                data = subchunk_group[dataset_name][:]
                if len(data.shape) == 3:
                    channel_data = data.reshape(num_channels, -1)
                    channel_data = channel_data[:, ~np.isnan(channel_data).any(axis=0)]
                    if channel_data.shape[1] == 0:
                        continue
                    counts += channel_data.shape[1]
                    delta = channel_data - means[:, np.newaxis]
                    means += np.sum(delta, axis=1) / counts
                    delta2 = channel_data - means[:, np.newaxis]
                    M2s += np.sum(delta * delta2, axis=1)
                else:
                    data = data.flatten()
                    data = data[~np.isnan(data)]
                    if len(data) == 0:
                        continue
                    counts += len(data)
                    delta = data - means
                    means += np.sum(delta) / counts
                    delta2 = data - means
                    M2s += np.sum(delta * delta2)
    
    # Calculate final standard deviations
    stds = np.sqrt(M2s / (counts - 1))
    
    return means, stds

@click.command()
@click.argument('dataset', type=str)
@click.argument('resolution', type=int)
def calculate_stats(dataset, resolution):
    with h5py.File(dataset, 'a') as f:
        if str(resolution) not in f:
            click.echo(f"Resolution {resolution} not found in dataset", err=True)
            return
        res_group = f[f'{resolution}']
        datasets_to_process = ['climate']
        for dataset_name in datasets_to_process:
            means, stds = calculate_stats_welford(res_group, dataset_name)
        
            print(f"{dataset_name} mean: {means}")
            print(f"{dataset_name} std: {stds}")
            
            # Update attributes in hierarchical structure
            res_group.attrs[f'{dataset_name}_std'] = stds
            res_group.attrs[f'{dataset_name}_mean'] = means
            
if __name__ == '__main__':
    calculate_stats()