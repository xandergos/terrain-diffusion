import csv
import os
import click
import h5py
import torch

@click.command()
@click.argument('dataset-file')
@click.argument('val-pct', default=0.2)
@click.option('--seed', default=68197, help='Seed for random number generator.')
@click.option('--splits-csv', default="data/splits.csv", help='Path to CSV with chunk_id,split columns. If exists, uses it instead of generating new splits.')
def split_dataset(dataset_file, val_pct, seed, splits_csv):
    """
    Split the dataset into training and validation sets based on the filename attribute.
    Searches through all groups and datasets in the HDF5 file hierarchy.

    Args:
        dataset_file (str): Path to the HDF5 file containing the dataset.
        val_pct (float): Percentage of the dataset to be used for validation.
        seed (int): Random seed for reproducibility.
        splits_csv (str): Path to CSV with predefined splits. If exists, uses it.
    """
    # Check if we should use existing splits from CSV
    if splits_csv and os.path.exists(splits_csv):
        print(f"Loading splits from {splits_csv}")
        split_map = {}
        with open(splits_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                split_map[row['chunk_id']] = row['split']
        
        train_tags = [k for k, v in split_map.items() if v == 'train']
        val_tags = [k for k, v in split_map.items() if v == 'val']
        
        with h5py.File(dataset_file, 'a') as f:
            def assign_splits(name, obj):
                if isinstance(obj, h5py.Dataset) and 'chunk_id' in obj.attrs:
                    chunk_id = obj.attrs['chunk_id']
                    if chunk_id in split_map:
                        obj.attrs['split'] = split_map[chunk_id]
                    else:
                        print(f"Warning: chunk_id {chunk_id} not found in CSV, skipping")
            
            f.visititems(assign_splits)
        
        print(f"Applied splits from CSV. Training set: {len(train_tags)}, Validation set: {len(val_tags)}")
        return
    
    tags = set()
    
    def collect_tags(name, obj):
        if isinstance(obj, h5py.Dataset) and 'chunk_id' in obj.attrs:
            tags.add(obj.attrs['chunk_id'])
    
    with h5py.File(dataset_file, 'a') as f:
        # Recursively visit all datasets
        f.visititems(collect_tags)
        
        print(f"Found {len(tags)} unique tags")
        
        tags = list(tags)
        tags.sort()
        length_train = int((1 - val_pct) * len(tags))
        indices = torch.randperm(len(tags), generator=torch.Generator().manual_seed(seed)).tolist()
        tags = [tags[i] for i in indices]
        train_tags = tags[:length_train]
        val_tags = tags[length_train:]
        
        def assign_splits(name, obj):
            if isinstance(obj, h5py.Dataset) and 'chunk_id' in obj.attrs:
                if obj.attrs['chunk_id'] in train_tags:
                    obj.attrs['split'] = 'train'
                else:
                    obj.attrs['split'] = 'val'
        
        # Recursively assign splits
        f.visititems(assign_splits)
        
        print(f"Finished splitting dataset. Training set: {len(train_tags)} unique tags, Validation set: {len(val_tags)} unique tags.")
    
    # Save splits to CSV
    if splits_csv:
        with open(splits_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['chunk_id', 'split'])
            for chunk_id in train_tags:
                writer.writerow([chunk_id, 'train'])
            for chunk_id in val_tags:
                writer.writerow([chunk_id, 'val'])
        print(f"Saved splits to {splits_csv}")
    
    

if __name__ == '__main__':
    split_dataset()

