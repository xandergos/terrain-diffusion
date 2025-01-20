import click
import h5py
import torch

@click.command()
@click.argument('dataset-file')
@click.argument('val-pct', default=0.2)
@click.option('--seed', default=68197, help='Seed for random number generator.')
def split_dataset(dataset_file, val_pct=0.2, seed=68197):
    """
    Split the dataset into training and validation sets based on the filename attribute.
    Searches through all groups and datasets in the HDF5 file hierarchy.

    Args:
        dataset_file (str): Path to the HDF5 file containing the dataset.
        val_pct (float): Percentage of the dataset to be used for validation.
        seed (int): Random seed for reproducibility.
    """
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
    
    

if __name__ == '__main__':
    split_dataset()

