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

    Args:
        dataset_file (str): Path to the HDF5 file containing the dataset.
        val_pct (float): Percentage of the dataset to be used for validation.
    """
    tags = set()
    with h5py.File(dataset_file, 'a') as f:
        for key in f.keys():
            tags.add(f[key].attrs['chunk_id'])

        tags = list(tags)
        tags.sort()
        length_train = int((1 - val_pct) * len(tags))
        indices = torch.randperm(len(tags), generator=torch.Generator().manual_seed(seed)).tolist()
        tags = [tags[i] for i in indices]
        train_tags = tags[:length_train]
        val_tags = tags[length_train:]
        
        for key in f.keys():
            if f[key].attrs['chunk_id'] in train_tags:
                f[key].attrs['split'] = 'train'
            else:
                f[key].attrs['split'] = 'val'

        print(f"Finished splitting dataset. Training set: {len(train_tags)} unique tags, Validation set: {len(val_tags)} unique tags.")
    
    

if __name__ == '__main__':
    split_dataset()

