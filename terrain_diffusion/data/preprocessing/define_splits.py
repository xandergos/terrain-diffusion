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
    filenames = set()
    with h5py.File(dataset_file, 'a') as f:
        for key in f.keys():
            filenames.add(f[key].attrs['filename'])

        filenames = list(filenames)
        filenames.sort()
        length_train = int((1 - val_pct) * len(filenames))
        indices = torch.randperm(len(filenames), generator=torch.Generator().manual_seed(seed)).tolist()
        filenames = [filenames[i] for i in indices]
        train_filenames = filenames[:length_train]
        val_filenames = filenames[length_train:]
        
        f.attrs['split:train'] = train_filenames
        f.attrs['split:val'] = val_filenames

        print(f"Finished splitting dataset. Training set: {len(train_filenames)} unique files, Validation set: {len(val_filenames)} unique files.")
    
    

if __name__ == '__main__':
    split_dataset()

