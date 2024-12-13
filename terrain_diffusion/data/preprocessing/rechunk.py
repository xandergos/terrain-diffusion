"""
Rechunks an encoded dataset created by build_encoded_dataset.py into smaller chunks.
Can be used to split large chunks into smaller ones for to allow for more variety in the validation split.
"""

import h5py
import click
from tqdm import tqdm
import numpy as np

@click.command()
@click.option('--input-file', required=True, help='Input HDF5 file containing the encoded dataset')
@click.option('--output-file', required=True, help='Output HDF5 file (can be same as input)')
@click.option('--split-factor', required=True, type=int, help='Number of splits along each dimension')
def rechunk_dataset(input_file: str, output_file: str, split_factor: int):
    """
    Splits chunks in an encoded dataset into smaller chunks.

    Args:
        input_file: Path to input HDF5 dataset created by build_encoded_dataset.py
        output_file: Path to output HDF5 file
        split_factor: Number of splits along each dimension (e.g., 2 splits a chunk into 4 pieces)
    """
    with h5py.File(input_file, 'r') as f_in, h5py.File(output_file, 'w') as f_out:
        for key in tqdm(f_in.keys()):
            chunk = f_in[key][()]

if __name__ == '__main__':
    rechunk_dataset()
