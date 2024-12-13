import h5py

with h5py.File('dataset_full_encoded.h5', 'r') as f:
    for key in list(f.keys()):
        print(key, f[key].shape)
