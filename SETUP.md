# Setup From Scratch

All steps can be completed with 24GB of GPU RAM

### Download data

#### 1. Download DEM data

```./util_scripts/download_dem.sh```

#### 2. Download landcover data

```./util_scripts/download_landcover.sh```

#### 3. Download water coverage data

```./util_scripts/download_landcover_water.sh```

#### 4. Download ETOPO

Download the file [here](https://www.ngdc.noaa.gov/thredds/catalog/global/ETOPO2022/30s/30s_bed_elev_netcdf/catalog.html?dataset=globalDatasetScan/ETOPO2022/30s/30s_bed_elev_netcdf/ETOPO_2022_v1_30s_N90W180_bed.nc) and place it in `data/global`

#### 5. Download WorldClim data

Download `bio 10m`, `bio 30s`, and `elev 10m` [here](https://www.worldclim.org/data/worldclim21.html). Extract all into `data/global`.

#### 6. Download Koppen-geiger data

1. Go to https://www.gloh2o.org/koppen/
2. Download the data
3. Extract `1991_2020/0p00833333.tif` to `data/global/koppen_geiger_0p00833333.tif`
4. Extra files can be discarded

### Create Base Dataset

##### Prerequisites: Requires data to be downloaded and placed in data/ folder

```./util_scripts/create_base_dataset.sh```

### Train GAN

##### Prerequisites: WorldClim data downloaded, uses ~4GB GPU RAM

```python terrain_diffusion train-gan --config ./configs/gan/gan.cfg```

Optional: Save disk space by deleting all checkpoints except `latest_checkpoint` in `checkpoints/gan`.

### Train AutoEncoder

##### Prerequisites: Base dataset, 18GB GPU RAM

```python terrain_diffusion train-ae --config ./configs/autoencoder/autoencoder_x8.cfg```

Optional: Save disk space by deleting all checkpoints except `latest_checkpoint` in `checkpoints/autoencoder_x8`.

### Create Encoded Dataset

##### Prerequisites: Trained autoencoder + Base dataset. 8GB of GPU RAM

```./util_scripts/create_encoded_dataset.sh```

### Train Diffusion Decoder

Train main model:

```python terrain_diffusion train --config ./configs/diffusion_decoder_64-3.cfg```

Train guidance model:

```python terrain_diffusion train --config ./configs/diffusion_decoder_32-3.cfg```

Perform autoguidance sweep:

```python ./terrain_diffusion/inference/evaluation/sweep_fid.py --config ./configs/sweeps/decoder_fid_64_32.cfg```

Train consistency model:

```python terrain_diffusion distill --config ./configs/diffusion_decoder/consistency/