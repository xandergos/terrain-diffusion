## Download `data`

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
(Requires data to be downloaded and placed in data/ folder)

```./util_scripts/create_base_dataset.sh```


## Train models from scratch


### 1. Train GAN

##### Prerequisites: WorldClim data downloaded

```python terrain_diffusion train-gan --config ./configs/gan/gan.cfg```

### 2. Train AutoEncoder

##### Prerequisites: Base dataset created

```python terrain_diffusion train-ae --config ./configs/autoencoder/autoencoder_x8.yaml```

