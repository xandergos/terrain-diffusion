# Setup and training from scratch *(not recommended)*

All steps can be completed with 24GB of GPU RAM

### Download data

#### 1. Download DEM data

Sometimes downloading certain files can fail. You may have to run the script multiple times until there are no failures. Previously downloaded files will not be re-downloaded.

```
./util_scripts/download_dem.sh
```

#### 2. Download ETOPO

Download the "30 Arc-Second Resolution GeoTIFF" [here](https://www.ncei.noaa.gov/products/etopo-global-relief-model) and place it in `data/global`.

#### 3. Download WorldClim data

Download `bio 10m` and `bio 30s` [here](https://www.worldclim.org/data/worldclim21.html). Extract all into `data/global`.

### Create Base Dataset

After all data has been downloaded, run:
```./util_scripts/create_base_dataset.sh```

### Train AutoEncoder

##### Prerequisites: Base dataset, 18GB GPU RAM

```
python terrain_diffusion train-ae --config ./configs/autoencoder/autoencoder_x8.cfg
```

Save the model with

```
python -m terrain_diffusion.training.save_model -c checkpoints/autoencoder_x8/latest_checkpoint -s 0.05
```

Move the resulting folder (Probably `checkpoints/autoencoder_x8/latest_checkpoint/saved_model`) to `checkpoints/models/autoencoder_x8`

Optional after training: Save disk space by deleting all checkpoints except `latest_checkpoint` in `checkpoints/autoencoder_x8`.

### Create Encoded Dataset

##### Prerequisites: Trained autoencoder + Base dataset. 8GB of GPU RAM

```
./util_scripts/create_encoded_dataset.sh
```

### Train Diffusion Decoder

#### Train main model:

```
accelerate launch -m terrain_diffusion train --config ./configs/diffusion_decoder_64-3.cfg
```

#### (Optional) Train guidance model:

This is only needed if you want to use AutoGuidance (significant quality improvement).

```
accelerate launch -m terrain_diffusion train --config ./configs/diffusion_decoder_32-3.cfg
```

#### (Optional) Perform autoguidance sweep:

If you trained a guidance model, use this to find out the (close to) optimal guidance parameters.

```
accelerate launch -m terrain_diffusion.training.sweeps/sweep_diffusion_decoder.py --config configs/diffusion_decoder/diffusion_decoder_64-3.cfg --n-trials 300 --storage
```

#### Distill into consistency model:

First, save the main model:

```
python -m terrain_diffusion.training.save_model -c checkpoints/diffusion_decoder-64x3/latest_checkpoint -s <OPTIMAL_EMA_SIGMA>
```

If you did not do an autoguidance sweep, you can just use -s 0.05 as a decent default.

```
python -m terrain_diffusion.training.save_model -c checkpoints/diffusion_decoder-32x3/latest_checkpoint -s <OPTIMAL_EMA_SIGMA> -e <OPTIMAL_EMA_STEP>
```

Regardless of whether you ran an AutoGuidance sweep or not, update the values in `configs/diffusion_decoder/consistency_decoder_64x3.cfg`:

```yaml
[model]
main_path="checkpoints/diffusion_decoder-64x3/latest_checkpoint/saved_model"
guide_path="checkpoints/diffusion_decoder-32x3/latest_checkpoint/saved_model"
guidance_scale=<OPTIMAL_GUIDANCE_SCALE>
```

Leave `main_path` as-is. Leave `guide_path` as-is **if** you are using AutoGuidance, otherwise set to `null`.

Then you can distill into a consistency model:

```
accelerate launch -m terrain_diffusion distill --config ./configs/diffusion_decoder/consistency_decoder_64-3.cfg
```

#### (Optional) Consistency model sweep

Run a sweep over consistency model params:

```
accelerate launch -m terrain_diffusion.training.sweeps/sweep_consistency_decoder.py --config configs/diffusion_decoder/consistency_decoder_64-3.cfg --n-trials 300 --storage
```

#### Save consistency model

```
python -m terrain_diffusion.training.save_model -c checkpoints/consistency_decoder-64x3/latest_checkpoint -s <OPTIMAL_EMA_SIGMA> -e <OPTIMAL_EMA_STEP>
```

Move the output folder (Probably `checkpoints/consistency_decoder-64x3/latest_checkpoint/saved_model`) to `checkpoints/models/consistency_decoder-64x3`


### Train Diffusion Base

#### Train main model:

```
accelerate launch -m terrain_diffusion train --config ./configs/diffusion_base/diffusion_192-3.cfg
```

#### (Optional) Train guidance model:

This is only needed if you want to use AutoGuidance (significant quality improvement).

```
accelerate launch -m terrain_diffusion train --config ./configs/diffusion_base/diffusion_128-3.cfg
```

#### (Optional) Perform autoguidance sweep:

If you trained a guidance model, use this to find out the (close to) optimal guidance parameters.

```
accelerate launch -m terrain_diffusion.training.sweeps/sweep_diffusion_base.py --config configs/diffusion_base/diffusion_192-3.cfg --n-trials 300 --storage
```

#### Distill into consistency model:

First, save the main model:

```
python -m terrain_diffusion.training.save_model -c checkpoints/diffusion_base-192x3/latest_checkpoint -s <OPTIMAL_EMA_SIGMA>
```

If you did not do an autoguidance sweep, you can just use -s 0.05 as a decent default.

```
python -m terrain_diffusion.training.save_model -c checkpoints/diffusion_base-128x3/latest_checkpoint -s <OPTIMAL_EMA_SIGMA> -e <OPTIMAL_EMA_STEP>
```

Regardless of whether you ran an AutoGuidance sweep or not, update the values in `configs/diffusion_base/consistency_base_192-3.cfg`:

```yaml
[model]
main_path="checkpoints/diffusion_base-192x3/latest_checkpoint/saved_model"
guide_path="checkpoints/diffusion_base-128x3/latest_checkpoint/saved_model"
guidance_scale=<OPTIMAL_GUIDANCE_SCALE>
```

Leave `main_path` as-is. Leave `guide_path` as-is **if** you are using AutoGuidance, otherwise set to `null`.

Then you can distill into a consistency model:

```
accelerate launch -m terrain_diffusion distill --config ./configs/diffusion_base/consistency_base_192-3.cfg
```

#### (Optional) Consistency model sweep

Run a sweep over consistency model params:

```
accelerate launch -m terrain_diffusion.training.sweeps/sweep_consistency_base.py --config configs/diffusion_base/consistency_base_192-3.cfg --n-trials 300 --storage
```

#### Save consistency model

```
python -m terrain_diffusion.training.save_model -c checkpoints/consistency_base-192x3/latest_checkpoint -s <OPTIMAL_EMA_SIGMA> -e <OPTIMAL_EMA_STEP>
```

Move the output folder (Probably `checkpoints/consistency_base-192x3/latest_checkpoint/saved_model`) to `checkpoints/models/consistency_base-192x3`

### Train Coarse Model

Train with:

```
accelerate launch -m terrain_diffusion train --config ./configs/diffusion_coarse/diffusion_coarse.cfg
```

Save the model:

```
python -m terrain_diffusion.training.save_model -c checkpoints/diffusion_coarse/latest_checkpoint -s 0.05
```

Move the output folder (Probably `checkpoints/diffusion_coarse/latest_checkpoint/saved_model`) to `checkpoints/models/diffusion_coarse`
