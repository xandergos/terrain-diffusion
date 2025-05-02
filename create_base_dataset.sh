# # 360m resolution
# python3.11 ./terrain_diffusion/data/preprocessing/build_base_dataset.py \
#     --highres-elevation-folder /mnt/ntfs2/shared/data/terrain/dem_data/ \
#     --lowres-elevation-folder /mnt/ntfs2/shared/data/terrain/ETOPO_2022_v1_30s_N90W180_bed.tif \
#     --landcover-folder /mnt/ntfs2/shared/data/terrain/landcover_data/ \
#     --watercover-folder /mnt/ntfs2/shared/data/terrain/landcover_water_data/ \
#     --koppen-geiger-folder /mnt/ntfs2/shared/data/terrain/koppen_geiger.tif \
#     --climate-folder /mnt/ntfs2/shared/data/terrain/climate/ \
#     --highres-size 1024 \
#     --lowres-size 128 \
#     --lowres-sigma 5 \
#     --resolution 360 \
#     --num-chunks 1  \
#     --output-file dataset.h5 \
#     --num-workers 15 \
#     --prefetch 1 \
#     --edge-margin 2;
# 
# 
# # 180m resolution
# python3.11 ./terrain_diffusion/data/preprocessing/build_base_dataset.py \
#     --highres-elevation-folder /mnt/ntfs2/shared/data/terrain/dem_data/ \
#     --lowres-elevation-folder /mnt/ntfs2/shared/data/terrain/ETOPO_2022_v1_30s_N90W180_bed.tif \
#     --landcover-folder /mnt/ntfs2/shared/data/terrain/landcover_data/ \
#     --watercover-folder /mnt/ntfs2/shared/data/terrain/landcover_water_data/ \
#     --koppen-geiger-folder /mnt/ntfs2/shared/data/terrain/koppen_geiger.tif \
#     --climate-folder /mnt/ntfs2/shared/data/terrain/climate/ \
#     --highres-size 2048 \
#     --lowres-size 256 \
#     --lowres-sigma 5 \
#     --resolution 180 \
#     --num-chunks 1  \
#     --output-file dataset.h5 \
#     --num-workers 15 \
#     --prefetch 1 \
#     --edge-margin 3;

# 90m resolution
python3.11 ./terrain_diffusion/data/preprocessing/build_base_dataset.py  \
    --highres-elevation-folder /mnt/ntfs2/shared/data/terrain/dem_data/ \
    --lowres-elevation-folder /mnt/ntfs2/shared/data/terrain/ETOPO_2022_v1_30s_N90W180_bed.tif \
    --landcover-folder /mnt/ntfs2/shared/data/terrain/landcover_class/ \
    --watercover-folder /mnt/ntfs2/shared/data/terrain/landcover_water/ \
    --climate-folder /mnt/ntfs2/shared/data/terrain/climate/ \
    --highres-size 4096 \
    --lowres-size 512 \
    --lowres-sigma 5 \
    --resolution 90 \
    --num-chunks 2  \
    --output-file dataset.h5 \
    --num-workers 15 \
    --prefetch 1 \
    --edge-margin 5;
