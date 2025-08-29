# 90m resolution
#python3.11 ./terrain_diffusion/data/preprocessing/build_base_dataset.py  \
#    --highres-elevation-folder data/dem_data/ \
#    --lowres-elevation-file data/global/ETOPO_2022_v1_30s_N90W180_bed.tif \
#    --landcover-folder data/landcover_class/ \
#    --watercover-folder data/landcover_water/ \
#    --climate-folder data/global/ \
#    --highres-size 4096 \
#    --lowres-size 512 \
#    --lowres-sigma 5 \
#    --resolution 90 \
#    --num-chunks 2  \
#    --output-file data/dataset.h5 \
#    --num-workers 15 \
#    --prefetch 1 \
#    --edge-margin 5;

python3.11 ./terrain_diffusion/data/preprocessing/define_splits.py data/dataset.h5 0.2

python3.11 ./terrain_diffusion/data/preprocessing/beauty_score.py data/dataset.h5