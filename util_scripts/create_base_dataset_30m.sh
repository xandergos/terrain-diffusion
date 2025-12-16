# 30m resolution (Copernicus)
python3.11 -m terrain_diffusion.data.preprocessing.build_base_dataset  \
    --highres-elevation-folder data/copernicus_data/ \
    --lowres-elevation-file data/global/ETOPO_2022_v1_30s_N90W180_bed.tif \
    --climate-folder data/global/ \
    --highres-size 2048 \
    --lowres-size 256 \
    --lowres-sigma 5 \
    --resolution 30.71 \
    --res-group 30 \
    --num-chunks 1  \
    --output-file data/dataset_30m.h5 \
    --num-workers 15 \
    --prefetch 1 \
    --overwrite \
    --edge-margin 5 \
    --data-source copernicus \
    --ocean-tile-divisor 10;

python3.11 -m terrain_diffusion.data.preprocessing.define_splits data/dataset_30m.h5 0.2 --splits-csv data/splits_30m.csv

python3.11 -m terrain_diffusion.data.preprocessing.beauty_score data/dataset_30m.h5
