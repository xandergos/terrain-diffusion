# 90m resolution
python3.11 ./terrain_diffusion/data/preprocessing/build_encoded_dataset.py  \
    --dataset dataset.h5 \
    --resolution 90 \
    --encoder ./checkpoints/models/autoencoder \
    --use-fp16 \
    --compile-model \
    --overwrite;

# # 180m resolution
# python3.11 ./terrain_diffusion/data/preprocessing/build_encoded_dataset.py \
#     --dataset dataset.h5 \
#     --resolution 180 \
#     --encoder ./checkpoints/models/autoencoder \
#     --use-fp16 \
#     --compile-model;
# 
# # 360m resolution
# python3.11 ./terrain_diffusion/data/preprocessing/build_encoded_dataset.py \
#     --dataset dataset.h5 \
#     --resolution 360 \
#     --encoder ./checkpoints/models/autoencoder \
#     --use-fp16 \
#     --compile-model;