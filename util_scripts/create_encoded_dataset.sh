# Call this script after running create_base_dataset.sh

# Save autoencoder model
python3.11 ./terrain_diffusion/training/save_model.py -c ./checkpoints/autoencoder_x8/latest_checkpoint -s 0.05

# 90m resolution
python3.11 ./terrain_diffusion/data/preprocessing/build_encoded_dataset.py  \
    --dataset data/dataset.h5 \
    --resolution 90 \
    --encoder ./checkpoints/autoencoder_x8/latest_checkpoint/model \
    --use-fp16 \
    --compile-model \
    --overwrite;