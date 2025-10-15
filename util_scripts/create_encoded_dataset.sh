# Call this script after running create_base_dataset.sh

# Save autoencoder model if not already saved
if [ ! -d "./checkpoints/models/autoencoder_x8" ]; then
    python3.11 ./terrain_diffusion/training/save_model.py -c ./checkpoints/autoencoder_x8/latest_checkpoint -s 0.05
    mkdir -p ./checkpoints/models/autoencoder_x8
    mv ./checkpoints/autoencoder_x8/latest_checkpoint/saved_model/* ./checkpoints/models/autoencoder_x8/
fi

# 90m resolution
python3.11 ./terrain_diffusion/data/preprocessing/build_encoded_dataset.py  \
    --dataset data/dataset.h5 \
    --resolution 90 \
    --encoder ./checkpoints/models/autoencoder_x8 \
    --use-fp16 \
    --compile-model \
    --residual-mean 0.0 \
    --residual-std 1.1678 \
    --overwrite;