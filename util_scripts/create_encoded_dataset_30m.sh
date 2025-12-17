# Call this script after running create_base_dataset.sh

# Save autoencoder model if not already saved
if [ ! -d "./checkpoints/models/autoencoder_x8" ]; then
    python3.11 -m terrain_diffusion.training.save_model -c ./checkpoints/autoencoder_x8/latest_checkpoint -s 0.05
    mkdir -p ./checkpoints/models/autoencoder_x8
    mv ./checkpoints/autoencoder_x8/latest_checkpoint/saved_model/* ./checkpoints/models/autoencoder_x8/
fi

# 90m resolution
python3.11 -m terrain_diffusion.data.preprocessing.build_encoded_dataset  \
    --dataset data/dataset_30m.h5 \
    --resolution 30 \
    --encoder ./checkpoints/models/autoencoder_x8 \
    --use-fp16 \
    --compile-model \
    --residual-mean 0.0 \
    --residual-std 0.7 \
    --overwrite;