import os
import re

checkpoint_dir = "checkpoints/autoencoder_x8/phema"

if os.path.exists(checkpoint_dir):
    for filename in os.listdir(checkpoint_dir):
        if re.match(r"1\.(\d+)\.pt$", filename):
            step_num = int(re.match(r"1\.(\d+)\.pt$", filename).group(1))
            if step_num % 5120 != 0:
                file_path = os.path.join(checkpoint_dir, filename)
                os.remove(file_path)
                print(f"Deleted: {filename}")
