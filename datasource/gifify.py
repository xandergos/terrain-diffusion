import os
import glob
from PIL import Image

def create_gif_from_pngs(input_folder, output_file, duration=500):
    """
    Create a GIF from all PNG files in the specified folder.
    
    :param input_folder: Path to the folder containing PNG files
    :param output_file: Path and filename for the output GIF
    :param duration: Duration for each frame in milliseconds (default: 500ms)
    """
    # Get all PNG files in the specified folder
    png_files = glob.glob(os.path.join(input_folder, '*.png'))
    png_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    
    if not png_files:
        print(f"No PNG files found in {input_folder}")
        return
    
    # Open all images
    images = [Image.open(file) for file in png_files]
    
    # Save as GIF
    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    
    print(f"GIF created successfully: {output_file}")

# Usage
input_folder = 'checkpoints/64-i'
output_file = 'checkpoints/64-i/output.gif'

create_gif_from_pngs(input_folder, output_file)
