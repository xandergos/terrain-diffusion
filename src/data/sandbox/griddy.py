import os
import math
from PIL import Image

def create_image_grid(input_folder, output_file, grid_width=4):
    # Get all image files from the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = image_files[:20]
    
    # Ensure the number of images is a multiple of grid_width
    num_images = len(image_files) - (len(image_files) % grid_width)
    image_files = image_files[:num_images]
    
    # Open the first image to get dimensions
    with Image.open(os.path.join(input_folder, image_files[0])) as img:
        img_width, img_height = img.size
    
    # Calculate grid dimensions
    grid_height = num_images // grid_width
    grid_img = Image.new('RGB', (grid_width * img_width, grid_height * img_height))
    
    # Place images in the grid
    for i, img_file in enumerate(image_files):
        with Image.open(os.path.join(input_folder, img_file)) as img:
            x = (i % grid_width) * img_width
            y = (i // grid_width) * img_height
            grid_img.paste(img, (x, y))
    
    # Save the grid image
    grid_img.save(output_file)
    print(f"Grid image saved to {output_file}")

# Usage
input_folder = 'outputs/64_current'
output_file = 'outputs/64_current/current_grid.png'
create_image_grid(input_folder, output_file)
