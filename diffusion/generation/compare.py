import os
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.image as mpimg

def compare_images(folder1, folder2):
    # Get list of files in first folder
    files = sorted(os.listdir(folder1))
    
    # Create figure with subplots
    num_images = len(files)
    fig = plt.figure(figsize=(12, 4))
    
    for i, filename in enumerate(files):
        # Load images from both folders
        img1 = mpimg.imread(os.path.join(folder1, filename))
        img2 = mpimg.imread(os.path.join(folder2, filename))
        
        # Create side-by-side subplots
        plt.subplot(1, 2, 1)
        plt.imshow(img1, cmap='gray')
        plt.title(f'Folder 1: {filename}')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img2, cmap='gray')
        plt.title(f'Folder 2: {filename}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        plt.close()

if __name__ == "__main__":
    # Example usage
    folder1 = "outputs/fid_images_64x3_ema005_step192k/original"
    folder2 = "outputs/fid_images_64x3_ema005_step192k/reconstructed"
    compare_images(folder1, folder2)
