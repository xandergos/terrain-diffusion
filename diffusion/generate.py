import click
from confection import Config, registry

from diffusion.registry import build_registry
from diffusion.samplers.tiled import TiledSampler
from diffusion.unet import EDMUnet2D

import os
from torchvision.utils import save_image


@click.command()
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), required=True)
@click.option("-m", "--model", "model_path", type=click.Path(exists=True), required=True)
@click.option("-o", "--output", "output_folder", type=click.Path(exists=False), required=True)
@click.option("-r", "--region", "region", type=str, required=True)
@click.option("-bs", "--batches", "batches", type=int, required=True)
@click.option("-p", "--num_parallel", "num_parallel", type=int, required=False, default=64)
@click.option("-s", "--seed", "seed", type=int, required=False, default=None)
@click.option("--overlap", "overlap", type=int, required=False, default=None)
def main(config_path, model_path, output_folder, batches, num_parallel, region, seed, overlap):
    build_registry()
    
    valid_keys = ['sampler', 'scheduler']
    config = Config().from_disk(config_path) if config_path else None
    config = {k: v for k, v in config.items() if k in valid_keys}
    
    model = EDMUnet2D.from_pretrained(model_path).to(config['sampler']['init']['device'])
    
    resolved = registry.resolve(config, validate=False)
    scheduler = resolved['scheduler']
    
    resolved['sampler']['init']['boundary'] = tuple(int(x) for x in region.split(','))
    resolved['sampler']['init']['batch_size'] = batches
    resolved['sampler']['init']['generation_batch_size'] = num_parallel // batches
    resolved['sampler']['init']['seed'] = seed
    if overlap is not None:
        resolved['sampler']['init']['overlap'] = overlap
    sampler = TiledSampler(model=model, 
                        scheduler=scheduler,
                        **resolved['sampler']['init'])
    print("Seed:", sampler.seed)
    images = sampler.get_region(*tuple(int(x) for x in region.split(',')))

    os.makedirs(output_folder, exist_ok=True)
    
    for i, image in enumerate(images):
        # Save the image
        save_image((image - image.min()) / (image.max() - image.min()), os.path.join(output_folder, f'output_{i:04d}.png'))
        def plot_image():
            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors

            samples_np = image[0].cpu().numpy()

            def normalize_data(x):
                return np.sign(x) * np.sqrt(np.abs(x))

            # Normalize the data
            data = samples_np
            normalized_data = normalize_data(data)
            
            # Solving min + t * (max - min) = 0 for t
            min_height = -9000
            max_height = 8000
            center = -normalize_data(min_height) / (normalize_data(max_height) - normalize_data(min_height))
            
            # Create a colorscale that sharply transitions from yellow to blue at 0
            colors_below = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffd9', '#addd8e', '#8c6d31', '#969696', '#ffffff']
            ticks = [t / 4 * center for t in range(5)] + [t / 4 * (1 - center) + center for t in range(5)]

            # Create custom colormap
            custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom", list(zip(ticks, colors_below)))

            # Create the figure and axis
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create the heatmap
            im = ax.imshow(normalized_data, cmap=custom_cmap,
                           vmin=normalize_data(min_height), vmax=normalize_data(max_height))

            # Remove axis ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
            cbar.set_label("Elevation (Nonlinear)", rotation=270, labelpad=15)

            # Set colorbar ticks
            cbar.set_ticks([normalized_data.min(), 0, normalized_data.max()])
            cbar.set_ticklabels([f" {int(data.min())}", "0", f" {int(data.max())}"])

            # Update the layout for a more appealing look
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')

            plt.tight_layout()
            plt.show()
            
        plot_image()

    print(f"Saved {len(images)} images to {output_folder}")
    

if __name__ == '__main__':
    main()