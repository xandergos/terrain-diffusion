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
def main(config_path, model_path, output_folder):
    build_registry()
    
    valid_keys = ['sampler', 'scheduler']
    config = Config().from_disk(config_path) if config_path else None
    config = {k: v for k, v in config.items() if k in valid_keys}
    
    model = EDMUnet2D.from_pretrained(model_path).to(config['sampler']['init']['device'])
    
    resolved = registry.resolve(config, validate=False)
    scheduler = resolved['scheduler']
    
    sampler = TiledSampler(model=model, 
                        scheduler=scheduler,
                        **resolved['sampler']['init'])
    images = sampler.get_region(*resolved['sampler']['region'])

    os.makedirs(output_folder, exist_ok=True)
    
    for i, image in enumerate(images):
        # Ensure the image is in the correct range [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        
        # Save the image
        save_image(image, os.path.join(output_folder, f'output_{i:04d}.png'))

    print(f"Saved {len(images)} images to {output_folder}")
    

if __name__ == '__main__':
    main()