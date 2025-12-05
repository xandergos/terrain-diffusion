import click
from terrain_diffusion.training.train import main as train_main
from terrain_diffusion.training.save_model import save_model as save_model_main
from terrain_diffusion.data.preprocessing.build_base_dataset import process_base_dataset as build_base_dataset_main
from terrain_diffusion.data.preprocessing.build_encoded_dataset import process_encoded_dataset as build_encoded_dataset_main
from terrain_diffusion.data.preprocessing.define_splits import split_dataset as define_splits_main
from terrain_diffusion.inference.world_explorer import main as explore_main
from terrain_diffusion.inference.world_generator import main as generate_main
from terrain_diffusion.inference.api import main as api_main
from terrain_diffusion.inference.minecraft_api import main as mc_api_main

@click.group()
def cli():
    """Terrain Diffusion CLI - Main entry point for all commands"""
    pass

# Training commands
cli.add_command(train_main, name='train')
cli.add_command(save_model_main, name='save-model')

# Data preprocessing commands
cli.add_command(build_base_dataset_main, name='build-base-dataset')
cli.add_command(build_encoded_dataset_main, name='build-encoded-dataset')
cli.add_command(define_splits_main, name='define-splits')

# Inference commands
cli.add_command(explore_main, name='explore')
cli.add_command(generate_main, name='generate')
cli.add_command(api_main, name='api')
cli.add_command(mc_api_main, name='mc-api')

if __name__ == '__main__':
    cli()
