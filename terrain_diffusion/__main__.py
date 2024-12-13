import click
from terrain_diffusion.training.diffusion.train import main as train_main
from terrain_diffusion.training.autoencoder.train_ae import main as train_ae_main
from terrain_diffusion.training.consistency.distill import distill as distill_main
from terrain_diffusion.training.save_model import save_model as save_model_main
from terrain_diffusion.inference.evaluation.evaluate_sr_fid import evaluate_sr_fid_cli as evaluate_sr_fid_main
from terrain_diffusion.inference.evaluation.evaluate_sr_fid_consistency import evaluate_sr_fid as evaluate_sr_fid_consistency_main
from terrain_diffusion.data.preprocessing.build_base_dataset import process_dataset as build_base_dataset_main
from terrain_diffusion.data.preprocessing.build_encoded_dataset import process_dataset as build_encoded_dataset_main
from terrain_diffusion.data.preprocessing.define_splits import split_dataset as define_splits_main

@click.group()
def cli():
    """Terrain Diffusion CLI - Main entry point for all commands"""
    pass

# Training commands
cli.add_command(train_main, name='train')
cli.add_command(train_ae_main, name='train-ae')
cli.add_command(distill_main, name='distill')
cli.add_command(save_model_main, name='save-model')

# Evaluation commands
cli.add_command(evaluate_sr_fid_main, name='evaluate-sr-fid')
cli.add_command(evaluate_sr_fid_consistency_main, name='evaluate-sr-fid-consistency')

# Data preprocessing commands
cli.add_command(build_base_dataset_main, name='build-base-dataset')
cli.add_command(build_encoded_dataset_main, name='build-encoded-dataset')
cli.add_command(define_splits_main, name='define-splits')

if __name__ == '__main__':
    cli()
