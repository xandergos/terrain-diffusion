import json
import os
import catalogue
from confection import registry
from training.datasets.datasets import BaseTerrainDataset, CachedTiffDataset, H5AutoencoderDataset, H5BaseTerrainDataset, MultiDataset, SuperresTerrainDataset, H5SuperresTerrainDataset
from data.laplacian_encoder import *
from data.laplacian_encoder import encode_postprocess
from training.loss import SqrtLRScheduler
from inference.samplers.image_sampler import ImageSampler
from inference.samplers.stacked_sampler import StackedSampler
from inference.samplers.superresolution import superresolution_sampler
from inference.samplers.tiled import TiledSampler
from inference.samplers.sampler_utils import base_terrain_nocond, constant_label_network_inputs
from inference.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from training.diffusion.unet import DiffusionAutoencoder, EDMAutoencoder, EDMUnet2D, PatchGANDiscriminator
import torchvision.transforms.v2 as T
    

def build_registry():
    registry.scheduler = catalogue.create("confection", "schedulers", entry_points=False)
    registry.scheduler.register("edm_dpm", func=EDMDPMSolverMultistepScheduler)

    registry.model = catalogue.create("confection", "models", entry_points=False)
    registry.model.register("unet", func=EDMUnet2D)
    registry.model.register("autoencoder", func=EDMAutoencoder)
    registry.model.register("diffusion_autoencoder", func=DiffusionAutoencoder)
    registry.model.register("patchgan_discriminator", func=PatchGANDiscriminator)
    
    registry.lr_sched = catalogue.create("confection", "lr_sched", entry_points=False)
    registry.lr_sched.register("sqrt", func=SqrtLRScheduler)

    registry.dataset = catalogue.create("confection", "datasets", entry_points=False)
    registry.dataset.register("base_terrain", func=BaseTerrainDataset)
    registry.dataset.register("superresolution_terrain", func=SuperresTerrainDataset)
    registry.dataset.register("multi_dataset", func=MultiDataset)
    registry.dataset.register("h5_base_terrain", func=H5BaseTerrainDataset)
    registry.dataset.register("h5_autoencoder", func=H5AutoencoderDataset)
    registry.dataset.register("h5_superres_terrain", func=H5SuperresTerrainDataset)
    
    registry.utils = catalogue.create("confection", "utils", entry_points=False)
    registry.utils.register("create_list", func=lambda *args: list(args))
    
    registry.encoder = catalogue.create("confection", "encoder", entry_points=False)
    registry.encoder.register("laplacian_pyramid_encoder", func=LaplacianPyramidEncoder)
