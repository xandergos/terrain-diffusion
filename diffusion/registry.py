import json
import os
import catalogue
from confection import registry
from diffusion.datasets.datasets import BaseTerrainDataset, CachedTiffDataset, H5AutoencoderDataset, H5BaseTerrainDataset, MultiDataset, SuperresTerrainDataset, H5SuperresTerrainDataset
from diffusion.encoder import *
from diffusion.encoder import encode_postprocess
from diffusion.loss import SqrtLRScheduler
from diffusion.samplers.image_sampler import ImageSampler
from diffusion.samplers.stacked_sampler import StackedSampler
from diffusion.samplers.superresolution import superresolution_sampler
from diffusion.samplers.tiled import TiledSampler
from diffusion.samplers.sampler_utils import base_terrain_nocond, constant_label_network_inputs
from diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from diffusion.unet import DiffusionAutoencoder, EDMAutoencoder, EDMUnet2D, PatchGANDiscriminator
import torchvision.transforms.v2 as T

def simple_tiff_dataset(paths, image_size, crop_size, root_dir=None):
    if isinstance(paths, str):
        if os.path.isdir(paths):
            paths = [os.path.join(paths, p) for p in os.listdir(paths)]
        else:
            with open(paths) as f:
                paths = json.load(f)
    if root_dir is not None:
        paths = [os.path.join(root_dir, p) for p in paths]
    return CachedTiffDataset(paths, 
                             pretransform=T.Compose([T.Resize(image_size), T.CenterCrop(crop_size)]), 
                             posttransform=T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=False)]))

def evaluation_superres_sampler(batch_size,
                                conditioning_paths, conditioning_image_size, conditioning_crop_size, conditioning_root_dir, 
                                upsample_factor):
    """Create a superresolution sampler for evaluation during training."""
    samplers = [
        ImageSampler.from_dataset(simple_tiff_dataset(conditioning_paths, image_size=conditioning_image_size, 
                                                      crop_size=conditioning_crop_size, root_dir=conditioning_root_dir), 
                                   count=batch_size),
        
    ]
    superresolution_sampler()
    

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
    registry.dataset.register("simple_tiff", func=simple_tiff_dataset)
    registry.dataset.register("base_terrain", func=BaseTerrainDataset)
    registry.dataset.register("superresolution_terrain", func=SuperresTerrainDataset)
    registry.dataset.register("multi_dataset", func=MultiDataset)
    registry.dataset.register("h5_base_terrain", func=H5BaseTerrainDataset)
    registry.dataset.register("h5_autoencoder", func=H5AutoencoderDataset)
    registry.dataset.register("h5_superres_terrain", func=H5SuperresTerrainDataset)
    
    registry.sampler = catalogue.create("confection", "samplers", entry_points=False)
    registry.sampler.register("image", func=ImageSampler)
    registry.sampler.register("image_from_pil", func=ImageSampler.from_pil)
    registry.sampler.register("image_from_dataset", func=ImageSampler.from_dataset)
    registry.sampler.register("stacked", func=StackedSampler)
    registry.sampler.register("superresolution", func=superresolution_sampler)
    registry.sampler.register("tiled", func=TiledSampler)
    
    registry.postprocessor = catalogue.create("confection", "postprocessor", entry_points=False)
    registry.postprocessor.register("encode", func=encode_postprocess)
    registry.postprocessor.register("decode", func=decode_postprocess)
    
    registry.utils = catalogue.create("confection", "utils", entry_points=False)
    registry.utils.register("create_list", func=lambda *args: list(args))
    
    registry.encoder = catalogue.create("confection", "encoder", entry_points=False)
    registry.encoder.register("laplacian_pyramid_encoder", func=LaplacianPyramidEncoder)
    
    registry.network_inputs = catalogue.create("confection", "network_inputs", entry_points=False)
    registry.network_inputs.register("constant_label", func=constant_label_network_inputs)
    registry.network_inputs.register("base_terrain_nocond", func=base_terrain_nocond)
