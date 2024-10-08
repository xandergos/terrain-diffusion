import catalogue
from confection import registry
from diffusion.datasets.datasets import BaseTerrainDataset, MultiDataset, UpsamplingTerrainDataset
from diffusion.encoder import *
from diffusion.encoder import encode_postprocess
from diffusion.loss import SqrtLRScheduler
from diffusion.samplers.image_sampler import ImageSampler
from diffusion.samplers.stacked_sampler import StackedSampler
from diffusion.samplers.tiled import TiledSampler, constant_label_network_inputs
from diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from diffusion.unet import EDMUnet2D

def build_registry():
    registry.scheduler = catalogue.create("confection", "schedulers", entry_points=False)
    registry.scheduler.register("edm_dpm", func=EDMDPMSolverMultistepScheduler)

    registry.model = catalogue.create("confection", "models", entry_points=False)
    registry.model.register("unet", func=EDMUnet2D)

    registry.lr_sched = catalogue.create("confection", "lr_sched", entry_points=False)
    registry.lr_sched.register("sqrt", func=SqrtLRScheduler)

    registry.dataset = catalogue.create("confection", "datasets", entry_points=False)
    registry.dataset.register("base_terrain", func=BaseTerrainDataset)
    registry.dataset.register("upsampling_terrain", func=UpsamplingTerrainDataset)
    registry.dataset.register("multi_dataset", func=MultiDataset)
    
    registry.sampler = catalogue.create("confection", "samplers", entry_points=False)
    registry.sampler.register("image", func=ImageSampler)
    registry.sampler.register("image_from_pil", func=ImageSampler.from_pil)
    registry.sampler.register("stacked", func=StackedSampler)
    
    registry.postprocessor = catalogue.create("confection", "postprocessor", entry_points=False)
    registry.postprocessor.register("encode", func=encode_postprocess)
    registry.postprocessor.register("decode", func=decode_postprocess)
    
    registry.encoder = catalogue.create("confection", "encoder", entry_points=False)
    registry.encoder.register("laplacian_pyramid_encoder", func=LaplacianPyramidEncoder)
    
    registry.network_inputs = catalogue.create("confection", "network_inputs", entry_points=False)
    registry.network_inputs.register("constant_label", func=constant_label_network_inputs)

