import catalogue
from confection import registry
from terrain_diffusion.training.datasets.datasets import *
from terrain_diffusion.data.laplacian_encoder import *
from terrain_diffusion.training.gan.discriminator import MPDiscriminator
from terrain_diffusion.training.gan.discriminator_basic import PatchDiscriminator, ResNetDiscriminator
from terrain_diffusion.training.gan.generator import MPGenerator
from terrain_diffusion.training.loss import CosineLRScheduler, SqrtLRScheduler, ConstantLRScheduler
from terrain_diffusion.inference.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from terrain_diffusion.training.unet import EDMAutoencoder, EDMUnet2D

def build_registry():
    registry.scheduler = catalogue.create("confection", "schedulers", entry_points=False)
    registry.scheduler.register("edm_dpm", func=EDMDPMSolverMultistepScheduler)

    registry.model = catalogue.create("confection", "models", entry_points=False)
    registry.model.register("unet", func=EDMUnet2D)
    registry.model.register("autoencoder", func=EDMAutoencoder)
    registry.model.register("generator", func=MPGenerator)
    registry.model.register("discriminator", func=MPDiscriminator)
    registry.model.register("discriminator_basic", func=PatchDiscriminator)
    registry.model.register("discriminator_resnet", func=ResNetDiscriminator)
    
    registry.lr_sched = catalogue.create("confection", "lr_sched", entry_points=False)
    registry.lr_sched.register("sqrt", func=SqrtLRScheduler)
    registry.lr_sched.register("cosine", func=CosineLRScheduler)
    registry.lr_sched.register("constant", func=ConstantLRScheduler)
    
    registry.dataset = catalogue.create("confection", "datasets", entry_points=False)
    registry.dataset.register("h5_decoder_terrain", func=H5DecoderTerrainDataset)
    registry.dataset.register("h5_upsample_terrain", func=H5UpsamplingTerrainDataset)
    registry.dataset.register("h5_autoencoder", func=H5AutoencoderDataset)
    registry.dataset.register("h5_latents", func=H5LatentsDataset)
    registry.dataset.register("h5_latents_simple", func=H5LatentsSimpleDataset)
    registry.dataset.register("file_gan", func=FileGANDataset)
    
    registry.utils = catalogue.create("confection", "utils", entry_points=False)
    registry.utils.register("create_list", func=lambda *args: list(args))
