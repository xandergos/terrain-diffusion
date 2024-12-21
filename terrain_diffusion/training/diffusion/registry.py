import catalogue
from confection import registry
from terrain_diffusion.training.datasets.datasets import *
from terrain_diffusion.data.laplacian_encoder import *
from terrain_diffusion.training.loss import SqrtLRScheduler
from terrain_diffusion.inference.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from terrain_diffusion.training.unet import EDMAutoencoder, EDMUnet2D
    

def build_registry():
    registry.scheduler = catalogue.create("confection", "schedulers", entry_points=False)
    registry.scheduler.register("edm_dpm", func=EDMDPMSolverMultistepScheduler)

    registry.model = catalogue.create("confection", "models", entry_points=False)
    registry.model.register("unet", func=EDMUnet2D)
    registry.model.register("autoencoder", func=EDMAutoencoder)
    
    registry.lr_sched = catalogue.create("confection", "lr_sched", entry_points=False)
    registry.lr_sched.register("sqrt", func=SqrtLRScheduler)

    registry.dataset = catalogue.create("confection", "datasets", entry_points=False)
    registry.dataset.register("h5_superres_terrain", func=H5SuperresTerrainDataset)
    registry.dataset.register("h5_autoencoder", func=H5AutoencoderDataset)
    registry.dataset.register("h5_latents", func=H5LatentsDataset)
    
    registry.utils = catalogue.create("confection", "utils", entry_points=False)
    registry.utils.register("create_list", func=lambda *args: list(args))
    
    registry.encoder = catalogue.create("confection", "encoder", entry_points=False)
    registry.encoder.register("laplacian_pyramid_encoder", func=LaplacianPyramidEncoder)
