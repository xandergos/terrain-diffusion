from setuptools import setup, find_packages

setup(
    name="terrain-diffusion",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "accelerate>=0.34.2",
        "catalogue>=2.0.10",
        "click>=8.1.6",
        "confection>=0.1.5",
        "diffusers>=0.30.3",
        "earthengine_api>=1.1.2",
        "easydict>=1.13",
        "ema_pytorch>=0.7.7",
        "global_land_mask>=1.0.0",
        "h5py>=3.12.1",
        "lpips>=0.1.4",
        "matplotlib>=3.9.2",
        "networkx>=3.2.1",
        "numpy>=2.1.3",
        "Pillow>=11.0.0",
        "safetensors>=0.4.5",
        "schedulefree>=1.3",
        "scipy>=1.14.1",
        "tifffile>=2024.9.20",
        "torch>=2.4.1",
        "torchvision>=0.19.1",
        "tqdm>=4.66.5",
    ],
    entry_points={
        'console_scripts': [
            'terrain-diffusion=terrain_diffusion.__main__:cli',
        ],
    },
)