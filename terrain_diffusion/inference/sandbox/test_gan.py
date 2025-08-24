import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from confection import Config, registry
from ema_pytorch import PostHocEMA
from terrain_diffusion.training.registry import build_registry

# FID processing constants used to denorm for visualization
MEAN = -2607
STD = 2435
MIN_ELEVATION = -10000
MAX_ELEVATION = 9000
ELEVATION_RANGE = MAX_ELEVATION - MIN_ELEVATION

def main():
  config_path = "configs/gan/gan-simple.cfg"
  device = "cuda" if torch.cuda.is_available() else "cpu"

  build_registry()
  cfg = Config().from_disk(config_path)
  resolved = registry.resolve(cfg, validate=False)

  gen = resolved["generator"].to(device).eval()

  # Prepare EMA and load its state
  ckpt_dir = os.path.join(resolved["logging"]["save_dir"], "latest_checkpoint")
  ema_path = os.path.join(ckpt_dir, "phema.pt")
  assert os.path.exists(ema_path), f"Missing EMA checkpoint at {ema_path}. Train first or adjust path."

  resolved["ema"]["checkpoint_folder"] = os.path.join(resolved["logging"]["save_dir"], "phema")
  ema = PostHocEMA(gen, **resolved["ema"]).to(device)
  ema.load_state_dict(torch.load(ema_path, map_location=device))

  # Copy synthesized EMA weights into generator
  ema.synthesize_ema_model(sigma_rel=0.15).copy_params_from_ema_to_model()

  # Sample a single image
  z = torch.randn(
    1,
    cfg["generator"]["latent_channels"],
    256,
    256,
    device=device
  )

  with torch.no_grad():
    sample = gen(z)[:, :1]  # first (elevation) channel, shape [1, 1, H, W]

  img = sample[0, 0].detach().cpu()

  # Denormalize to elevation and map to [0,1] for display
  vis = torch.clamp(img * STD + MEAN, MIN_ELEVATION, MAX_ELEVATION)
  vis = vis.numpy()
  vis = np.sign(vis) * np.sqrt(np.abs(vis))

  plt.figure(figsize=(4, 4))
  plt.imshow(vis, cmap="terrain")
  plt.title("GAN sample (EMA)")
  plt.axis("off")
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  main()
