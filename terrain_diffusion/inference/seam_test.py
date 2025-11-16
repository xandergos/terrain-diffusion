from terrain_diffusion.inference.world_pipeline import WorldPipeline
import numpy as np
import matplotlib.pyplot as plt

def main():
    x = WorldPipeline(
        'world_mc.h5', device='cuda', seed=1, log_mode='debug',
        drop_water_pct=0.5,
        frequency_mult=[2.0, 2.0, 2.0, 2.0, 2.0],
        cond_snr=[0.5, 0.5, 0.5, 0.5, 0.5],
        histogram_raw=[0.0, 0.0, 0.0, 2.0, 2.0],
        mode="a",
    )
    
    arr = np.zeros((2048, 2048), dtype=np.float32)
    
    for i in range(0, 2048, 128):
        for j in range(0, 2048, 128):
            arr[i:i+128, j:j+128] = x.get_22(i, j, i+128, j+128, with_climate=True)['climate'][0]
    
    corner = arr[1023:1025, 1023:1025]
    print(corner)
        
        
main()